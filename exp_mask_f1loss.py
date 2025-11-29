"""
Experiment: Masking multiple nodes per graph sample with SOFT MACRO F1 LOSS
- Directly optimizes macro F1 score using soft probabilities
- Uses temperature-scaled softmax for smoother gradients
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import TAGConv, GraphNorm, DataParallel as GeoDataParallel
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score, classification_report
import random
import pickle
import os

# ============== Experiment Name ==============
EXP_NAME = 'exp_mask_f1loss'

# ============== Soft Macro F1 Loss ==============
class SoftMacroF1Loss(nn.Module):
    """
    Differentiable approximation of macro F1 loss.
    Uses soft predictions to compute TP, FP, FN.
    """
    def __init__(self, epsilon: float = 1e-7, temperature: float = 1.0):
        super().__init__()
        self.epsilon = epsilon
        self.temperature = temperature

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)

        # Temperature-scaled softmax for smoother gradients
        probs = F.softmax(logits / self.temperature, dim=-1)
        num_classes = probs.size(-1)

        targets_onehot = F.one_hot(targets.long(), num_classes=num_classes).float()

        probs_flat = probs.reshape(-1, num_classes)
        targets_flat = targets_onehot.reshape(-1, num_classes)

        # Soft TP, FP, FN
        tp = (probs_flat * targets_flat).sum(dim=0)
        fp = (probs_flat * (1 - targets_flat)).sum(dim=0)
        fn = ((1 - probs_flat) * targets_flat).sum(dim=0)

        # Soft F1 per class
        soft_f1 = (2 * tp + self.epsilon) / (2 * tp + fp + fn + self.epsilon)

        # Macro F1 = mean across classes
        macro_f1 = soft_f1.mean()
        loss = 1.0 - macro_f1
        return loss


# ============== Load Precomputed Splits ==============
print("Loading precomputed splits...")
SPLITS_DIR = 'splits'

if not os.path.exists(SPLITS_DIR):
    raise RuntimeError(f"Splits directory '{SPLITS_DIR}' not found. Run 'python precompute_splits.py' first.")

train_nodes = np.load(os.path.join(SPLITS_DIR, 'train_nodes.npy')).tolist()
val_nodes = np.load(os.path.join(SPLITS_DIR, 'val_nodes.npy')).tolist()
test_nodes = np.load(os.path.join(SPLITS_DIR, 'test_nodes.npy')).tolist()
unknown_nodes = np.load(os.path.join(SPLITS_DIR, 'unknown_nodes.npy')).tolist()
node_labels = np.load(os.path.join(SPLITS_DIR, 'node_labels_masked.npy'))  # Test nodes are masked!

with open(os.path.join(SPLITS_DIR, 'labels.txt'), 'r') as f:
    labels = [line.strip() for line in f]

with open(os.path.join(SPLITS_DIR, 'metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f)

max_node = metadata['max_node']
print(f"Labels: {labels}")
print(f"Train nodes: {len(train_nodes)}, Val nodes: {len(val_nodes)}, Test nodes: {len(test_nodes)}")

# Create node_class matrix from masked labels (test nodes already masked as -1)
node_class = np.zeros((max_node + 1, len(labels)))
for n in range(max_node + 1):
    if node_labels[n] >= 0:
        node_class[n, node_labels[n]] = 1
    else:
        node_class[n, :] = np.ones(len(labels)) / len(labels)  # Uniform for masked/unknown

# ============== Load Edge Data ==============
print("Loading edge data...")
data = pd.read_csv('CR_real_masks_more_labeled_veritices_agreed.csv')
data['node_id1'] -= 1
data['node_id2'] -= 1

# ============== Device Setup ==============
DEVICE_CHOICE = "cuda"
device = torch.device(DEVICE_CHOICE if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
multi_gpu = (device.type == "cuda" and num_gpus >= 2)
print(f"Device: {device}, GPUs: {num_gpus}, Multi-GPU: {multi_gpu}")

# Unknown nodes subset
random.seed(42)
unknown_nodes_shuffled = unknown_nodes.copy()
random.shuffle(unknown_nodes_shuffled)
NUM_UNKNOWN_FRACTION = 0.05
num_unknown_to_use = int(len(unknown_nodes_shuffled) * NUM_UNKNOWN_FRACTION)
unknown_nodes_subset = unknown_nodes_shuffled[:num_unknown_to_use]
print(f"Using {num_unknown_to_use} unknown nodes ({NUM_UNKNOWN_FRACTION*100:.0f}%)")

# ============== Masking Config ==============
MASK_COUNT = 64
FEATURE_TYPE = 'onehot'
F1_TEMPERATURE = 1.0  # Temperature for soft F1 loss (higher = smoother gradients)

def compute_graph_based_features(sorted_nodes, subgraph_edges, node_class, num_classes):
    num_nodes = len(sorted_nodes)
    node_mapping = {node: i for i, node in enumerate(sorted_nodes)}
    node_labels_arr = np.array([np.argmax(node_class[n]) if np.max(node_class[n]) > 0.5 else -1 for n in sorted_nodes])
    features = np.zeros((num_nodes, 5 * num_classes))
    
    for _, row in subgraph_edges.iterrows():
        n1, n2, w = row['node_id1'], row['node_id2'], row['ibd_sum']
        if n1 in node_mapping and n2 in node_mapping:
            i, j = node_mapping[n1], node_mapping[n2]
            if node_labels_arr[j] >= 0:
                c = node_labels_arr[j]
                features[i, c] += 1
                features[i, num_classes + c] += w
                features[i, 3*num_classes + c] = max(features[i, 3*num_classes + c], w)
                features[i, 4*num_classes + c] += 1
            if node_labels_arr[i] >= 0:
                c = node_labels_arr[i]
                features[j, c] += 1
                features[j, num_classes + c] += w
                features[j, 3*num_classes + c] = max(features[j, 3*num_classes + c], w)
                features[j, 4*num_classes + c] += 1
    
    for c in range(num_classes):
        mask = features[:, c] > 0
        features[mask, num_classes + c] /= features[mask, c]
    
    return torch.tensor(features, dtype=torch.float)


class MaskedGraphDataset(Dataset):
    def __init__(self, data_df, unknown_nodes_subset, train_nodes, val_nodes, test_nodes, 
                 split='train', mask_count=64, num_samples=500, return_node_ids=False):
        super().__init__()
        self.data_df = data_df
        self.unknown_nodes_subset = unknown_nodes_subset
        self.train_nodes = train_nodes
        self.val_nodes = val_nodes
        self.test_nodes = test_nodes
        self.split = split
        self.mask_count = mask_count
        self.num_samples = num_samples
        self.return_node_ids = return_node_ids
        
        if split == 'train':
            self.target_nodes = train_nodes
        elif split == 'val':
            self.target_nodes = val_nodes
        else:
            self.target_nodes = test_nodes
    
    def len(self):
        if self.split == 'train':
            return self.num_samples
        else:
            return (len(self.target_nodes) + self.mask_count - 1) // self.mask_count
    
    def get(self, idx):
        nodes_to_include = self.train_nodes.copy()
        nodes_to_include.extend(self.unknown_nodes_subset)
        
        if self.split == 'train':
            if len(self.train_nodes) < self.mask_count:
                mask_nodes = self.train_nodes.copy()
            else:
                mask_nodes = random.sample(self.train_nodes, self.mask_count)
        else:
            start_idx = idx * self.mask_count
            end_idx = min(start_idx + self.mask_count, len(self.target_nodes))
            mask_nodes = self.target_nodes[start_idx:end_idx]
            for n in mask_nodes:
                if n not in nodes_to_include:
                    nodes_to_include.append(n)
        
        nodes_set = set(nodes_to_include)
        mask = (self.data_df['node_id1'].isin(nodes_set) & self.data_df['node_id2'].isin(nodes_set))
        subgraph_edges = self.data_df[mask]
        
        if len(subgraph_edges) == 0:
            return None
        
        sorted_nodes = sorted(nodes_to_include)
        node_mapping = {node: i for i, node in enumerate(sorted_nodes)}
        
        src = [node_mapping[n] for n in subgraph_edges['node_id1'].values]
        dst = [node_mapping[n] for n in subgraph_edges['node_id2'].values]
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        
        if FEATURE_TYPE == 'graph_based':
            node_features = compute_graph_based_features(sorted_nodes, subgraph_edges, node_class, len(labels))
        else:
            node_features = torch.tensor(node_class[sorted_nodes], dtype=torch.float)
        
        mask_indices = [node_mapping[n] for n in mask_nodes if n in node_mapping]
        predict_mask = torch.zeros(len(sorted_nodes), dtype=torch.bool)
        predict_mask[mask_indices] = True
        
        node_features_masked = node_features.clone()
        if FEATURE_TYPE == 'graph_based':
            node_features_masked[predict_mask] = torch.zeros(5 * len(labels))
        else:
            node_features_masked[predict_mask] = torch.ones(len(labels)) / len(labels)
        
        onehot = torch.tensor(node_class[sorted_nodes], dtype=torch.float)
        y = torch.tensor([torch.argmax(onehot[i]).item() for i in range(len(sorted_nodes))], dtype=torch.long)
        
        ibd = subgraph_edges['ibd_sum'].values
        edge_weights = torch.tensor(list(ibd) + list(ibd), dtype=torch.float)
        
        if self.return_node_ids:
            masked_node_ids = torch.tensor([mask_nodes[mask_indices.index(i)] for i in mask_indices], dtype=torch.long)
            return Data(
                x=node_features_masked,
                edge_index=edge_index,
                y=y,
                weight=edge_weights,
                num_classes=len(labels),
                predict_mask=predict_mask,
                masked_node_ids=masked_node_ids
            )
        
        return Data(
            x=node_features_masked,
            edge_index=edge_index,
            y=y,
            weight=edge_weights,
            num_classes=len(labels),
            predict_mask=predict_mask
        )


# ============== Model ==============
class TAGConvModel(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=512):
        super().__init__()
        self.first_linear = torch.nn.Linear(num_features, hidden_dim)
        self.conv1 = TAGConv(hidden_dim, hidden_dim)
        self.conv2 = TAGConv(hidden_dim, hidden_dim)
        self.conv3 = TAGConv(hidden_dim, hidden_dim)
        self.n1 = GraphNorm(hidden_dim)
        self.n2 = GraphNorm(hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.weight
        x = self.first_linear(x)
        x_ = x.clone()
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = x_ + x
        x = self.n1(x)
        x = F.elu(self.conv2(x, edge_index, edge_weight))
        x = x_ + x
        x = self.n2(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = x_ + x
        return self.linear(x)


class SingleDeviceWrapper(torch.nn.Module):
    def __init__(self, module, device):
        super().__init__()
        self.module = module
        self.device = device

    def forward(self, data_list):
        if isinstance(data_list, list):
            batch = Batch.from_data_list(data_list).to(self.device, non_blocking=True)
        else:
            batch = data_list.to(self.device, non_blocking=True)
        return self.module(batch), batch


# ============== Datasets ==============
train_dataset = MaskedGraphDataset(data, unknown_nodes_subset, train_nodes, val_nodes, test_nodes,
                                    split='train', mask_count=MASK_COUNT, num_samples=500)
val_dataset = MaskedGraphDataset(data, unknown_nodes_subset, train_nodes, val_nodes, test_nodes,
                                  split='val', mask_count=MASK_COUNT)
test_dataset = MaskedGraphDataset(data, unknown_nodes_subset, train_nodes, val_nodes, test_nodes,
                                   split='test', mask_count=MASK_COUNT, return_node_ids=True)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

# ============== Model Setup ==============
num_features = 5 * len(labels) if FEATURE_TYPE == 'graph_based' else len(labels)
base_model = TAGConvModel(num_features=num_features, num_classes=len(labels)).to(
    torch.device("cuda:0") if device.type == "cuda" else device
)

if multi_gpu:
    model = GeoDataParallel(base_model, device_ids=list(range(num_gpus)))
    print(f"Multi-GPU mode: {num_gpus} GPUs")
else:
    model = SingleDeviceWrapper(base_model, torch.device("cuda:0") if device.type == "cuda" else device)
    print("Single GPU mode")

print(f"Features: {FEATURE_TYPE} ({num_features}), Params: {sum(p.numel() for p in model.parameters()):,}")


# ============== Evaluation ==============
def evaluate(model, loader, use_amp=True):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval', leave=False):
            with torch.amp.autocast('cuda', enabled=use_amp):
                if isinstance(model, SingleDeviceWrapper):
                    logits, batch_data = model(batch)
                else:
                    batch_data = Batch.from_data_list(batch).to(device)
                    logits = model.module(batch_data)
                
                    predict_mask = batch_data.predict_mask
            
            preds = torch.argmax(logits[predict_mask], dim=-1).cpu().tolist()
            trues = batch_data.y[predict_mask].cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(trues)
    return f1_score(y_true, y_pred, average='macro'), y_true, y_pred


def generate_submission(model, loader, output_path, use_amp=True):
    """Generate submission file with predictions for test nodes."""
    model.eval()
    all_node_ids = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Generating submission', leave=False):
            with torch.amp.autocast('cuda', enabled=use_amp):
                if isinstance(model, SingleDeviceWrapper):
                    logits, batch_data = model(batch)
                else:
                    batch_data = Batch.from_data_list(batch).to(device)
                    logits = model.module(batch_data)
                
                predict_mask = batch_data.predict_mask
                masked_node_ids = batch_data.masked_node_ids
            
            preds = torch.argmax(logits[predict_mask], dim=-1).cpu().tolist()
            node_ids = masked_node_ids.cpu().tolist()
            
            all_node_ids.extend(node_ids)
            all_predictions.extend(preds)
    
    submission_df = pd.DataFrame({
        'node_id': all_node_ids,
        'predicted_label': [labels[p] for p in all_predictions]
    })
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    return submission_df


# ============== Training ==============
LR, WD, EPOCHS, PATIENCE = 0.0001, 0.0001, 10, 5
BATCH_SIZE = 1

train_loader = DataListLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataListLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataListLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Use Soft Macro F1 Loss
criterion = SoftMacroF1Loss(temperature=F1_TEMPERATURE)
print(f"Using SoftMacroF1Loss with temperature={F1_TEMPERATURE}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = StepLR(optimizer, step_size=50, gamma=0.95)

scaler = torch.amp.GradScaler('cuda')
use_amp = device.type == 'cuda'

best_val_f1, patience_counter, best_state = 0.0, 0, None

for epoch in range(1, EPOCHS + 1):
    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch-1}")
        break
    
    model.train()
    losses = []
    loop = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    
    for batch_idx, batch in enumerate(loop):
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            if isinstance(model, SingleDeviceWrapper):
                logits, batch_data = model(batch)
            else:
                batch_data = Batch.from_data_list(batch).to(device)
                logits = model.module(batch_data)
            
            predict_mask = batch_data.predict_mask
            loss = criterion(logits[predict_mask], batch_data.y[predict_mask])
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.append(loss.item())
        
        if batch_idx % 50 == 0:
            loop.set_postfix(loss=f"{loss.item():.4f}")
            torch.cuda.empty_cache()
    
    val_f1, _, _ = evaluate(model, val_loader, use_amp)
    
    if val_f1 > best_val_f1:
        best_val_f1, patience_counter = val_f1, 0
        to_save = getattr(model, 'module', model)
        best_state = {k: v.cpu().clone() for k, v in to_save.state_dict().items()}
        print(f"[Epoch {epoch}] val_f1={best_val_f1:.4f} ↑ | loss={np.mean(losses):.4f}")
    else:
        patience_counter += 1
        print(f"[Epoch {epoch}] val_f1={val_f1:.4f} | loss={np.mean(losses):.4f} | patience={patience_counter}")

print(f"\nBest val F1: {best_val_f1:.4f}")

if best_state:
    to_load = getattr(model, 'module', model)
    to_load.load_state_dict(best_state)
    print("Loaded best model state")

test_f1, y_true, y_pred = evaluate(model, test_loader, use_amp)
print(f"Test F1: {test_f1:.4f}")
print(classification_report(y_true, y_pred, target_names=labels, digits=4))

# ============== Generate Submission ==============
os.makedirs('submissions', exist_ok=True)
submission_path = f'submissions/{EXP_NAME}.csv'
generate_submission(model, test_loader, submission_path, use_amp)
print(f"\nTo score: python score.py {submission_path}")
