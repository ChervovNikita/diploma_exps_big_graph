"""
Experiment: Trainable node embeddings instead of one-hot encoding
- Instead of using labels as features (OHE or graph-based), create a small trainable embedding (32-dim) for each node
- The embedding is learned end-to-end with the GNN
- This allows the model to learn node representations without explicit label information
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
EXP_NAME = 'exp_embed'

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

# ============== Config ==============
EMBED_DIM = 32  # Trainable embedding dimension


class EmbedGraphDataset(Dataset):
    """
    Dataset that returns node indices for embedding lookup instead of features.
    The model will use a trainable embedding table.
    """
    def __init__(self, data_df, unknown_nodes_subset, train_nodes, val_nodes, test_nodes, 
                 split='train', return_node_ids=False):
        super().__init__()
        self.data_df = data_df
        self.unknown_nodes_subset = unknown_nodes_subset
        self.train_nodes = train_nodes
        self.val_nodes = val_nodes
        self.test_nodes = test_nodes
        self.split = split
        self.return_node_ids = return_node_ids
        self.target_nodes = {'train': train_nodes, 'val': val_nodes, 'test': test_nodes}[split]
    
    def len(self):
        return len(self.target_nodes)
    
    def get(self, idx):
        target_node = self.target_nodes[idx]
        
        nodes_to_include = self.train_nodes.copy()
        if self.split != 'train' and target_node not in nodes_to_include:
            nodes_to_include.append(target_node)
        nodes_to_include.extend(self.unknown_nodes_subset)
        
        nodes_set = set(nodes_to_include)
        mask = (self.data_df['node_id1'].isin(nodes_set) & self.data_df['node_id2'].isin(nodes_set))
        subgraph_edges = self.data_df[mask]
        
        if len(subgraph_edges) == 0:
            return None
        
        sorted_nodes = sorted(nodes_to_include)
        node_mapping = {node: i for i, node in enumerate(sorted_nodes)}
        
        # Undirected graph
        src = [node_mapping[n] for n in subgraph_edges['node_id1'].values]
        dst = [node_mapping[n] for n in subgraph_edges['node_id2'].values]
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        
        # Store original node IDs for embedding lookup
        node_ids = torch.tensor(sorted_nodes, dtype=torch.long)
        
        target_idx = node_mapping[target_node]
        
        # Labels
        onehot = torch.tensor(node_class[sorted_nodes], dtype=torch.float)
        y = torch.tensor([torch.argmax(onehot[i]).item() for i in range(len(sorted_nodes))], dtype=torch.long)
        target_label = y[target_idx].clone()
        
        # Edge weights
        ibd = subgraph_edges['ibd_sum'].values
        edge_weights = torch.tensor(list(ibd) + list(ibd), dtype=torch.float)
        
        if self.return_node_ids:
            return Data(
                node_ids=node_ids,  # Original node IDs for embedding lookup
                edge_index=edge_index,
                y=y,
                weight=edge_weights,
                num_classes=len(labels),
                target_node_idx=torch.tensor(target_idx, dtype=torch.long),
                target_label=target_label,
                target_node_id=torch.tensor(target_node, dtype=torch.long)  # For submission
            )
        
        return Data(
            node_ids=node_ids,  # Original node IDs for embedding lookup
            edge_index=edge_index,
            y=y,
            weight=edge_weights,
            num_classes=len(labels),
            target_node_idx=torch.tensor(target_idx, dtype=torch.long),
            target_label=target_label
        )


# ============== Model with Trainable Embeddings ==============
class NodeEmbedding(nn.Module):
    """Trainable node embedding table."""
    def __init__(self, num_nodes, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        # Initialize with small random values
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
    
    def forward(self, node_ids):
        return self.embedding(node_ids)


class TAGConvModelWithEmbed(nn.Module):
    def __init__(self, num_nodes, embed_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.node_embed = NodeEmbedding(num_nodes, embed_dim)
        self.first_linear = nn.Linear(embed_dim, hidden_dim)
        self.conv1 = TAGConv(hidden_dim, hidden_dim)
        self.conv2 = TAGConv(hidden_dim, hidden_dim)
        self.conv3 = TAGConv(hidden_dim, hidden_dim)
        self.n1 = GraphNorm(hidden_dim)
        self.n2 = GraphNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        node_ids, edge_index, edge_weight = data.node_ids, data.edge_index, data.weight
        
        # Get embeddings for nodes
        x = self.node_embed(node_ids)
        
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


class SingleDeviceWrapper(nn.Module):
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
train_dataset = EmbedGraphDataset(data, unknown_nodes_subset, train_nodes, val_nodes, test_nodes, split='train')
val_dataset = EmbedGraphDataset(data, unknown_nodes_subset, train_nodes, val_nodes, test_nodes, split='val')
test_dataset = EmbedGraphDataset(data, unknown_nodes_subset, train_nodes, val_nodes, test_nodes, split='test', return_node_ids=True)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# ============== Model Setup ==============
num_nodes = max_node + 1  # Total number of possible nodes
base_model = TAGConvModelWithEmbed(
    num_nodes=num_nodes, 
    embed_dim=EMBED_DIM, 
    num_classes=len(labels)
).to(torch.device("cuda:0") if device.type == "cuda" else device)

if multi_gpu:
    model = GeoDataParallel(base_model, device_ids=list(range(num_gpus)))
    print(f"Multi-GPU mode: {num_gpus} GPUs")
else:
    model = SingleDeviceWrapper(base_model, torch.device("cuda:0") if device.type == "cuda" else device)
    print("Single GPU mode")

embed_params = sum(p.numel() for p in base_model.node_embed.parameters())
total_params = sum(p.numel() for p in model.parameters())
print(f"Embed dim: {EMBED_DIM}, Embed params: {embed_params:,}, Total params: {total_params:,}")


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
            
            sizes = [int(d.num_nodes) for d in batch]
            ptr = torch.as_tensor(np.cumsum([0] + sizes), device=logits.device)
            target_indices = torch.as_tensor(
                [ptr[i].item() + int(d.target_node_idx) for i, d in enumerate(batch)],
                device=logits.device
            )
            preds = torch.argmax(logits[target_indices], dim=-1).tolist()
            trues = [int(d.target_label) for d in batch]
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
            
            sizes = [int(d.num_nodes) for d in batch]
            ptr = torch.as_tensor(np.cumsum([0] + sizes), device=logits.device)
            target_indices = torch.as_tensor(
                [ptr[i].item() + int(d.target_node_idx) for i, d in enumerate(batch)],
                device=logits.device
            )
            preds = torch.argmax(logits[target_indices], dim=-1).cpu().tolist()
            node_ids = [int(d.target_node_id) for d in batch]
            
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
TRAIN_BATCH_SIZE = num_gpus if multi_gpu else 1
EVAL_BATCH_SIZE = num_gpus if multi_gpu else 1

num_workers = min(4, os.cpu_count() or 2)
pin_memory = (device.type == "cuda")

train_loader = DataListLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                               num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataListLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
test_loader = DataListLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

class_counts = [sum(1 for n in train_nodes if node_labels[n] == c) for c in range(len(labels))]
class_weights = torch.tensor([max(class_counts) / c for c in class_counts], dtype=torch.float).to(torch.device("cuda:0"))
print(f"Class weights: {dict(zip(labels, [f'{w:.2f}' for w in class_weights.tolist()]))}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = StepLR(optimizer, step_size=50, gamma=0.95)

scaler = torch.amp.GradScaler('cuda')
use_amp = device.type == 'cuda'
print(f"AMP enabled: {use_amp}")

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
                out, batch_data = model(batch)
            else:
                batch_data = Batch.from_data_list(batch).to(device)
                out = model.module(batch_data)
            
            sizes = [int(d.num_nodes) for d in batch]
            ptr = torch.as_tensor(np.cumsum([0] + sizes), device=out.device)
            target_indices = torch.as_tensor(
                [ptr[i].item() + int(d.target_node_idx) for i, d in enumerate(batch)],
                device=out.device
            )
            targets = torch.as_tensor([int(d.target_label) for d in batch], device=out.device)
            
            loss = criterion(out[target_indices], targets)
        
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
