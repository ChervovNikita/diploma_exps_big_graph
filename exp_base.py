import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import TAGConv, GraphNorm, DataParallel as GeoDataParallel
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import random
import gc
import os


data = pd.read_csv('CR_real_masks_more_labeled_veritices_agreed.csv')
data['node_id1'] -= 1
data['node_id2'] -= 1

labels = data['label_id1'].unique().tolist() + data['label_id2'].unique().tolist()
labels = list(set(labels))
print(labels)

labels = [label for label in labels if label != 'masked']

df = pd.concat([data[['node_id1', 'label_id1']].rename(columns={'node_id1': 'node_id', 'label_id1': 'label'}), data[['node_id2', 'label_id2']].rename(columns={'node_id2': 'node_id', 'label_id2': 'label'})], ignore_index=True)
df = df.drop_duplicates(subset=['node_id'])

max_node = max(data['node_id1'].max(), data['node_id2'].max())
node_class = np.zeros((max_node+1, len(labels)))
known_nodes = []
unknown_nodes = []
good = np.zeros(max_node+1)
for _, row in tqdm(df.iterrows()):
    if row['label'] == 'masked':
        node_class[row['node_id'], :] = np.ones(4) / 4
        good[row['node_id']] = 1
        unknown_nodes.append(row['node_id'])
    else:
        node_class[row['node_id'], labels.index(row['label'])] = 1
        good[row['node_id']] = 1
        known_nodes.append(row['node_id'])
known_nodes = list(set(known_nodes))
unknown_nodes = list(set(unknown_nodes))

train_nodes, temp_nodes = train_test_split(known_nodes, test_size=0.4, random_state=42)
val_nodes, test_nodes = train_test_split(temp_nodes, test_size=0.5, random_state=42)

print(f"Train nodes: {len(train_nodes)}")
print(f"Val nodes: {len(val_nodes)}")
print(f"Test nodes: {len(test_nodes)}")

# Multi-GPU setup
DEVICE_CHOICE = "cuda"
device = torch.device(DEVICE_CHOICE if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
multi_gpu = (device.type == "cuda" and num_gpus >= 2)
print(f"Device: {device}, GPUs available: {num_gpus}, Multi-GPU: {multi_gpu}")

# Shuffle unknown nodes with seed for reproducibility
random.seed(42)
unknown_nodes_shuffled = unknown_nodes.copy()
random.shuffle(unknown_nodes_shuffled)

# Use 25% of unknown nodes
NUM_UNKNOWN_FRACTION = 0.05
num_unknown_to_use = int(len(unknown_nodes_shuffled) * NUM_UNKNOWN_FRACTION)
unknown_nodes_subset = unknown_nodes_shuffled[:num_unknown_to_use]
print(f"Using {num_unknown_to_use} unknown nodes ({NUM_UNKNOWN_FRACTION*100:.0f}% of {len(unknown_nodes)})")

# Feature type: 'onehot' or 'graph_based'
FEATURE_TYPE = 'onehot'

def compute_graph_based_features(sorted_nodes, subgraph_edges, node_class, num_classes):
    """
    Compute graph-based features per GENLINK paper (Eq. 7):
    For each node i and class c: n_i,c, w̄_i,c, σ_i,c, w_max_i,c, IBD_i,c
    Returns: (num_nodes, 5 * num_classes) tensor
    """
    num_nodes = len(sorted_nodes)
    node_mapping = {node: i for i, node in enumerate(sorted_nodes)}
    
    # Get node labels (only labeled nodes contribute) - vectorized
    node_class_subset = node_class[sorted_nodes]
    node_labels = np.argmax(node_class_subset, axis=1)
    node_labels[np.max(node_class_subset, axis=1) <= 0.5] = -1
    
    features = np.zeros((num_nodes, 5 * num_classes), dtype=np.float32)
    
    # Use dict of lists for better performance than nested lists
    from collections import defaultdict
    weights_per_node_class = defaultdict(list)
    
    # Vectorized edge processing
    edges = subgraph_edges[['node_id1', 'node_id2', 'ibd_sum']].values
    
    for n1, n2, w in edges:
        if n1 in node_mapping and n2 in node_mapping:
            i, j = node_mapping[n1], node_mapping[n2]
            c_j = node_labels[j]
            c_i = node_labels[i]
            
            # Only labeled neighbors contribute (per paper)
            if c_j >= 0:
                features[i, c_j] += 1  # n_i,c: neighbor count
                features[i, num_classes + c_j] += w  # sum for w̄_i,c (will be divided to get average)
                features[i, 3*num_classes + c_j] = max(features[i, 3*num_classes + c_j], w)  # w_max_i,c
                features[i, 4*num_classes + c_j] += w  # IBD_i,c: total IBD sum to class c
                weights_per_node_class[(i, c_j)].append(w)
            
            if c_i >= 0:
                features[j, c_i] += 1
                features[j, num_classes + c_i] += w
                features[j, 3*num_classes + c_i] = max(features[j, 3*num_classes + c_i], w)
                features[j, 4*num_classes + c_i] += w  # IBD_i,c: total IBD sum to class c
                weights_per_node_class[(j, c_i)].append(w)
    
    # Vectorized average computation
    count_mask = features[:, :num_classes] > 0
    features[:, num_classes:2*num_classes][count_mask] /= features[:, :num_classes][count_mask]
    
    # Compute std only where needed
    for (i, c), weights in weights_per_node_class.items():
        if len(weights) > 1:
            features[i, 2*num_classes + c] = np.std(weights, dtype=np.float32)
    
    return torch.tensor(features, dtype=torch.float32)

class GraphDataset(Dataset):
    def __init__(self, data_df, unknown_nodes_subset, train_nodes, val_nodes, test_nodes, split='train'):
        super().__init__()
        self.data_df = data_df
        self.unknown_nodes_subset = unknown_nodes_subset
        self.train_nodes = train_nodes
        self.val_nodes = val_nodes
        self.test_nodes = test_nodes
        self.split = split
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

        # Node features
        if FEATURE_TYPE == 'graph_based':
            node_features = compute_graph_based_features(sorted_nodes, subgraph_edges, node_class, len(labels))
        else:
            node_features = torch.tensor(node_class[sorted_nodes], dtype=torch.float)
        
        target_idx = node_mapping[target_node]
        
        # Mask target node
        node_features_masked = node_features.clone()
        if FEATURE_TYPE == 'graph_based':
            node_features_masked[target_idx] = torch.zeros(5 * len(labels))
        else:
            node_features_masked[target_idx] = torch.ones(len(labels)) / len(labels)
        
        # Labels
        onehot = torch.tensor(node_class[sorted_nodes], dtype=torch.float)
        y = torch.tensor([torch.argmax(onehot[i]).item() for i in range(len(sorted_nodes))], dtype=torch.long)
        target_label = y[target_idx].clone()
        
        # Edge weights
        ibd = subgraph_edges['ibd_sum'].values
        edge_weights = torch.tensor(list(ibd) + list(ibd), dtype=torch.float)

        return Data(
            x=node_features_masked, 
            edge_index=edge_index, 
            y=y,
            weight=edge_weights,
            num_classes=len(labels),
            target_node_idx=torch.tensor(target_idx, dtype=torch.long),
            target_label=target_label
        )

train_dataset = GraphDataset(data, unknown_nodes_subset, train_nodes, val_nodes, test_nodes, split='train')
val_dataset = GraphDataset(data, unknown_nodes_subset, train_nodes, val_nodes, test_nodes, split='val')
test_dataset = GraphDataset(data, unknown_nodes_subset, train_nodes, val_nodes, test_nodes, split='test')

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

# Wrapper for single-device that handles list input (like DataParallel)
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
        return self.module(batch)

# Feature dimension depends on feature type
num_features = 5 * len(labels) if FEATURE_TYPE == 'graph_based' else len(labels)
base_model = TAGConvModel(num_features=num_features, num_classes=len(labels)).to(torch.device("cuda:0") if device.type == "cuda" else device)

if multi_gpu:
    device_ids = list(range(num_gpus))
    model = GeoDataParallel(base_model, device_ids=device_ids)
    print(f"Multi-GPU mode: {num_gpus} GPUs, device_ids={device_ids}")
else:
    model = SingleDeviceWrapper(base_model, torch.device("cuda:0") if device.type == "cuda" else device)
    print(f"Single GPU mode")

print(f"Features: {FEATURE_TYPE} ({num_features}), Params: {sum(p.numel() for p in model.parameters()):,}")


def evaluate(model, loader):
    """Evaluate model using DataListLoader for multi-GPU compatibility."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval', leave=False):
            logits = model(batch)
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

LR, WD, EPOCHS, PATIENCE = 0.0001, 0.0001, 5, 5
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
print(f"Train batch size: {TRAIN_BATCH_SIZE} (1 per GPU), Eval batch size: {EVAL_BATCH_SIZE}")

class_counts = [sum(1 for n in train_nodes if np.argmax(node_class[n]) == c) for c in range(len(labels))]
class_weights = torch.tensor([max(class_counts) / c for c in class_counts], dtype=torch.float).to(torch.device("cuda:0"))
print(f"Class weights: {dict(zip(labels, [f'{w:.2f}' for w in class_weights.tolist()]))}")

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = StepLR(optimizer, step_size=50, gamma=0.95)

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
        
        out = model(batch)
        
        # Compute loss for each graph's target node
        sizes = [int(d.num_nodes) for d in batch]
        ptr = torch.as_tensor(np.cumsum([0] + sizes), device=out.device)
        target_indices = torch.as_tensor(
            [ptr[i].item() + int(d.target_node_idx) for i, d in enumerate(batch)],
            device=out.device
        )
        targets = torch.as_tensor([int(d.target_label) for d in batch], device=out.device)
        
        loss = criterion(out[target_indices], targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        
        if batch_idx % 50 == 0:
            loop.set_postfix(loss=f"{loss.item():.4f}")
            torch.cuda.empty_cache()
    
    val_f1, _, _ = evaluate(model, val_loader)
    
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

test_f1, y_true, y_pred = evaluate(model, test_loader)
print(f"Test F1: {test_f1:.4f}")
print(classification_report(y_true, y_pred, target_names=labels, digits=4))
