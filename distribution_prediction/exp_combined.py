"""
Experiment: Combined/Synthetic nodes for training
- For val/test: mask and predict real nodes (same as before)
- For train: create synthetic combined nodes from pairs of train nodes
  - Sample 2*mask_count nodes, pair them to create mask_count synthetic nodes
  - Synthetic nodes have mixed labels and sampled edges from both parents
  - Include edges between synthetic nodes based on parent connectivity
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import TAGConv, GraphNorm
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score, classification_report
from common import (
    MaskedGraphDataset, TAGConvModel, SingleDeviceWrapper,
    evaluate, generate_submission
)
import random
from collections import defaultdict
import os

NUM_UNKNOWN_FRACTION = 0.05
EXP_NAME = f'exp_combined_{NUM_UNKNOWN_FRACTION}'
SPLITS_DIR = 'splits'
DEVICE = 'cuda:0'

MASK_COUNT = 64
NUM_SAMPLES = 500

LR, WD, EPOCHS, PATIENCE = 0.0001, 0.0001, 10, 5
BATCH_SIZE = 1

assert os.path.exists(SPLITS_DIR)

train_nodes = np.load(os.path.join(SPLITS_DIR, 'train_nodes.npy')).tolist()
val_nodes = np.load(os.path.join(SPLITS_DIR, 'val_nodes.npy')).tolist()
test_nodes = np.load(os.path.join(SPLITS_DIR, 'test_nodes.npy')).tolist()
unknown_nodes = np.load(os.path.join(SPLITS_DIR, 'unknown_nodes.npy')).tolist()
node_labels = np.load(os.path.join(SPLITS_DIR, 'node_labels_masked.npy'))

with open(os.path.join(SPLITS_DIR, 'labels.txt'), 'r') as f:
    labels = [line.strip() for line in f]

print(f"Labels: {labels}")
print(f"Train nodes: {len(train_nodes)}, Val nodes: {len(val_nodes)}, Test nodes: {len(test_nodes)}")

data = pd.read_csv(os.path.join(SPLITS_DIR, 'edges_data.csv'))

# Build adjacency list for efficient edge lookup
print("Building adjacency list...")
adj_list = defaultdict(set)
for _, row in tqdm(data.iterrows(), total=len(data), desc="Building adj"):
    adj_list[row['node_id1']].add(row['node_id2'])
    adj_list[row['node_id2']].add(row['node_id1'])
print(f"Adjacency list built for {len(adj_list)} nodes")

device = torch.device(DEVICE)
print(f"Device: {device}")

random.seed(42)
unknown_nodes_shuffled = unknown_nodes.copy()
random.shuffle(unknown_nodes_shuffled)
num_unknown_to_use = int(len(unknown_nodes_shuffled) * NUM_UNKNOWN_FRACTION)
unknown_nodes_subset = unknown_nodes_shuffled[:num_unknown_to_use]
print(f"Using {num_unknown_to_use} unknown nodes ({NUM_UNKNOWN_FRACTION*100:.0f}%)")


class CombinedNodeTrainDataset(Dataset):
    """Training dataset that creates synthetic combined nodes from pairs of train nodes."""
    
    def __init__(self, data_df, adj_list, unknown_nodes_subset, train_nodes, 
                 mask_count, num_samples, node_classes):
        super().__init__()
        self.data_df = data_df
        self.adj_list = adj_list
        self.unknown_nodes_subset = unknown_nodes_subset
        self.train_nodes = train_nodes
        self.mask_count = mask_count
        self.num_samples = num_samples
        self.node_classes = node_classes
        self.num_classes = node_classes.shape[1]

    def len(self):
        return self.num_samples

    def get(self, idx):
        # Sample 2*mask_count train nodes for creating pairs
        sample_nodes = random.sample(self.train_nodes, 2 * self.mask_count)
        
        # Create pairs: (node1, node2), (node3, node4), ...
        pairs = [(sample_nodes[2*i], sample_nodes[2*i+1]) for i in range(self.mask_count)]
        
        # Base nodes for the graph: all train nodes + unknown subset
        base_nodes = self.train_nodes.copy()
        base_nodes.extend(self.unknown_nodes_subset)
        nodes_set = set(base_nodes)
        
        # Get edges within base nodes
        mask = (self.data_df['node_id1'].isin(nodes_set) & self.data_df['node_id2'].isin(nodes_set))
        subgraph_edges = self.data_df[mask]
        
        # Create node mapping for base nodes
        sorted_base_nodes = sorted(base_nodes)
        node_mapping = {node: i for i, node in enumerate(sorted_base_nodes)}
        
        # Create synthetic nodes starting after base nodes
        synthetic_start_idx = len(sorted_base_nodes)
        synthetic_node_features = []
        synthetic_node_labels = []
        synthetic_edges_src = []
        synthetic_edges_dst = []
        
        # Track which synthetic nodes came from which parents (for inter-synthetic edges)
        synthetic_parents = []
        
        for syn_idx, (node1, node2) in enumerate(pairs):
            # Random mixing coefficient
            coef = random.randint(1, 9) / 10
            
            # Combined label (soft label)
            label1 = self.node_classes[node1]
            label2 = self.node_classes[node2]
            combined_label = coef * label1 + (1 - coef) * label2
            synthetic_node_labels.append(combined_label)
            
            # Features will be masked (uniform), so we just store placeholder
            synthetic_node_features.append(np.ones(self.num_classes) / self.num_classes)
            
            # Get edges from parents
            edges1 = list(self.adj_list.get(node1, set()) & nodes_set)
            edges2 = list(self.adj_list.get(node2, set()) & nodes_set)
            
            # Sample edges based on coefficient
            if edges1:
                mask1 = np.random.rand(len(edges1)) < coef
                sampled_edges1 = [edges1[j] for j in range(len(edges1)) if mask1[j]]
            else:
                sampled_edges1 = []
                
            if edges2:
                mask2 = np.random.rand(len(edges2)) < (1 - coef)
                sampled_edges2 = [edges2[j] for j in range(len(edges2)) if mask2[j]]
            else:
                sampled_edges2 = []
            
            # Add edges from synthetic node to base nodes
            syn_global_idx = synthetic_start_idx + syn_idx
            for neighbor in sampled_edges1 + sampled_edges2:
                if neighbor in node_mapping:
                    # Bidirectional edges
                    synthetic_edges_src.append(syn_global_idx)
                    synthetic_edges_dst.append(node_mapping[neighbor])
                    synthetic_edges_src.append(node_mapping[neighbor])
                    synthetic_edges_dst.append(syn_global_idx)
            
            synthetic_parents.append((node1, node2, coef))
        
        # Add edges between synthetic nodes based on parent connectivity
        for i in range(len(pairs)):
            node1_i, node2_i, coef_i = synthetic_parents[i]
            parents_i = {node1_i, node2_i}
            
            for j in range(i + 1, len(pairs)):
                node1_j, node2_j, coef_j = synthetic_parents[j]
                parents_j = {node1_j, node2_j}
                
                # Check if any parent of i is connected to any parent of j
                connected = False
                for p_i in parents_i:
                    for p_j in parents_j:
                        if p_j in self.adj_list.get(p_i, set()):
                            connected = True
                            break
                    if connected:
                        break
                
                if connected:
                    # Add bidirectional edge between synthetic nodes i and j
                    syn_i = synthetic_start_idx + i
                    syn_j = synthetic_start_idx + j
                    synthetic_edges_src.extend([syn_i, syn_j])
                    synthetic_edges_dst.extend([syn_j, syn_i])
        
        # Build edge_index for base graph
        src = [node_mapping[n] for n in subgraph_edges['node_id1'].values]
        dst = [node_mapping[n] for n in subgraph_edges['node_id2'].values]
        
        # Combine with synthetic edges
        all_src = src + synthetic_edges_src
        all_dst = dst + synthetic_edges_dst
        edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
        
        # Build features: base nodes have true labels, synthetic nodes have uniform (masked)
        base_features = torch.tensor(self.node_classes[sorted_base_nodes], dtype=torch.float)
        syn_features = torch.tensor(np.array(synthetic_node_features), dtype=torch.float)
        node_features = torch.cat([base_features, syn_features], dim=0)
        
        # Build labels: base nodes + synthetic nodes
        base_labels = torch.tensor(self.node_classes[sorted_base_nodes], dtype=torch.float)
        syn_labels = torch.tensor(np.array(synthetic_node_labels), dtype=torch.float)
        y = torch.cat([base_labels, syn_labels], dim=0)
        
        # Predict mask: only synthetic nodes
        total_nodes = len(sorted_base_nodes) + len(pairs)
        predict_mask = torch.zeros(total_nodes, dtype=torch.bool)
        predict_mask[synthetic_start_idx:] = True
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            y=y,
            num_classes=self.num_classes,
            predict_mask=predict_mask,
            masked_node_ids=torch.tensor(list(range(self.mask_count)), dtype=torch.long)  # placeholder
        )


# Create datasets
train_dataset = CombinedNodeTrainDataset(
    data, adj_list, unknown_nodes_subset, train_nodes,
    mask_count=MASK_COUNT, num_samples=NUM_SAMPLES, node_classes=node_labels
)

# Val and test use the original MaskedGraphDataset
val_dataset = MaskedGraphDataset(data, unknown_nodes_subset, train_nodes, val_nodes, test_nodes,
                                  split='val', mask_count=MASK_COUNT, num_samples=NUM_SAMPLES, node_classes=node_labels)
test_dataset = MaskedGraphDataset(data, unknown_nodes_subset, train_nodes, val_nodes, test_nodes,
                                   split='test', mask_count=MASK_COUNT, num_samples=NUM_SAMPLES, node_classes=node_labels)
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

num_features = len(labels)
base_model = TAGConvModel(num_features=num_features, num_classes=len(labels)).to(DEVICE)
model = SingleDeviceWrapper(base_model, DEVICE)
print("Single GPU mode")
print(f"Features: {num_features}, Params: {sum(p.numel() for p in model.parameters()):,}")


train_loader = DataListLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataListLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataListLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# Use MSE loss for soft labels (distribution prediction)
criterion = torch.nn.MSELoss()
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
            # Softmax the logits to get predictions, compare with soft labels
            preds = torch.softmax(logits[predict_mask], dim=-1)
            targets = batch_data.y[predict_mask]
            loss = criterion(preds, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.append(loss.item())
        
        if batch_idx % 50 == 0:
            loop.set_postfix(loss=f"{loss.item():.4f}")
            torch.cuda.empty_cache()
    
    val_f1, _, _ = evaluate(model, val_loader, use_amp, labels)
    
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

# ============== Generate Submission ==============
os.makedirs('submissions', exist_ok=True)
submission_path = f'submissions/{EXP_NAME}.csv'
generate_submission(model, test_loader, submission_path, use_amp, labels)
print(f"\nTo score: python score.py {submission_path}")

