import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.loader import DataListLoader
from torch.optim.lr_scheduler import StepLR
from common import (
    MaskedGraphDataset, TAGConvModel, SingleDeviceWrapper,
    evaluate, generate_submission
)
import random
from collections import defaultdict
import os

NUM_UNKNOWN_FRACTION = 0.05
EXP_NAME = f'exp_combined_{NUM_UNKNOWN_FRACTION}'
SPLITS_DIR = 'splits_v2'
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

    def len(self):
        return self.num_samples

    def get(self, idx):
        sample_nodes = random.sample(self.train_nodes, 2 * self.mask_count)
        pairs = [(sample_nodes[2*i], sample_nodes[2*i+1]) for i in range(self.mask_count)]
        
        parent_nodes_set = set(sample_nodes)
        
        all_base_nodes = self.train_nodes.copy()  # Includes parent nodes
        all_base_nodes.extend(self.unknown_nodes_subset)
        all_nodes_set = set(all_base_nodes)
        
        sorted_all_base_nodes = sorted(all_base_nodes)
        full_node_mapping = {node: i for i, node in enumerate(sorted_all_base_nodes)}
        
        synthetic_start_idx = len(sorted_all_base_nodes)
        
        # Build base edges using full graph (includes parent nodes)
        mask = (self.data_df['node_id1'].isin(all_nodes_set) & self.data_df['node_id2'].isin(all_nodes_set))
        subgraph_edges = self.data_df[mask]
        base_src = [full_node_mapping[n] for n in subgraph_edges['node_id1'].values]
        base_dst = [full_node_mapping[n] for n in subgraph_edges['node_id2'].values]
        
        # Now remove parent nodes for final graph
        base_nodes = [n for n in self.train_nodes if n not in parent_nodes_set]
        base_nodes.extend(self.unknown_nodes_subset)
        nodes_set = set(base_nodes)
        sorted_base_nodes = sorted(base_nodes)
        node_mapping = {node: i for i, node in enumerate(sorted_base_nodes)}
        
        synthetic_edges_dict = defaultdict(list)
        synthetic_node_features = []
        synthetic_node_labels = []
        
        num_classes = self.node_classes.shape[1]
        
        for syn_idx, (node1, node2) in enumerate(pairs):
            coef = random.randint(1, 9) / 10
            label1 = self.node_classes[node1]
            label2 = self.node_classes[node2]
            combined_label = coef * label1 + (1 - coef) * label2
            synthetic_node_labels.append(combined_label)
            synthetic_node_features.append(np.ones(num_classes) / num_classes)
            node1_full_idx = full_node_mapping[node1]
            node2_full_idx = full_node_mapping[node2]

            node1_base_indices = [full_node_mapping[n] for n in self.adj_list.get(node1, set()) if n in full_node_mapping]
            node2_base_indices = [full_node_mapping[n] for n in self.adj_list.get(node2, set()) if n in full_node_mapping]
            
            node1_syn_neighbors = synthetic_edges_dict.get(node1_full_idx, [])
            node2_syn_neighbors = synthetic_edges_dict.get(node2_full_idx, [])
            
            edges1_all_indices = node1_base_indices + node1_syn_neighbors
            edges2_all_indices = node2_base_indices + node2_syn_neighbors
            
            if edges1_all_indices:
                mask1 = np.random.rand(len(edges1_all_indices)) < coef
                sampled_edges1 = [edges1_all_indices[j] for j in range(len(edges1_all_indices)) if mask1[j]]
            else:
                sampled_edges1 = []
                
            if edges2_all_indices:
                mask2 = np.random.rand(len(edges2_all_indices)) < (1 - coef)
                sampled_edges2 = [edges2_all_indices[j] for j in range(len(edges2_all_indices)) if mask2[j]]
            else:
                sampled_edges2 = []

            syn_global_idx = synthetic_start_idx + syn_idx
            new_edges_indices = list(set(sampled_edges1 + sampled_edges2))

            synthetic_edges_dict[syn_global_idx] = new_edges_indices.copy()

            for neighbor_idx in new_edges_indices:
                synthetic_edges_dict[neighbor_idx].append(syn_global_idx)
        
        parent_indices_full = {full_node_mapping[n] for n in parent_nodes_set}
        full_to_final = {}
        final_idx = 0
        for full_idx in range(synthetic_start_idx):
            if full_idx not in parent_indices_full:
                full_to_final[full_idx] = final_idx
                final_idx += 1
        for syn_idx in range(len(pairs)):
            full_to_final[synthetic_start_idx + syn_idx] = len(sorted_base_nodes) + syn_idx
        
        base_src_final = [full_to_final[idx] for idx in base_src if idx in full_to_final]
        base_dst_final = [full_to_final[idx] for idx in base_dst if idx in full_to_final]
        
        extra_src = []
        extra_dst = []
        for node_idx, neighbors in synthetic_edges_dict.items():
            for neighbor_idx in neighbors:
                if node_idx >= synthetic_start_idx or neighbor_idx >= synthetic_start_idx:
                    if node_idx in full_to_final and neighbor_idx in full_to_final:
                        extra_src.append(full_to_final[node_idx])
                        extra_dst.append(full_to_final[neighbor_idx])
        
        all_src = base_src_final + extra_src
        all_dst = base_dst_final + extra_dst
        edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
        
        base_features = torch.tensor(self.node_classes[sorted_base_nodes], dtype=torch.float)
        syn_features = torch.tensor(np.array(synthetic_node_features), dtype=torch.float)
        node_features = torch.cat([base_features, syn_features], dim=0)
        
        base_labels = torch.tensor(self.node_classes[sorted_base_nodes], dtype=torch.float)
        syn_labels = torch.tensor(np.array(synthetic_node_labels), dtype=torch.float)
        y = torch.cat([base_labels, syn_labels], dim=0)
        
        total_nodes = len(sorted_base_nodes) + len(pairs)
        predict_mask = torch.zeros(total_nodes, dtype=torch.bool)
        # Synthetic nodes start after base nodes in final graph
        final_synthetic_start_idx = len(sorted_base_nodes)
        predict_mask[final_synthetic_start_idx:] = True
        
        # Validation: check for invalid values
        if len(edge_index[0]) == 0:
            raise ValueError("Empty edge_index!")
        if edge_index.max() >= total_nodes:
            raise ValueError(f"Edge index out of bounds: max={edge_index.max()}, total_nodes={total_nodes}")
        if torch.isnan(y).any() or torch.isinf(y).any():
            raise ValueError("NaN or Inf in labels!")
        if predict_mask.sum() == 0:
            raise ValueError("No nodes to predict!")
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            y=y,
            num_classes=num_classes,
            predict_mask=predict_mask,
            masked_node_ids=torch.tensor(list(range(self.mask_count)), dtype=torch.long)  # placeholder
        )


train_dataset = CombinedNodeTrainDataset(
    data, adj_list, unknown_nodes_subset, train_nodes,
    mask_count=MASK_COUNT, num_samples=NUM_SAMPLES, node_classes=node_labels
)

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

criterion = torch.nn.CrossEntropyLoss()
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

