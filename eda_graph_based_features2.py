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
import random
import pickle
import os
from collections import defaultdict
import igraph as ig

EXP_NAME = 'exp_mask_graph_based_simple_graph_based_extra'

print("Loading precomputed splits...")
SPLITS_DIR = 'splits'
assert os.path.exists(SPLITS_DIR)

train_nodes = np.load(os.path.join(SPLITS_DIR, 'train_nodes.npy')).tolist()
val_nodes = np.load(os.path.join(SPLITS_DIR, 'val_nodes.npy')).tolist()
test_nodes = np.load(os.path.join(SPLITS_DIR, 'test_nodes.npy')).tolist()
unknown_nodes = np.load(os.path.join(SPLITS_DIR, 'unknown_nodes.npy')).tolist()
node_labels = np.load(os.path.join(SPLITS_DIR, 'node_labels_masked.npy'))

with open(os.path.join(SPLITS_DIR, 'labels.txt'), 'r') as f:
    labels = [line.strip() for line in f]

with open(os.path.join(SPLITS_DIR, 'metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f)

max_node = metadata['max_node']
print(f"Labels: {labels}")
print(f"Train nodes: {len(train_nodes)}, Val nodes: {len(val_nodes)}, Test nodes: {len(test_nodes)}")

node_class = np.zeros((max_node + 1, len(labels)))
for n in range(max_node + 1):
    if node_labels[n] >= 0:
        node_class[n, node_labels[n]] = 1
    else:
        node_class[n, :] = np.ones(len(labels)) / len(labels)  # Uniform for masked/unknown

data = pd.read_csv('CR_real_masks_more_labeled_veritices_agreed.csv')
data['node_id1'] -= 1
data['node_id2'] -= 1

device = torch.device("cuda:0")
print(f"Device: {device}")

random.seed(42)
unknown_nodes_shuffled = unknown_nodes.copy()
random.shuffle(unknown_nodes_shuffled)
NUM_UNKNOWN_FRACTION = 0.25
num_unknown_to_use = int(len(unknown_nodes_shuffled) * NUM_UNKNOWN_FRACTION)
unknown_nodes_subset = unknown_nodes_shuffled[:num_unknown_to_use]
print(f"Using {num_unknown_to_use} unknown nodes ({NUM_UNKNOWN_FRACTION*100:.0f}%)")

MASK_COUNT = 64
FEATURE_TYPE = 'graph_based'

# Train Node2Vec on the train graph
from torch_geometric.nn import Node2Vec
import networkx as nx

NODE2VEC_DIM = 32
NODE2VEC_WALK_LENGTH = 20
NODE2VEC_CONTEXT_SIZE = 10
NODE2VEC_WALKS_PER_NODE = 10
NODE2VEC_P = 1.0
NODE2VEC_Q = 1.0
NODE2VEC_EPOCHS = 50

print("Building train graph for Node2Vec...")
# Build graph from train nodes + unknown subset
train_graph_nodes = set(train_nodes + unknown_nodes_subset)
train_edges_mask = (data['node_id1'].isin(train_graph_nodes) & data['node_id2'].isin(train_graph_nodes))
train_edges = data[train_edges_mask]

# Create node mapping for train graph
all_train_graph_nodes = sorted(train_graph_nodes)
node_to_idx = {n: i for i, n in enumerate(all_train_graph_nodes)}
idx_to_node = {i: n for n, i in node_to_idx.items()}

# Build edge_index for Node2Vec
src = [node_to_idx[n] for n in train_edges['node_id1'].values]
dst = [node_to_idx[n] for n in train_edges['node_id2'].values]
self_loops = list(range(len(all_train_graph_nodes)))
train_edge_index = torch.tensor(
    [src + dst + self_loops, dst + src + self_loops],
    dtype=torch.long
)

print(f"Train graph: {len(all_train_graph_nodes)} nodes, {train_edge_index.shape[1]} edges")

# Train Node2Vec
node2vec_model = Node2Vec(
    train_edge_index,
    embedding_dim=NODE2VEC_DIM,
    walk_length=NODE2VEC_WALK_LENGTH,
    context_size=NODE2VEC_CONTEXT_SIZE,
    walks_per_node=NODE2VEC_WALKS_PER_NODE,
    p=NODE2VEC_P,
    q=NODE2VEC_Q,
    num_negative_samples=1,
    sparse=True
).to(device)

node2vec_loader = node2vec_model.loader(batch_size=128, shuffle=True, num_workers=4)
node2vec_optimizer = torch.optim.SparseAdam(node2vec_model.parameters(), lr=0.01)

print("Training Node2Vec...")
node2vec_model.train()
for epoch in range(1, NODE2VEC_EPOCHS + 1):
    total_loss = 0
    for pos_rw, neg_rw in tqdm(node2vec_loader, desc=f'Node2Vec Epoch {epoch}'):
        node2vec_optimizer.zero_grad()
        loss = node2vec_model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        node2vec_optimizer.step()
        total_loss += loss.item()
    print(f"Node2Vec Epoch {epoch}: loss={total_loss/len(node2vec_loader):.4f}")

# Get embeddings for all train graph nodes
node2vec_model.eval()
with torch.no_grad():
    train_node_embeddings = node2vec_model.embedding.weight.cpu()

print(f"Node2Vec embeddings shape: {train_node_embeddings.shape}")

# Create a lookup: original_node_id -> embedding
# For nodes not in train graph, we'll use the query vector
node2vec_embeddings = {}
for idx, node in idx_to_node.items():
    node2vec_embeddings[node] = train_node_embeddings[idx].numpy()

# Clean up Node2Vec model and loader to free GPU memory
del node2vec_model, node2vec_loader, node2vec_optimizer, train_node_embeddings
torch.cuda.empty_cache()
import gc
gc.collect()
print("Cleaned up Node2Vec resources")

def compute_graph_based_features(sorted_nodes, subgraph_edges, node_class_input, num_classes):
    num_nodes = len(sorted_nodes)
    node_mapping = {node: i for i, node in enumerate(sorted_nodes)}
    node_class_subset = node_class_input[sorted_nodes]
    node_labels_arr = np.argmax(node_class_subset, axis=1)
    node_labels_arr[np.max(node_class_subset, axis=1) <= 0.5] = -1
    features = np.zeros((num_nodes, 5 * num_classes), dtype=np.float32)
    weights_per_node_class = defaultdict(list)
    edges = subgraph_edges[['node_id1', 'node_id2', 'ibd_sum']].values
    for n1, n2, w in edges:
        if n1 in node_mapping and n2 in node_mapping:
            i, j = node_mapping[n1], node_mapping[n2]
            c_j, c_i = node_labels_arr[j], node_labels_arr[i]
            if c_j >= 0:
                features[i, c_j] += 1
                features[i, num_classes + c_j] += w
                features[i, 3*num_classes + c_j] = max(features[i, 3*num_classes + c_j], w)
                features[i, 4*num_classes + c_j] += w
                weights_per_node_class[(i, c_j)].append(w)
            if c_i >= 0:
                features[j, c_i] += 1
                features[j, num_classes + c_i] += w
                features[j, 3*num_classes + c_i] = max(features[j, 3*num_classes + c_i], w)
                features[j, 4*num_classes + c_i] += w
                weights_per_node_class[(j, c_i)].append(w)
    count_mask = features[:, :num_classes] > 0
    features[:, num_classes:2*num_classes][count_mask] /= features[:, :num_classes][count_mask]
    for (i, c), weights in weights_per_node_class.items():
        if len(weights) > 1:
            features[i, 2*num_classes + c] = np.std(weights, dtype=np.float32)
    return torch.tensor(features, dtype=torch.float32)


def compute_graph_based_features_extra(sorted_nodes, subgraph_edges, node_class_input, num_classes):
    num_nodes = len(sorted_nodes)
    node_mapping = {node: i for i, node in enumerate(sorted_nodes)}
    node_class_subset = node_class_input[sorted_nodes]
    node_labels_arr = np.argmax(node_class_subset, axis=1)
    node_labels_arr[np.max(node_class_subset, axis=1) <= 0.5] = -1

    features = np.zeros((num_nodes, 5 * num_classes + 6), dtype=np.float32)
    
    weights_per_node_class = defaultdict(list)
    edges = subgraph_edges[['node_id1', 'node_id2', 'ibd_sum']].values
    
    adj_list = defaultdict(list)
    edge_list = []
    
    for n1, n2, w in edges:
        if n1 in node_mapping and n2 in node_mapping:
            i, j = node_mapping[n1], node_mapping[n2]
            adj_list[i].append((j, w))
            adj_list[j].append((i, w))
            edge_list.append((i, j, w))

            c_j, c_i = node_labels_arr[j], node_labels_arr[i]
            if c_j >= 0:
                features[i, c_j] += 1
                features[i, num_classes + c_j] += w
                features[i, 3*num_classes + c_j] = max(features[i, 3*num_classes + c_j], w)
                features[i, 4*num_classes + c_j] += w
                weights_per_node_class[(i, c_j)].append(w)
            if c_i >= 0:
                features[j, c_i] += 1
                features[j, num_classes + c_i] += w
                features[j, 3*num_classes + c_i] = max(features[j, 3*num_classes + c_i], w)
                features[j, 4*num_classes + c_i] += w
                weights_per_node_class[(j, c_i)].append(w)
    
    # Compute label-based average weights
    count_mask = features[:, :num_classes] > 0
    features[:, num_classes:2*num_classes][count_mask] /= features[:, :num_classes][count_mask]
    
    # Compute label-based std
    for (i, c), weights in weights_per_node_class.items():
        if len(weights) > 1:
            features[i, 2*num_classes + c] = np.std(weights, dtype=np.float32)
    
    # Label-free features start at index 5*num_classes
    base_idx = 5 * num_classes
    
    for i in range(num_nodes):
        features[i, base_idx] = len(adj_list[i])
    
    for i in range(num_nodes):
        features[i, base_idx + 1] = sum(w for _, w in adj_list[i])
    
    g = ig.Graph()
    g.add_vertices(num_nodes)
    edges = []
    weights = []
    edge_weight_dict = defaultdict(float)
    for i, j, w in edge_list:   
        if i != j:
            key = (min(i, j), max(i, j))
            edge_weight_dict[key] += w

    for (i, j), w in edge_weight_dict.items():
        edges.append((i, j))
        weights.append(w)
    
    g.add_edges(edges)
    g.es['weight'] = weights

    clustering = g.transitivity_local_undirected()
    clustering_array = np.array(clustering, dtype=np.float32)
    is_nan = np.isnan(clustering_array)
    clustering_array[is_nan] = 0
    features[:, base_idx + 2] = clustering_array
    features[:, base_idx + 3] = is_nan.astype(np.float32)

    pagerank = g.pagerank(weights='weight')
    features[:, base_idx + 4] = np.array(pagerank, dtype=np.float32)

    # eigenvector_centrality = g.eigenvector_centrality(weights='weight')
    # features[:, base_idx + 4] = np.array(eigenvector_centrality, dtype=np.float32)

    return torch.tensor(features, dtype=torch.float32)


# Dataset: Node2Vec embeddings + graph-based features combined
# For masked nodes: query vector for node2vec part, zero for graph-based part

class Node2VecGraphBasedDataset(Dataset):
    def __init__(self, data_df, unknown_nodes_subset, train_nodes, val_nodes, test_nodes,
                 node2vec_embeddings, node2vec_dim, num_classes,
                 split='train', mask_count=64, num_samples=500, return_node_ids=False):
        super().__init__()
        self.data_df = data_df
        self.unknown_nodes_subset = unknown_nodes_subset
        self.train_nodes = train_nodes
        self.val_nodes = val_nodes
        self.test_nodes = test_nodes
        self.node2vec_embeddings = node2vec_embeddings
        self.node2vec_dim = node2vec_dim
        self.total_classes = num_classes
        self.graph_based_dim = 5 * num_classes
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
        
        mask_indices = [node_mapping[n] for n in mask_nodes if n in node_mapping]
        predict_mask = torch.zeros(len(sorted_nodes), dtype=torch.bool)
        predict_mask[mask_indices] = True
        
        # === Node2Vec embeddings ===
        node2vec_features = np.zeros((len(sorted_nodes), self.node2vec_dim), dtype=np.float32)
        use_query_mask = torch.zeros(len(sorted_nodes), dtype=torch.bool)
        
        mask_nodes_set = set(mask_nodes)
        for i, node in enumerate(sorted_nodes):
            if node in mask_nodes_set:
                use_query_mask[i] = True  # Will use query vector
            elif node in self.node2vec_embeddings:
                node2vec_features[i] = self.node2vec_embeddings[node]
            else:
                assert False
        
        node2vec_features = torch.tensor(node2vec_features, dtype=torch.float32)
        
        # === Graph-based features (with masking for data leak prevention) ===
        node_class_masked = node_class.copy()
        for n in mask_nodes:
            node_class_masked[n, :] = np.ones(self.total_classes) / self.total_classes
        
        graph_features = compute_graph_based_features_extra(sorted_nodes, subgraph_edges, node_class_masked, self.total_classes)
        # graph_features[predict_mask] = 0
        
        onehot = torch.tensor(node_class_masked[sorted_nodes], dtype=torch.float)
        node_features = torch.cat([node2vec_features, graph_features, onehot], dim=1)
        # node_features = torch.cat([graph_features, onehot], dim=1)
        
        # Labels
        onehot = torch.tensor(node_class[sorted_nodes], dtype=torch.float)
        y = torch.tensor([torch.argmax(onehot[i]).item() for i in range(len(sorted_nodes))], dtype=torch.long)
        
        # Edge weights
        ibd = subgraph_edges['ibd_sum'].values
        edge_weights = torch.tensor(list(ibd) + list(ibd), dtype=torch.float)
        
        if self.return_node_ids:
            masked_node_ids = torch.tensor([mask_nodes[mask_indices.index(i)] for i in mask_indices], dtype=torch.long)
            return Data(
                x=node_features,
                edge_index=edge_index,
                y=y,
                weight=edge_weights,
                num_classes=self.total_classes,
                predict_mask=predict_mask,
                use_query_mask=use_query_mask,
                masked_node_ids=masked_node_ids
            )
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            y=y,
            weight=edge_weights,
            num_classes=self.total_classes,
            predict_mask=predict_mask,
            use_query_mask=use_query_mask
        )


class TAGConvModelWithQuery(torch.nn.Module):
    def __init__(self, node2vec_dim, graph_based_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.node2vec_dim = node2vec_dim
        self.graph_based_dim = graph_based_dim
        
        self.query_vector = torch.nn.Parameter(torch.randn(node2vec_dim))
        
        total_features = node2vec_dim + graph_based_dim + num_classes
        # total_features = graph_based_dim + num_classes
        self.first_linear = torch.nn.Linear(total_features, hidden_dim)
        self.conv1 = TAGConv(hidden_dim, hidden_dim)
        self.conv2 = TAGConv(hidden_dim, hidden_dim)
        self.conv3 = TAGConv(hidden_dim, hidden_dim)
        self.n1 = GraphNorm(hidden_dim)
        self.n2 = GraphNorm(hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.weight
        use_query_mask = data.use_query_mask
        
        node2vec_part = x[:, :self.node2vec_dim].clone()
        graph_based_part = x[:, self.node2vec_dim:]
        
        node2vec_part[use_query_mask] = self.query_vector
        
        x = torch.cat([node2vec_part, graph_based_part], dim=1)
        
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
            batch = Batch.from_data_list(data_list).to(self.device)
        else:
            batch = data_list.to(self.device)
        return self.module(batch), batch


NUM_CLASSES = len(labels)
GRAPH_BASED_DIM = 5 * NUM_CLASSES + 6
# GRAPH_BASED_DIM = 5 * NUM_CLASSES

train_dataset = Node2VecGraphBasedDataset(data, unknown_nodes_subset, train_nodes, val_nodes, test_nodes,
                                           node2vec_embeddings, NODE2VEC_DIM, NUM_CLASSES,
                                           split='train', mask_count=MASK_COUNT, num_samples=500)
val_dataset = Node2VecGraphBasedDataset(data, unknown_nodes_subset, train_nodes, val_nodes, test_nodes,
                                         node2vec_embeddings, NODE2VEC_DIM, NUM_CLASSES,
                                         split='val', mask_count=MASK_COUNT)
test_dataset = Node2VecGraphBasedDataset(data, unknown_nodes_subset, train_nodes, val_nodes, test_nodes,
                                          node2vec_embeddings, NODE2VEC_DIM, NUM_CLASSES,
                                          split='test', mask_count=MASK_COUNT, return_node_ids=True)


print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

total_features = NODE2VEC_DIM + GRAPH_BASED_DIM
base_model = TAGConvModelWithQuery(
    node2vec_dim=NODE2VEC_DIM, 
    graph_based_dim=GRAPH_BASED_DIM, 
    num_classes=NUM_CLASSES
).to(device)

model = SingleDeviceWrapper(base_model, device)
print("Single GPU mode")

print(f"Features: node2vec({NODE2VEC_DIM}) + graph_based({GRAPH_BASED_DIM}) = {total_features}")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
# print(f"Query vector shape: {base_model.query_vector.shape}")


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

LR, WD, EPOCHS, PATIENCE = 0.0001, 0.0001, 10, 5
BATCH_SIZE = 1

train_loader = DataListLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
val_loader = DataListLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True)
test_loader = DataListLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True)

class_counts = [sum(1 for n in train_nodes if node_labels[n] == c) for c in range(len(labels))]
class_weights = torch.tensor([max(class_counts) / c for c in class_counts], dtype=torch.float).to(device)
print(f"Class weights: {dict(zip(labels, [f'{w:.2f}' for w in class_weights.tolist()]))}")

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = StepLR(optimizer, step_size=50, gamma=0.95)

scaler = torch.amp.GradScaler(device)
use_amp = 'cuda' in str(device.type)

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
        
        with torch.amp.autocast('cuda' if use_amp else 'cpu', enabled=use_amp):
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
    
    # Clear cache after each epoch and step scheduler
    torch.cuda.empty_cache()
    gc.collect()
    
    val_f1, _, _ = evaluate(model, val_loader, use_amp)
    
    if val_f1 > best_val_f1:
        best_val_f1, patience_counter = val_f1, 0
        to_save = getattr(model, 'module', model)
        best_state = {k: v.cpu().clone() for k, v in to_save.state_dict().items()}
        print(f"[Epoch {epoch}] val_f1={best_val_f1:.4f} ↑ | loss={np.mean(losses):.4f}")
    else:
        patience_counter += 1
        print(f"[Epoch {epoch}] val_f1={val_f1:.4f} | loss={np.mean(losses):.4f} | patience={patience_counter}")
    
    # Force garbage collection between epochs
    torch.cuda.empty_cache()
    gc.collect()

print(f"\nBest val F1: {best_val_f1:.4f}")

if best_state:
    to_load = getattr(model, 'module', model)
    to_load.load_state_dict(best_state)
    print("Loaded best model state")

test_f1, y_true, y_pred = evaluate(model, test_loader, use_amp)
print(f"Test F1: {test_f1:.4f}")
print(classification_report(y_true, y_pred, target_names=labels, digits=4))

os.makedirs('submissions', exist_ok=True)
submission_path = f'submissions/{EXP_NAME}.csv'
generate_submission(model, test_loader, submission_path, use_amp)
print(f"\nTo score: python score.py {submission_path}")
