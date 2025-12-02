"""
Experiment: Contrastive learning with Triplet/Margin Loss
- Learn node embeddings via contrastive learning
- Use kNN on embeddings for classification
- Select nodes as triplets (anchor, positive, negative) when masking
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

EXP_NAME = 'exp_contrastive_triplet'

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


print("Loading precomputed splits...")
SPLITS_DIR = 'splits'

if not os.path.exists(SPLITS_DIR):
    raise RuntimeError(f"Splits directory '{SPLITS_DIR}' not found. Run 'python precompute_splits.py' first.")

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
        node_class[n, :] = np.ones(len(labels)) / len(labels)

train_nodes_by_class = {c: [] for c in range(len(labels))}
for n in train_nodes:
    if node_labels[n] >= 0:
        train_nodes_by_class[node_labels[n]].append(n)

print("Loading edge data...")
data = pd.read_csv('CR_real_masks_more_labeled_veritices_agreed.csv')
data['node_id1'] -= 1
data['node_id2'] -= 1

DEVICE_CHOICE = "cuda"
device = torch.device(DEVICE_CHOICE if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
multi_gpu = (device.type == "cuda" and num_gpus >= 2)
print(f"Device: {device}, GPUs: {num_gpus}, Multi-GPU: {multi_gpu}")

random.seed(42)
unknown_nodes_shuffled = unknown_nodes.copy()
random.shuffle(unknown_nodes_shuffled)
NUM_UNKNOWN_FRACTION = 0.05
num_unknown_to_use = int(len(unknown_nodes_shuffled) * NUM_UNKNOWN_FRACTION)
unknown_nodes_subset = unknown_nodes_shuffled[:num_unknown_to_use]
print(f"Using {num_unknown_to_use} unknown nodes ({NUM_UNKNOWN_FRACTION*100:.0f}%)")

NUM_TRIPLETS = 21
EMBED_DIM = 512
MARGIN = 1.0


def sample_triplets(train_nodes_by_class, num_triplets):
    triplets = []
    classes = [c for c in train_nodes_by_class.keys() if len(train_nodes_by_class[c]) >= 2]
    if len(classes) < 2:
        return []
    
    for _ in range(num_triplets):
        anchor_class = random.choice(classes)
        neg_class = random.choice([c for c in classes if c != anchor_class])
        
        anchor, positive = random.sample(train_nodes_by_class[anchor_class], 2)
        negative = random.choice(train_nodes_by_class[neg_class])
        
        triplets.append((anchor, positive, negative))
    
    return triplets


class TripletGraphDataset(Dataset):
    def __init__(self, data_df, unknown_nodes_subset, train_nodes, val_nodes, test_nodes,
                 train_nodes_by_class, split='train', num_triplets=21, num_samples=500, 
                 mask_count=64, return_node_ids=False):
        super().__init__()
        self.data_df = data_df
        self.unknown_nodes_subset = unknown_nodes_subset
        self.train_nodes = train_nodes
        self.val_nodes = val_nodes
        self.test_nodes = test_nodes
        self.train_nodes_by_class = train_nodes_by_class
        self.split = split
        self.num_triplets = num_triplets
        self.num_samples = num_samples
        self.mask_count = mask_count
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
            triplets = sample_triplets(self.train_nodes_by_class, self.num_triplets)
            if len(triplets) == 0:
                return None
            mask_nodes = []
            for a, p, n in triplets:
                mask_nodes.extend([a, p, n])
        else:
            start_idx = idx * self.mask_count
            end_idx = min(start_idx + self.mask_count, len(self.target_nodes))
            mask_nodes = self.target_nodes[start_idx:end_idx]
            triplets = None
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

        node_features = torch.tensor(node_class[sorted_nodes], dtype=torch.float)

        mask_indices = [node_mapping[n] for n in mask_nodes if n in node_mapping]
        predict_mask = torch.zeros(len(sorted_nodes), dtype=torch.bool)
        predict_mask[mask_indices] = True

        node_features_masked = node_features.clone()
        node_features_masked[predict_mask] = torch.ones(len(labels)) / len(labels)

        onehot = torch.tensor(node_class[sorted_nodes], dtype=torch.float)
        y = torch.tensor([torch.argmax(onehot[i]).item() for i in range(len(sorted_nodes))], dtype=torch.long)

        ibd = subgraph_edges['ibd_sum'].values
        edge_weights = torch.tensor(list(ibd) + list(ibd), dtype=torch.float)

        if self.split == 'train' and triplets is not None:
            triplet_indices = []
            for a, p, n in triplets:
                if a in node_mapping and p in node_mapping and n in node_mapping:
                    triplet_indices.append([node_mapping[a], node_mapping[p], node_mapping[n]])
            triplet_indices = torch.tensor(triplet_indices, dtype=torch.long) if triplet_indices else torch.zeros((0, 3), dtype=torch.long)
        else:
            triplet_indices = torch.zeros((0, 3), dtype=torch.long)

        if self.return_node_ids:
            masked_node_ids = torch.tensor([mask_nodes[mask_indices.index(i)] for i in mask_indices], dtype=torch.long)
            return Data(
                x=node_features_masked, edge_index=edge_index, y=y, weight=edge_weights,
                num_classes=len(labels), predict_mask=predict_mask, masked_node_ids=masked_node_ids,
                triplet_indices=triplet_indices
            )

        return Data(
            x=node_features_masked, edge_index=edge_index, y=y, weight=edge_weights,
            num_classes=len(labels), predict_mask=predict_mask, triplet_indices=triplet_indices
        )


class EmbeddingModel(nn.Module):
    def __init__(self, num_features, embed_dim=512):
        super().__init__()
        self.first_linear = nn.Linear(num_features, embed_dim)
        self.conv1 = TAGConv(embed_dim, embed_dim)
        self.conv2 = TAGConv(embed_dim, embed_dim)
        self.conv3 = TAGConv(embed_dim, embed_dim)
        self.n1 = GraphNorm(embed_dim)
        self.n2 = GraphNorm(embed_dim)

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
        return F.normalize(x, p=2, dim=-1)


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


train_dataset = TripletGraphDataset(data, unknown_nodes_subset, train_nodes, val_nodes, test_nodes,
                                     train_nodes_by_class, split='train', num_triplets=NUM_TRIPLETS, num_samples=500)
val_dataset = TripletGraphDataset(data, unknown_nodes_subset, train_nodes, val_nodes, test_nodes,
                                   train_nodes_by_class, split='val', num_triplets=NUM_TRIPLETS)
test_dataset = TripletGraphDataset(data, unknown_nodes_subset, train_nodes, val_nodes, test_nodes,
                                    train_nodes_by_class, split='test', num_triplets=NUM_TRIPLETS, return_node_ids=True)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
print(f"Triplets per sample: {NUM_TRIPLETS} (= {NUM_TRIPLETS * 3} masked nodes)")

base_model = EmbeddingModel(num_features=len(labels), embed_dim=EMBED_DIM).to(
    torch.device("cuda:0") if device.type == "cuda" else device
)

if multi_gpu:
    model = GeoDataParallel(base_model, device_ids=list(range(num_gpus)))
else:
    model = SingleDeviceWrapper(base_model, torch.device("cuda:0") if device.type == "cuda" else device)

print(f"Params: {sum(p.numel() for p in model.parameters()):,}")


def compute_embeddings(model, loader, use_amp=True):
    model.eval()
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Computing embeddings', leave=False):
            with torch.amp.autocast('cuda', enabled=use_amp):
                if isinstance(model, SingleDeviceWrapper):
                    embeddings, batch_data = model(batch)
                else:
                    batch_data = Batch.from_data_list(batch).to(device)
                    embeddings = model.module(batch_data)
                predict_mask = batch_data.predict_mask

            emb = embeddings[predict_mask].cpu()
            lbl = batch_data.y[predict_mask].cpu()
            all_embeddings.append(emb)
            all_labels.append(lbl)
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def knn_predict(train_emb, train_y, query_emb, k=10):
    sims = torch.matmul(query_emb, train_emb.T)
    topk = torch.topk(sims, k=min(k, train_emb.size(0)), dim=1).indices
    preds = []
    for neighbors in topk:
        neighbor_labels = train_y[neighbors].numpy()
        classes, counts = np.unique(neighbor_labels, return_counts=True)
        preds.append(int(classes[np.argmax(counts)]))
    return preds


def evaluate_knn(model, train_loader, eval_loader, k=10, use_amp=True):
    train_emb, train_y = compute_embeddings(model, train_loader, use_amp)
    eval_emb, eval_y = compute_embeddings(model, eval_loader, use_amp)
    y_pred = knn_predict(train_emb, train_y, eval_emb, k=k)
    f1 = f1_score(eval_y.tolist(), y_pred, average='macro')
    return f1, eval_y.tolist(), y_pred


def generate_submission(model, train_loader, test_loader, output_path, k=10, use_amp=True):
    train_emb, train_y = compute_embeddings(model, train_loader, use_amp)
    
    model.eval()
    all_node_ids, all_predictions = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Generating submission', leave=False):
            with torch.amp.autocast('cuda', enabled=use_amp):
                if isinstance(model, SingleDeviceWrapper):
                    embeddings, batch_data = model(batch)
                else:
                    batch_data = Batch.from_data_list(batch).to(device)
                    embeddings = model.module(batch_data)
                predict_mask = batch_data.predict_mask
                masked_node_ids = batch_data.masked_node_ids

            emb = embeddings[predict_mask].cpu()
            preds = knn_predict(train_emb, train_y, emb, k=k)
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

train_loader = DataListLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataListLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataListLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

criterion = TripletLoss(margin=MARGIN)
print(f"Using TripletLoss with margin={MARGIN}")

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
                embeddings, batch_data = model(batch)
            else:
                batch_data = Batch.from_data_list(batch).to(device)
                embeddings = model.module(batch_data)

            triplet_indices = batch_data.triplet_indices
            if triplet_indices.numel() == 0:
                continue

            anchors = embeddings[triplet_indices[:, 0]]
            positives = embeddings[triplet_indices[:, 1]]
            negatives = embeddings[triplet_indices[:, 2]]

            loss = criterion(anchors, positives, negatives)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.append(loss.item())

        if batch_idx % 50 == 0:
            loop.set_postfix(loss=f"{loss.item():.4f}")
            torch.cuda.empty_cache()

    val_f1, _, _ = evaluate_knn(model, train_loader, val_loader, k=10, use_amp=use_amp)

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

test_f1, y_true, y_pred = evaluate_knn(model, train_loader, test_loader, k=10, use_amp=use_amp)
print(f"Test F1: {test_f1:.4f}")
print(classification_report(y_true, y_pred, target_names=labels, digits=4))

os.makedirs('submissions', exist_ok=True)
submission_path = f'submissions/{EXP_NAME}.csv'
generate_submission(model, train_loader, test_loader, submission_path, k=10, use_amp=use_amp)
print(f"\nTo score: python score.py {submission_path}")
