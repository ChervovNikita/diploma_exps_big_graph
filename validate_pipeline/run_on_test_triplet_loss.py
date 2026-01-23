import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import TAGConv, GraphNorm, DataParallel as GeoDataParallel
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score, classification_report
import random
import pickle
import os
from common import SingleDeviceWrapper, TripletGraphDataset, compute_triplet_embeddings, knn_predict, TAGConvModelEmbeddings
import joblib


RANDOM_SEED = os.environ.get('RANDOM_SEED')
assert RANDOM_SEED is not None
RANDOM_SEED = int(RANDOM_SEED)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

EXP_NAME = os.environ.get('EXP_NAME')
assert EXP_NAME is not None

SPLITS_DIR = os.environ.get('SPLITS_DIR')
NUM_UNKNOWN_FRACTION = 1.0
TRAIN_MASK_COUNT = 64
MASK_COUNT = 1
NUM_SAMPLES = 500

NUM_WORKERS = 4

version = os.environ.get('VERSION', 'v1')

train_nodes = np.load(os.path.join(SPLITS_DIR, 'train_nodes.npy')).tolist()
val_nodes = np.load(os.path.join(SPLITS_DIR, 'val_nodes.npy')).tolist()
test_nodes = np.load(os.path.join(SPLITS_DIR, 'test_nodes.npy')).tolist()
unknown_nodes = np.load(os.path.join(SPLITS_DIR, 'unknown_nodes.npy')).tolist()

with open(os.path.join(SPLITS_DIR, 'labels.txt'), 'r') as f:
    labels = [line.strip() for line in f]

if version == 'v1':
    node_labels = np.load(os.path.join(SPLITS_DIR, 'node_labels_masked.npy'))  # use this masked labels

    print('test labels:', set(node_labels[test_nodes].tolist()))

    max_node = max(max(train_nodes), max(val_nodes), max(test_nodes), max(unknown_nodes))
    node_class = np.zeros((max_node + 1, len(labels)))
    for n in range(max_node + 1):
        if node_labels[n] >= 0:
            node_class[n, node_labels[n]] = 1
        else:
            node_class[n, :] = np.ones(len(labels)) / len(labels)

elif version == 'v2':
    node_class = torch.load(os.path.join(SPLITS_DIR, 'node_distr_masked.pt'))
    node_labels = torch.tensor([torch.argmax(t) for t in node_class])

data = pd.read_csv(os.path.join(SPLITS_DIR, 'edges_data.csv'))

device = torch.device("cuda:0")

random.seed(42)
unknown_nodes_shuffled = unknown_nodes.copy()
random.shuffle(unknown_nodes_shuffled)
num_unknown_to_use = int(len(unknown_nodes_shuffled) * NUM_UNKNOWN_FRACTION)
unknown_nodes_subset = unknown_nodes_shuffled[:num_unknown_to_use]

train_nodes_by_class = {c: [] for c in range(len(labels))}
for n in train_nodes:
    if node_labels[n] >= 0:
        train_nodes_by_class[node_labels[n]].append(n)

train_eval_dataset = TripletGraphDataset(data, unknown_nodes_subset, train_nodes, val_nodes, None,
                                    train_nodes_by_class, split='train_eval', mask_count=TRAIN_MASK_COUNT, num_samples=NUM_SAMPLES, node_classes=node_class)

test_dataset = TripletGraphDataset(data, unknown_nodes_subset, train_nodes, None, test_nodes,
                                  train_nodes_by_class, split='test', mask_count=MASK_COUNT, num_samples=NUM_SAMPLES, node_classes=node_class)

train_eval_loader = DataListLoader(train_eval_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataListLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

num_features = node_class.shape[1]
model = TAGConvModelEmbeddings(num_classes=len(labels)).to(device)
model.load_state_dict(torch.load(f'checkpoints/{EXP_NAME}_best.pt'))
model = SingleDeviceWrapper(model, device)

state = joblib.load(f'checkpoints/{EXP_NAME}_best.pkl')
best_val_f1 = state['best_val_f1']
best_val_k = state['best_val_k']
best_val_weighted = state['best_val_weighted']

def generate_submission(model, loader, output_path, use_amp=True):
    model.eval()
    all_node_ids = []
    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Generating submission', leave=False):
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, batch_data = model(batch)
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
    return submission_df


def generate_submission(model, train_loader, test_loader, output_path, k, weighted):
    train_emb, train_y = compute_triplet_embeddings(model, train_loader)
    
    model.eval()
    all_node_ids, all_predictions = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Generating submission', leave=False):
            embeddings, batch_data = model(batch)
            predict_mask = batch_data.predict_mask
            masked_node_ids = batch_data.masked_node_ids

            emb = embeddings[predict_mask].cpu()
            preds = knn_predict(train_emb, train_y, emb, k=k, weighted=weighted)
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


submission_df = generate_submission(model, train_eval_loader, test_loader, f'submissions/{EXP_NAME}.csv', best_val_k, best_val_weighted)
