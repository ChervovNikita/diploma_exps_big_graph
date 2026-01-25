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
from common import MaskedGraphDataset, TAGConvModel, SingleDeviceWrapper


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
MASK_COUNT = 64
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

device = torch.device(os.environ.get('DEVICE', 'cuda:0'))

random.seed(42)
unknown_nodes_shuffled = unknown_nodes.copy()
random.shuffle(unknown_nodes_shuffled)
num_unknown_to_use = int(len(unknown_nodes_shuffled) * NUM_UNKNOWN_FRACTION)
unknown_nodes_subset = unknown_nodes_shuffled[:num_unknown_to_use]


test_dataset = MaskedGraphDataset(data, unknown_nodes_subset, train_nodes, None, test_nodes,
                                  split='test', mask_count=MASK_COUNT, num_samples=NUM_SAMPLES, node_classes=node_class)

test_loader = DataListLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

num_features = node_class.shape[1]
model = TAGConvModel(num_features=num_features, num_classes=len(labels)).to(device)
model.load_state_dict(torch.load(f'checkpoints/{EXP_NAME}_best.pt'))
model = SingleDeviceWrapper(model, device)


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

submission_df = generate_submission(model, test_loader, f'submissions/{EXP_NAME}.csv', use_amp=True)
