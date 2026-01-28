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
from common import MaskedGraphDataset, TAGConvModelBase, SingleDeviceWrapper

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

LR = 0.0001
WD = 0.0001
EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 1
NUM_WORKERS = 16

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

device = torch.device(os.environ.get('DEVICE', 'cuda:1'))

random.seed(42)
unknown_nodes_shuffled = unknown_nodes.copy()
random.shuffle(unknown_nodes_shuffled)
num_unknown_to_use = int(len(unknown_nodes_shuffled) * NUM_UNKNOWN_FRACTION)
unknown_nodes_subset = unknown_nodes_shuffled[:num_unknown_to_use]


train_dataset = MaskedGraphDataset(data, unknown_nodes_subset, train_nodes, val_nodes, None,
                                    split='train', mask_count=MASK_COUNT, num_samples=NUM_SAMPLES, node_classes=node_class)
val_dataset = MaskedGraphDataset(data, unknown_nodes_subset, train_nodes, val_nodes, None,
                                  split='val', mask_count=MASK_COUNT, num_samples=NUM_SAMPLES, node_classes=node_class)

train_loader = DataListLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataListLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

num_features = node_class.shape[1]
model = SingleDeviceWrapper(TAGConvModelBase(num_features=num_features, num_classes=len(labels)).to(device), device)


def evaluate(model, loader, use_amp=True):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval', leave=False):
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, batch_data = model(batch)
                predict_mask = batch_data.predict_mask
            preds = torch.argmax(logits[predict_mask], dim=-1).cpu().tolist()
            trues = batch_data.y[predict_mask].cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(trues)
    return f1_score(y_true, y_pred, average='macro'), y_true, y_pred


class_counts = [sum(1 for n in train_nodes if node_labels[n] == c) for c in range(len(labels))]
class_weights = torch.tensor([max(class_counts) / c for c in class_counts], dtype=torch.float).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = StepLR(optimizer, step_size=50, gamma=0.95)
scaler = torch.amp.GradScaler(device)
use_amp = False #  device.type == 'cuda'

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
            logits, batch_data = model(batch)
            predict_mask = batch_data.predict_mask
            masked_node_ids = batch_data.masked_node_ids
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
        torch.save(best_state, f'checkpoints/{EXP_NAME}_best.pt')
        print(f"[Epoch {epoch}] val_f1={best_val_f1:.4f} ↑ | loss={np.mean(losses):.4f}")
    else:
        patience_counter += 1
        print(f"[Epoch {epoch}] val_f1={val_f1:.4f} | loss={np.mean(losses):.4f} | patience={patience_counter}")
