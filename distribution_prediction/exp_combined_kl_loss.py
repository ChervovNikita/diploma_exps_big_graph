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
EXP_NAME = f'exp_combined_kl_loss_{NUM_UNKNOWN_FRACTION}'
SPLITS_DIR = 'splits_jews'
DEVICE = 'cuda:1'
HAS_UNKNOWN = False

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

MASK_COUNT = 64
NUM_SAMPLES = 500

LR, WD, EPOCHS, PATIENCE = 0.0001, 0.0001, 10, 5
BATCH_SIZE = 1

assert os.path.exists(SPLITS_DIR)

train_nodes = np.load(os.path.join(SPLITS_DIR, 'train_nodes.npy')).tolist()
val_nodes = np.load(os.path.join(SPLITS_DIR, 'val_nodes.npy')).tolist()
test_nodes = np.load(os.path.join(SPLITS_DIR, 'test_nodes.npy')).tolist()
if HAS_UNKNOWN:
    unknown_nodes = np.load(os.path.join(SPLITS_DIR, 'unknown_nodes.npy')).tolist()
else:
    unknown_nodes = []
node_labels = np.load(os.path.join(SPLITS_DIR, 'node_labels_masked.npy'))

with open(os.path.join(SPLITS_DIR, 'labels.txt'), 'r') as f:
    labels = [line.strip() for line in f]

print(f"Labels: {labels}")
print(f"Train nodes: {len(train_nodes)}, Val nodes: {len(val_nodes)}, Test nodes: {len(test_nodes)}")

data = pd.read_csv(os.path.join(SPLITS_DIR, 'edges_data.csv'))

device = torch.device(DEVICE)
print(f"Device: {device}")

random.seed(42)
unknown_nodes_shuffled = unknown_nodes.copy()
random.shuffle(unknown_nodes_shuffled)
num_unknown_to_use = int(len(unknown_nodes_shuffled) * NUM_UNKNOWN_FRACTION)
unknown_nodes_subset = unknown_nodes_shuffled[:num_unknown_to_use]
print(f"Using {num_unknown_to_use} unknown nodes ({NUM_UNKNOWN_FRACTION*100:.0f}%)")


train_dataset = MaskedGraphDataset(
    data, unknown_nodes_subset, train_nodes, val_nodes, test_nodes,
    split='train', mask_count=MASK_COUNT, num_samples=NUM_SAMPLES, node_classes=node_labels
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

# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.KLDivLoss(reduction='batchmean')
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = StepLR(optimizer, step_size=50, gamma=0.95)

scaler = torch.amp.GradScaler('cuda')
use_amp = False

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
            log_probs = F.log_softmax(logits[predict_mask], dim=-1)
            loss = criterion(log_probs, batch_data.y[predict_mask])
            # loss = criterion(logits[predict_mask], batch_data.y[predict_mask])
        
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

