"""
Precompute train/val/test splits and save to disk.
This ensures all experiments use the exact same splits.

Outputs:
- splits/train_nodes.npy: training node IDs
- splits/val_nodes.npy: validation node IDs  
- splits/test_nodes.npy: test node IDs
- splits/unknown_nodes.npy: nodes with masked labels
- splits/node_labels.npy: ground truth labels (ALL nodes) - FOR SCORING ONLY
- splits/node_labels_masked.npy: labels with test nodes masked (-1) - FOR EXPERIMENTS
- splits/labels.txt: label names (one per line)
- splits/metadata.pkl: additional metadata (max_node, etc.)
- splits/test_ground_truth.pkl: test node IDs and their true labels (for scoring)
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
import os

# ============== Config ==============
RANDOM_SEED = 42
TEST_SIZE = 0.4  # 40% for val+test
VAL_TEST_RATIO = 0.5  # 50% of remaining goes to test (so 20% val, 20% test)
OUTPUT_DIR = 'splits'

# ============== Load Data ==============
print("Loading data...")
data = pd.read_csv('CR_real_masks_more_labeled_veritices_agreed.csv')
data['node_id1'] -= 1
data['node_id2'] -= 1

# Get unique labels (excluding 'masked')
labels = data['label_id1'].unique().tolist() + data['label_id2'].unique().tolist()
labels = sorted(list(set([l for l in labels if l != 'masked'])))
print(f"Labels ({len(labels)}): {labels}")

# Create node dataframe
df = pd.concat([
    data[['node_id1', 'label_id1']].rename(columns={'node_id1': 'node_id', 'label_id1': 'label'}),
    data[['node_id2', 'label_id2']].rename(columns={'node_id2': 'node_id', 'label_id2': 'label'})
], ignore_index=True)
df = df.drop_duplicates(subset=['node_id'])

# Process nodes
max_node = max(data['node_id1'].max(), data['node_id2'].max())
print(f"Max node ID: {max_node}")

# node_labels: -1 for unknown/masked, label_idx for known
node_labels = np.full(max_node + 1, -1, dtype=np.int32)
known_nodes = []
unknown_nodes = []

for _, row in tqdm(df.iterrows(), desc="Processing nodes"):
    if row['label'] == 'masked':
        unknown_nodes.append(row['node_id'])
    else:
        node_labels[row['node_id']] = labels.index(row['label'])
        known_nodes.append(row['node_id'])

known_nodes = sorted(list(set(known_nodes)))
unknown_nodes = sorted(list(set(unknown_nodes)))

print(f"Known nodes: {len(known_nodes)}, Unknown nodes: {len(unknown_nodes)}")

# ============== Create Splits ==============
train_nodes, temp_nodes = train_test_split(
    known_nodes, test_size=TEST_SIZE, random_state=RANDOM_SEED
)
val_nodes, test_nodes = train_test_split(
    temp_nodes, test_size=VAL_TEST_RATIO, random_state=RANDOM_SEED
)

train_nodes = sorted(train_nodes)
val_nodes = sorted(val_nodes)
test_nodes = sorted(test_nodes)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_nodes)} ({100*len(train_nodes)/len(known_nodes):.1f}%)")
print(f"  Val:   {len(val_nodes)} ({100*len(val_nodes)/len(known_nodes):.1f}%)")
print(f"  Test:  {len(test_nodes)} ({100*len(test_nodes)/len(known_nodes):.1f}%)")

# ============== Class Distribution ==============
def get_class_dist(nodes):
    counts = {}
    for n in nodes:
        lbl = labels[node_labels[n]]
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts

print("\nClass distributions:")
for split_name, split_nodes in [('Train', train_nodes), ('Val', val_nodes), ('Test', test_nodes)]:
    dist = get_class_dist(split_nodes)
    print(f"  {split_name}: {dist}")

# ============== Create Masked Labels (for experiments - no data leak) ==============
# node_labels has true labels for ALL nodes
# node_labels_masked has test nodes set to -1 (masked)
node_labels_masked = node_labels.copy()
for n in test_nodes:
    node_labels_masked[n] = -1  # Mask test node labels

print(f"\nMasked {len(test_nodes)} test nodes in node_labels_masked")

# ============== Save ==============
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.save(os.path.join(OUTPUT_DIR, 'train_nodes.npy'), np.array(train_nodes, dtype=np.int32))
np.save(os.path.join(OUTPUT_DIR, 'val_nodes.npy'), np.array(val_nodes, dtype=np.int32))
np.save(os.path.join(OUTPUT_DIR, 'test_nodes.npy'), np.array(test_nodes, dtype=np.int32))
np.save(os.path.join(OUTPUT_DIR, 'unknown_nodes.npy'), np.array(unknown_nodes, dtype=np.int32))

# Save BOTH versions of labels
np.save(os.path.join(OUTPUT_DIR, 'node_labels.npy'), node_labels)  # Full ground truth (for scoring)
np.save(os.path.join(OUTPUT_DIR, 'node_labels_masked.npy'), node_labels_masked)  # For experiments

# Save labels as text (easy to read)
with open(os.path.join(OUTPUT_DIR, 'labels.txt'), 'w') as f:
    for label in labels:
        f.write(f"{label}\n")

# Save metadata
metadata = {
    'max_node': max_node,
    'num_labels': len(labels),
    'num_known': len(known_nodes),
    'num_unknown': len(unknown_nodes),
    'random_seed': RANDOM_SEED,
}
with open(os.path.join(OUTPUT_DIR, 'metadata.pkl'), 'wb') as f:
    pickle.dump(metadata, f)

# Save test ground truth separately (for scoring)
# This includes node_id -> true label mapping for test nodes
test_ground_truth = {
    'node_ids': np.array(test_nodes, dtype=np.int32),
    'true_labels': np.array([node_labels[n] for n in test_nodes], dtype=np.int32),
    'label_names': labels
}
with open(os.path.join(OUTPUT_DIR, 'test_ground_truth.pkl'), 'wb') as f:
    pickle.dump(test_ground_truth, f)

print(f"\nSaved splits to '{OUTPUT_DIR}/':")
print(f"  - train_nodes.npy ({len(train_nodes)} nodes)")
print(f"  - val_nodes.npy ({len(val_nodes)} nodes)")
print(f"  - test_nodes.npy ({len(test_nodes)} nodes)")
print(f"  - unknown_nodes.npy ({len(unknown_nodes)} nodes)")
print(f"  - node_labels.npy (ALL labels - FOR SCORING ONLY)")
print(f"  - node_labels_masked.npy (test masked - FOR EXPERIMENTS)")
print(f"  - labels.txt ({len(labels)} classes)")
print(f"  - metadata.pkl")
print(f"  - test_ground_truth.pkl (for scoring)")

print("\nDone!")

