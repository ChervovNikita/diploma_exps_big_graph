import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
import os

RANDOM_SEED = 42
TEST_SIZE = 0.4
VAL_TEST_RATIO = 0.5
OUTPUT_DIR = 'splits'

data = pd.read_csv('/disk/10tb/home/shmelev/CR_agreed/CR_real_masks_more_labeled_veritices_agreed.csv')
print(data.head())
data['node_id1'] -= 1
data['node_id2'] -= 1

print(data.head())

labels = data['label_id1'].unique().tolist() + data['label_id2'].unique().tolist()
print(labels)
labels = sorted(list(set([l for l in labels if l != 'masked'])))
print(labels)

df = pd.concat([
    data[['node_id1', 'label_id1']].rename(columns={'node_id1': 'node_id', 'label_id1': 'label'}),
    data[['node_id2', 'label_id2']].rename(columns={'node_id2': 'node_id', 'label_id2': 'label'})
], ignore_index=True)
df = df.drop_duplicates(subset=['node_id'])

max_node = max(data['node_id1'].max(), data['node_id2'].max())
print(max_node)
print('Total number of unique nodes:', tnoun := len(np.unique(np.concatenate([data['node_id1'].to_numpy(), data['node_id2'].to_numpy()]))))
node_labels = np.full(max_node + 1, -1, dtype=np.int32)
print(node_labels)

known_nodes = []
unknown_nodes = []

for _, row in tqdm(df.iterrows(), desc="Processing nodes"):
    if row['label'] == 'masked':
        unknown_nodes.append(row['node_id'])
    else:
        node_labels[row['node_id']] = labels.index(row['label'])
        known_nodes.append(row['node_id'])

known_nodes = sorted(list(set(known_nodes)))
print('Amount of known nodes:', len(known_nodes))
unknown_nodes = sorted(list(set(unknown_nodes)))

train_nodes, temp_nodes = train_test_split(
    known_nodes, test_size=TEST_SIZE, random_state=RANDOM_SEED
)
val_nodes, test_nodes = train_test_split(
    temp_nodes, test_size=VAL_TEST_RATIO, random_state=RANDOM_SEED
)

print(nikita_tnoun := len(train_nodes) + len(val_nodes) + len(test_nodes), tnoun)
assert nikita_tnoun == len(known_nodes)
assert tnoun == len(known_nodes) + len(unknown_nodes)

train_nodes = sorted(train_nodes)
val_nodes = sorted(val_nodes)
test_nodes = sorted(test_nodes)

tn_classes = []
for tn in train_nodes:
    tn_classes.append(node_labels[tn])

print('Train class balance:', tcb := np.unique(tn_classes, return_counts=True))
print('Ground true class balance:', gtcb := np.unique(node_labels, return_counts=True))
print('Class balance test:', tcb[1] / gtcb[1][1:])



node_labels_masked = node_labels.copy()
for n in test_nodes:
    node_labels_masked[n] = -1


os.makedirs(OUTPUT_DIR, exist_ok=True)

np.save(os.path.join(OUTPUT_DIR, 'train_nodes.npy'), np.array(train_nodes, dtype=np.int32))
np.save(os.path.join(OUTPUT_DIR, 'val_nodes.npy'), np.array(val_nodes, dtype=np.int32))
np.save(os.path.join(OUTPUT_DIR, 'test_nodes.npy'), np.array(test_nodes, dtype=np.int32))
np.save(os.path.join(OUTPUT_DIR, 'unknown_nodes.npy'), np.array(unknown_nodes, dtype=np.int32))
np.save(os.path.join(OUTPUT_DIR, 'node_labels.npy'), node_labels)
np.save(os.path.join(OUTPUT_DIR, 'node_labels_masked.npy'), node_labels_masked)

graph_data_removed = data[["node_id1", "node_id2", "ibd_sum", "ibd_n"]]
graph_data_removed.to_csv(os.path.join(OUTPUT_DIR, 'edges_data.csv'), index=False)

with open(os.path.join(OUTPUT_DIR, 'labels.txt'), 'w') as f:  # here just label names
    for label in labels:
        f.write(f"{label}\n")

test_ground_truth = {
    'node_ids': np.array(test_nodes, dtype=np.int32),
    'true_labels': np.array([node_labels[n] for n in test_nodes], dtype=np.int32),
    'label_names': labels
}
with open(os.path.join(OUTPUT_DIR, 'test_ground_truth.pkl'), 'wb') as f:
    pickle.dump(test_ground_truth, f)
