import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
import os

RANDOM_SEED = os.environ.get('RANDOM_SEED')
assert RANDOM_SEED is not None
RANDOM_SEED = int(RANDOM_SEED)

TEST_SIZE = 0.4
VAL_TEST_RATIO = 0.5
TYPE='balanced'
OUTPUT_DIR = f'splits_{TYPE}'

data = pd.read_csv('../CR_real_masks_more_labeled_veritices_agreed.csv')
data['node_id1'] -= 1
data['node_id2'] -= 1

labels = data['label_id1'].unique().tolist() + data['label_id2'].unique().tolist()
labels = sorted(list(set([l for l in labels if l != 'masked'])))

df = pd.concat([
    data[['node_id1', 'label_id1']].rename(columns={'node_id1': 'node_id', 'label_id1': 'label'}),
    data[['node_id2', 'label_id2']].rename(columns={'node_id2': 'node_id', 'label_id2': 'label'})
], ignore_index=True)
df = df.drop_duplicates(subset=['node_id'])

max_node = max(data['node_id1'].max(), data['node_id2'].max())
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

if TYPE == 'not_balanced':
    train_nodes, temp_nodes = train_test_split(
        known_nodes, test_size=TEST_SIZE, random_state=RANDOM_SEED,
        # stratify=[node_labels[n] for n in known_nodes]
    )
    val_nodes, test_nodes = train_test_split(
        temp_nodes, test_size=VAL_TEST_RATIO, random_state=RANDOM_SEED,
        # stratify=[node_labels[n] for n in temp_nodes]
    )
else:
    train_nodes, temp_nodes = train_test_split(
        known_nodes, test_size=TEST_SIZE, random_state=RANDOM_SEED,
        stratify=[node_labels[n] for n in known_nodes]
    )
    val_nodes, test_nodes = train_test_split(
        temp_nodes, test_size=VAL_TEST_RATIO, random_state=RANDOM_SEED,
        stratify=[node_labels[n] for n in temp_nodes]
    )

for i in range(len(labels)):
    train_count = sum(1 for n in train_nodes if node_labels[n] == i)
    val_count = sum(1 for n in val_nodes if node_labels[n] == i)
    test_count = sum(1 for n in test_nodes if node_labels[n] == i)
    total_count = train_count + val_count + test_count
    print(f"Label {labels[i]}: Train={train_count / total_count}, Val={val_count / total_count}, Test={test_count / total_count}")

train_nodes = sorted(train_nodes)
val_nodes = sorted(val_nodes)
test_nodes = sorted(test_nodes)

connected_nodes = set()
for _, row in data.iterrows():
    connected_nodes.add(row['node_id1'])
    connected_nodes.add(row['node_id2'])

was_test = len(test_nodes)
test_nodes = sorted([node for node in test_nodes if node in connected_nodes])
print(f"Filtered test nodes: {len(test_nodes)} -> {was_test}")

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
