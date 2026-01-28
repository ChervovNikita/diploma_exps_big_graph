import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
import os
from collections import Counter


RANDOM_SEED = int(os.environ.get('RANDOM_SEED'))
assert RANDOM_SEED is not None

TEST_SIZE = 0.4
VAL_TEST_RATIO = 0.5
TYPE='balanced'
OUTPUT_DIR = os.environ.get('SPLITS_DIR')
assert OUTPUT_DIR is not None

data = pd.read_csv('../CR_gt80_semi_unlabeled_all_masks_2_classes.csv')
data['node_id1'] -= 1
data['node_id2'] -= 1

labels = [
    'Northen Russians',
    'Southern Russians'
]

max_node = max(data['node_id1'].max(), data['node_id2'].max())
node_distr = torch.load('../CR_new_node_feats.pt')
node_labels = torch.zeros(max_node + 1)

known_nodes = []
unknown_nodes = []

for i in range(len(node_distr)):
    if node_distr[i].sum() < 0.001:
        continue
    if node_distr[i].max() > 0.999:
        known_nodes.append(i)
        node_labels[i] = node_distr[i].argmax()
    else:
        unknown_nodes.append(i)

known_nodes = sorted(list(set(known_nodes)))
unknown_nodes = sorted(list(set(unknown_nodes)))

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

# connected_nodes = set()
# for _, row in tqdm(data.iterrows(), total=len(data)):
#     connected_nodes.add(row['node_id1'])
#     connected_nodes.add(row['node_id2'])

# was_test = len(test_nodes)
# test_nodes = sorted([node for node in test_nodes if node in connected_nodes])
# print(f"Filtered test nodes: {len(test_nodes)} -> {was_test}")

node_distr_masked = node_distr
for n in test_nodes:
    node_distr_masked[n] = 1/len(labels)

# for n in unknown_nodes:
#     node_distr_masked[n] = 1/len(labels)


os.makedirs(OUTPUT_DIR, exist_ok=True)

np.save(os.path.join(OUTPUT_DIR, 'train_nodes.npy'), np.array(train_nodes, dtype=np.int32))
np.save(os.path.join(OUTPUT_DIR, 'val_nodes.npy'), np.array(val_nodes, dtype=np.int32))
np.save(os.path.join(OUTPUT_DIR, 'test_nodes.npy'), np.array(test_nodes, dtype=np.int32))
np.save(os.path.join(OUTPUT_DIR, 'unknown_nodes.npy'), np.array(unknown_nodes, dtype=np.int32))
torch.save(node_distr_masked, os.path.join(OUTPUT_DIR, 'node_distr_masked.pt'))

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
