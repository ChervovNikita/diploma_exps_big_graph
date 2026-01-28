import pandas as pd

data = pd.read_csv('CR_gt80_semi_unlabeled_all_masks_2_classes.csv')
data['node_id1'] -= 1
data['node_id2'] -= 1

max_node = max(data.node_id1.max(), data.node_id2.max()) + 1

data1 = data.rename(columns={'node_id1': 'node', 'label1_Northen Russians': 'label_Northen Russians', 'label1_Southern Russians': 'label_Southern Russians'})
data2 = data.rename(columns={'node_id2': 'node', 'label2_Northen Russians': 'label_Northen Russians', 'label2_Southern Russians': 'label_Southern Russians'})

data1 = data1[['node', 'label_Northen Russians', 'label_Southern Russians']]
data2 = data2[['node', 'label_Northen Russians', 'label_Southern Russians']]
data_nodes = pd.concat([data1, data2])
data_nodes = data_nodes.drop_duplicates(keep='first')

import torch
import numpy as np

node_feats = torch.zeros((max_node, 2))
visited = torch.zeros(max_node)

# Northen Russians
# Southern Russians


for _, row in data_nodes.iterrows():
    n = row['node']
    t = [row['label_Northen Russians'], row['label_Southern Russians']]
    found = False
    for i in range(len(t)):
        if t[i] >= 3:
            found = True
            node_feats[n, i] = 1
            break
    if not found:
        node_feats[n] = torch.tensor(t) / 4
        node_feats[n] += (4 - sum(t)) * 1/4/len(t)

torch.save(node_feats, 'CR_new_node_feats.pt')
