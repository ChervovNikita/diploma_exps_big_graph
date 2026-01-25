import pandas as pd

data = pd.read_csv('CR_gt80_semi_unlabeled_all_masks_final.csv')
data['node_id1'] -= 1
data['node_id2'] -= 1

max_node = max(data.node_id1.max(), data.node_id2.max()) + 1

data1 = data.rename(columns={'node_id1': 'node', 'label1_Northen Russians': 'label_Northen Russians', 'label1_Southern Russians': 'label_Southern Russians', 'label1_Belarusians': 'label_Belarusians', 'label1_Ukranians': 'label_Ukranians'})
data2 = data.rename(columns={'node_id2': 'node', 'label2_Northen Russians': 'label_Northen Russians', 'label2_Southern Russians': 'label_Southern Russians', 'label2_Belarusians': 'label_Belarusians', 'label2_Ukranians': 'label_Ukranians'})

data1 = data1[['node', 'label_Northen Russians', 'label_Southern Russians', 'label_Belarusians', 'label_Ukranians']]
data2 = data2[['node', 'label_Northen Russians', 'label_Southern Russians', 'label_Belarusians', 'label_Ukranians']]
data_nodes = pd.concat([data1, data2])
data_nodes = data_nodes.drop_duplicates(keep='first')

import torch
import numpy as np

node_feats = torch.zeros((max_node, 4))
visited = torch.zeros(max_node)

# Belarusians
# Northen Russians
# Southern Russians
# Ukranians


for _, row in data_nodes.iterrows():
    n = row['node']
    t = [row['label_Belarusians'], row['label_Northen Russians'], row['label_Southern Russians'], row['label_Ukranians']]
    found = False
    for i in range(4):
        if t[i] >= 3:
            found = True
            node_feats[n, i] = 1
            break
    if not found:
        node_feats[n] = torch.tensor(t) / 4
        node_feats[n] += (4 - sum(t)) * 1/16

torch.save(node_feats, 'CR_new_node_feats.pt')