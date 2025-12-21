import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import os

SPLITS_DIR = 'splits'
SUBMISSION_PATH = 'submissions/simple_masking.csv'

with open(os.path.join(SPLITS_DIR, 'test_ground_truth.pkl'), 'rb') as f:
    data = pickle.load(f)
    gt_node_ids = data['node_ids'] # test nodes
    gt_labels = data['true_labels'] # test node labels
    label_names = data['label_names'] # 4 labels name

assert not -1 in gt_labels

gt_node_set = list(gt_node_ids) # set(gt_node_ids)
gt_node_to_label = {nid: lbl for nid, lbl in zip(gt_node_ids, gt_labels)}

df = pd.read_csv(SUBMISSION_PATH)
node_ids = df['node_id'].values
predictions = df['predicted_label'].values
label_to_idx = {name: i for i, name in enumerate(label_names)}
node_to_pred = {nid: pred for nid, pred in zip(node_ids, predictions)}

assert len(node_to_pred) == len(gt_node_to_label)

assert set(gt_node_to_label.keys()) == set(node_to_pred.keys())

y_true = []
y_pred = []

for node_id in gt_node_ids:
    y_true.append(gt_node_to_label[node_id])
    y_pred.append(label_to_idx[node_to_pred[node_id]])

print('test macro f1:', f1_score(y_true, y_pred, average='macro'))
print('test classification report:', classification_report(y_true, y_pred, target_names=label_names, digits=4))
print('test confusion matrix:', confusion_matrix(y_true, y_pred, labels=range(len(label_names))))
