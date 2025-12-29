import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from common import fuzzy_f1_score
import os
import torch


def overlap_score(y_true, y_pred):
    scores = []
    for i in range(len(y_true)):
        score = np.sum(np.minimum(np.array(y_true[i]), y_pred[i].numpy()))
        scores.append(score)
    return np.mean(scores)

def score_submission(submission_path, splits_dir='splits', verbose=False):
    with open(os.path.join(splits_dir, 'test_ground_truth.pkl'), 'rb') as f:
        data = pickle.load(f)
        gt_node_ids = data['node_ids']
        gt_labels = data['true_labels']
        label_names = data['label_names']
    
    gt_node_set = set(gt_node_ids)
    gt_node_to_label = {nid: lbl for nid, lbl in zip(gt_node_ids, gt_labels)}

    df = pd.read_csv(submission_path)
    node_ids = df['node_id'].values
    predictions = df['predicted_label'].values
    label_to_idx = {name: i for i, name in enumerate(label_names)}
    node_to_pred = {nid: pred for nid, pred in zip(node_ids, predictions)}

    assert set(gt_node_to_label.keys()) == set(node_to_pred.keys())

    y_true_distr = []
    y_pred_distr = []
    y_true = []
    y_pred = []

    for node_id in gt_node_ids:
        # if gt_node_to_label[node_id].max() > 0.999:
        y_true_distr.append(gt_node_to_label[node_id])
        y_pred_distr.append(torch.tensor(eval(node_to_pred[node_id])))
        y_true.append(gt_node_to_label[node_id].argmax().item())
        y_pred.append(torch.tensor(eval(node_to_pred[node_id])).argmax().item())
        # print(y_true[-1], y_pred[-1])

    print(y_true_distr[-10:])
    print(y_pred_distr[-10:])
    print(y_true[-10:])
    print(y_pred[-10:])
    
    f1 = fuzzy_f1_score(y_true_distr, y_pred_distr, label_names)
    print('test macro fuzzy-f1 score on one class nodes:', f1)
    f1 = f1_score(y_true, y_pred, average='macro')
    print('test macro f1 on one class nodes:', f1)
    overlap = overlap_score(y_true_distr, y_pred_distr)
    print('test overlap score:', overlap)
    return f1


def main():
    parser = argparse.ArgumentParser(description='Score a submission file')
    parser.add_argument('submission', type=str, help='Path to submission CSV file')
    parser.add_argument('--splits-dir', type=str, default='splits', 
                        help='Directory containing precomputed splits')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed classification report')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.submission):
        print(f"Error: Submission file not found: {args.submission}")
        return 1
    
    try:
        macro_f1 = score_submission(args.submission, args.splits_dir, args.verbose)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == '__main__':
    exit(main())
