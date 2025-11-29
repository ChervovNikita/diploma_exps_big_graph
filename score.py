"""
Score a submission file against ground truth.

Usage:
    python score.py submissions/exp_mask.csv
    python score.py submissions/exp_mask.csv --verbose

Submission file format (CSV):
    node_id,predicted_label
    123,Bribri
    456,Cabecar
    ...

The predicted_label can be either:
- Label name (e.g., "Bribri")
- Label index (e.g., 0)
"""
import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import os

def load_ground_truth(splits_dir='splits'):
    """Load ground truth from precomputed splits."""
    gt_path = os.path.join(splits_dir, 'test_ground_truth.pkl')
    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"Ground truth not found at '{gt_path}'. "
            "Run 'python precompute_splits.py' first."
        )
    
    with open(gt_path, 'rb') as f:
        gt = pickle.load(f)
    
    return gt['node_ids'], gt['true_labels'], gt['label_names']

def load_submission(filepath, label_names):
    """Load submission file and validate."""
    df = pd.read_csv(filepath)
    
    # Check required columns
    if 'node_id' not in df.columns:
        raise ValueError("Submission must have 'node_id' column")
    if 'predicted_label' not in df.columns:
        raise ValueError("Submission must have 'predicted_label' column")
    
    node_ids = df['node_id'].values
    predictions = df['predicted_label'].values
    
    # Convert label names to indices if needed
    label_to_idx = {name: i for i, name in enumerate(label_names)}
    pred_indices = []
    for p in predictions:
        if isinstance(p, str):
            if p not in label_to_idx:
                raise ValueError(f"Unknown label: '{p}'. Valid labels: {label_names}")
            pred_indices.append(label_to_idx[p])
        else:
            pred_indices.append(int(p))
    
    return node_ids, np.array(pred_indices)

def score_submission(submission_path, splits_dir='splits', verbose=False):
    """Score a submission against ground truth."""
    # Load ground truth
    gt_node_ids, gt_labels, label_names = load_ground_truth(splits_dir)
    gt_node_set = set(gt_node_ids)
    gt_node_to_label = {nid: lbl for nid, lbl in zip(gt_node_ids, gt_labels)}
    
    # Load submission
    sub_node_ids, sub_predictions = load_submission(submission_path, label_names)
    
    # Match predictions to ground truth
    y_true = []
    y_pred = []
    missing_nodes = []
    extra_nodes = []
    
    sub_node_set = set(sub_node_ids)
    sub_node_to_pred = {nid: pred for nid, pred in zip(sub_node_ids, sub_predictions)}
    
    # Find missing and extra nodes
    missing_nodes = list(gt_node_set - sub_node_set)
    extra_nodes = list(sub_node_set - gt_node_set)
    
    # Build aligned y_true and y_pred for nodes in both
    for node_id in gt_node_ids:
        if node_id in sub_node_to_pred:
            y_true.append(gt_node_to_label[node_id])
            y_pred.append(sub_node_to_pred[node_id])
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Print results
    print(f"\n{'='*60}")
    print(f"SCORING: {submission_path}")
    print(f"{'='*60}")
    
    print(f"\nSubmission stats:")
    print(f"  Ground truth nodes: {len(gt_node_ids)}")
    print(f"  Submitted nodes:    {len(sub_node_ids)}")
    print(f"  Matched nodes:      {len(y_true)}")
    if missing_nodes:
        print(f"  Missing nodes:      {len(missing_nodes)}")
    if extra_nodes:
        print(f"  Extra nodes:        {len(extra_nodes)}")
    
    print(f"\n{'='*60}")
    print(f"  MACRO F1 SCORE: {macro_f1:.4f}")
    print(f"{'='*60}")
    
    print(f"\nOther metrics:")
    print(f"  Micro F1:    {micro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"  Accuracy:    {(y_true == y_pred).mean():.4f}")
    
    if verbose:
        print(f"\n{'-'*60}")
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=label_names, digits=4))
        
        print(f"\n{'-'*60}")
        print("Confusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(f"Labels: {label_names}")
        print(cm)
    
    return macro_f1

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

