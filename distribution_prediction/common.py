import random
import torch
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.nn import TAGConv, GraphNorm
from torch_sparse import SparseTensor
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from torch_geometric.utils import to_torch_csr_tensor


class MaskedGraphDataset(Dataset):
    def __init__(self, data_df, unknown_nodes_subset, train_nodes, val_nodes, test_nodes, split, mask_count, num_samples, node_classes):
        super().__init__()
        self.data_df = data_df
        self.unknown_nodes_subset = unknown_nodes_subset
        self.train_nodes = train_nodes
        self.val_nodes = val_nodes
        self.test_nodes = test_nodes
        self.split = split
        self.mask_count = mask_count
        self.num_samples = num_samples
        self.node_classes = node_classes

        if split == 'train':
            self.target_nodes = train_nodes
        elif split == 'val':
            self.target_nodes = val_nodes
        else:
            self.target_nodes = test_nodes

    def len(self):
        if self.split == 'train':
            return self.num_samples
        else:
            return (len(self.target_nodes) + self.mask_count - 1) // self.mask_count

    def get(self, idx):
        nodes_to_include = self.train_nodes.copy()
        nodes_to_include.extend(self.unknown_nodes_subset)
        if self.split == 'train':
            mask_nodes = random.sample(self.train_nodes, self.mask_count)
        else:
            start_idx = idx * self.mask_count
            end_idx = min(start_idx + self.mask_count, len(self.target_nodes))
            mask_nodes = self.target_nodes[start_idx:end_idx]
            for n in mask_nodes:
                assert n not in nodes_to_include
                nodes_to_include.append(n)
        
        nodes_set = set(nodes_to_include)
        mask = (self.data_df['node_id1'].isin(nodes_set) & self.data_df['node_id2'].isin(nodes_set))
        subgraph_edges = self.data_df[mask]

        sorted_nodes = sorted(nodes_to_include)
        node_mapping = {node: i for i, node in enumerate(sorted_nodes)}

        src = [node_mapping[n] for n in subgraph_edges['node_id1'].values]
        dst = [node_mapping[n] for n in subgraph_edges['node_id2'].values]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        node_features = torch.tensor(self.node_classes[sorted_nodes], dtype=torch.float)

        mask_indices = [node_mapping[n] for n in mask_nodes if n in node_mapping]
        predict_mask = torch.zeros(len(sorted_nodes), dtype=torch.bool)
        predict_mask[mask_indices] = True

        node_features_masked = node_features.clone()
        node_features_masked[predict_mask] = torch.ones(self.node_classes.shape[1]) / self.node_classes.shape[1]

        if self.split in ['train', 'val']:
            y = torch.tensor(self.node_classes[sorted_nodes], dtype=torch.float)
        else:
            y = None

        mask_nodes_filtered = [n for n in mask_nodes if n in node_mapping]
        mask_nodes_in_graph_order = sorted(mask_nodes_filtered, key=lambda n: node_mapping[n])
        masked_node_ids = torch.tensor(mask_nodes_in_graph_order, dtype=torch.long)

        return Data(
            x=node_features_masked,
            edge_index=edge_index,
            y=y,
            num_classes=self.node_classes.shape[1],
            predict_mask=predict_mask,
            masked_node_ids=masked_node_ids
        )


class TAGConvModel(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=512):
        super().__init__()
        self.first_linear = torch.nn.Linear(num_features, hidden_dim)
        self.conv1 = TAGConv(hidden_dim, hidden_dim)
        self.conv2 = TAGConv(hidden_dim, hidden_dim)
        self.conv3 = TAGConv(hidden_dim, hidden_dim)
        self.n1 = GraphNorm(hidden_dim)
        self.n2 = GraphNorm(hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        num_nodes = x.size(0)
        adj = to_torch_csr_tensor(edge_index, size=(num_nodes, num_nodes))

        x = self.first_linear(x)

        x_ = x.clone()
        x = F.elu(self.conv1(x, adj))
        x = x_ + x
        x = self.n1(x)
        del x_

        x_ = x.clone()
        x = F.elu(self.conv2(x, adj))
        x = x_ + x
        x = self.n2(x)
        del x_

        x_ = x.clone()
        x = F.elu(self.conv3(x, adj))
        x = x_ + x
        del x_, adj

        return self.linear(x)


class SingleDeviceWrapper(torch.nn.Module):  # so one and multi gpu have the same interface
    def __init__(self, module, device):
        super().__init__()
        self.module = module
        self.device = device

    def forward(self, data_list):
        if isinstance(data_list, list):
            batch = Batch.from_data_list(data_list).to(self.device, non_blocking=True)
        else:
            batch = data_list.to(self.device, non_blocking=True)
        return self.module(batch), batch


def fuzzy_f1_score(y_true, y_pred, labels):
    class_f1 = []

    y_true = torch.tensor(y_true)
    y_pred = torch.concat([p for p in y_pred]).detach().cpu()

    for i in range(len(labels)):
        precision = torch.minimum(y_true[:, i], y_pred[:, i]).sum() / y_pred[:, i].sum()
        recall = torch.minimum(y_true[:, i], y_pred[:, i]).sum() / y_true[:, i].sum()
        f1 = 2 * precision * recall / (precision + recall)
        class_f1.append(f1)
    
    del y_true, y_pred
    return np.mean(class_f1)


def evaluate(model, loader, use_amp=True, labels=None):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval', leave=False):
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, batch_data = model(batch)
                predict_mask = batch_data.predict_mask
            preds = torch.softmax(logits[predict_mask], dim=-1).cpu()
            trues = batch_data.y[predict_mask].cpu().tolist()
            y_pred.append(preds)
            y_true.extend(trues)
            del logits, batch_data, preds
            torch.cuda.empty_cache()
    return fuzzy_f1_score(y_true, y_pred, labels), y_true, y_pred


def generate_submission(model, loader, output_path, use_amp=True, labels=None):
    """Generate submission file with predictions for test nodes."""
    model.eval()
    all_node_ids = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Generating submission', leave=False):
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, batch_data = model(batch)
                predict_mask = batch_data.predict_mask
                masked_node_ids = batch_data.masked_node_ids
            preds = torch.softmax(logits[predict_mask], dim=-1).cpu().tolist()
            node_ids = masked_node_ids.cpu().tolist()
            all_node_ids.extend(node_ids)
            all_predictions.extend(preds)

    submission_df = pd.DataFrame({
        'node_id': all_node_ids,
        'predicted_label': all_predictions
    })

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    return submission_df