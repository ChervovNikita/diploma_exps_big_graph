import random
import torch
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.nn import TAGConv, GraphNorm
from torch_geometric.utils import to_torch_csr_tensor
import torch.nn.functional as F
import numpy as np


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
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

        node_features = torch.tensor(self.node_classes[sorted_nodes], dtype=torch.float)

        mask_indices = [node_mapping[n] for n in mask_nodes if n in node_mapping]
        predict_mask = torch.zeros(len(sorted_nodes), dtype=torch.bool)
        predict_mask[mask_indices] = True

        node_features_masked = node_features.clone()
        node_features_masked[predict_mask] = torch.ones(self.node_classes.shape[1]) / self.node_classes.shape[1]

        if self.split in ['train', 'val']:
            y = torch.tensor([np.argmax(self.node_classes[n]) for n in sorted_nodes], dtype=torch.long)
        else:
            y = None
        
        ibd = subgraph_edges['ibd_sum'].values
        edge_weights = torch.tensor(list(ibd) + list(ibd), dtype=torch.float)

        # masked_node_ids = torch.tensor([mask_nodes[mask_indices.index(i)] for i in mask_indices], dtype=torch.long)
        mask_nodes_filtered = [n for n in mask_nodes if n in node_mapping]
        mask_nodes_in_graph_order = sorted(mask_nodes_filtered, key=lambda n: node_mapping[n])
        masked_node_ids = torch.tensor(mask_nodes_in_graph_order, dtype=torch.long)

        return Data(
            x=node_features_masked,
            edge_index=edge_index,
            y=y,
            weight=edge_weights,
            num_classes=self.node_classes.shape[1],
            predict_mask=predict_mask,
            masked_node_ids=masked_node_ids
        )


def compute_graph_based_features(sorted_nodes, subgraph_edges, node_class, num_classes):
    num_nodes = len(sorted_nodes)
    node_mapping = {node: i for i, node in enumerate(sorted_nodes)}

    # for n in sorted_nodes:
    #     print(node_class[n])
    #     print(np.max(node_class[n]))
    #     print(np.argmax(node_class[n]))

    node_labels = np.array([
        torch.argmax(node_class[n]) if torch.max(node_class[n]) > 0.5 else -1
        for n in sorted_nodes
    ])
    
    features = np.zeros((num_nodes, 5 * num_classes))
    
    for _, row in subgraph_edges.iterrows():
        n1, n2, w = row['node_id1'], row['node_id2'], row['ibd_sum']
        if n1 in node_mapping and n2 in node_mapping:
            i, j = node_mapping[n1], node_mapping[n2]
            if node_labels[j] >= 0:
                c = node_labels[j]
                features[i, c] += 1
                features[i, num_classes + c] += w
                features[i, 3*num_classes + c] = max(features[i, 3*num_classes + c], w)
                features[i, 4*num_classes + c] += 1
            if node_labels[i] >= 0:
                c = node_labels[i]
                features[j, c] += 1
                features[j, num_classes + c] += w
                features[j, 3*num_classes + c] = max(features[j, 3*num_classes + c], w)
                features[j, 4*num_classes + c] += 1

    for c in range(num_classes):
        count_col = c
        sum_col = num_classes + c
        mask = features[:, count_col] > 0
        features[mask, sum_col] /= features[mask, count_col]
    
    return torch.tensor(features, dtype=torch.float)

class MaskedGraphDatasetGraphBased(Dataset):
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
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

        node_features = torch.tensor(self.node_classes[sorted_nodes], dtype=torch.float)

        mask_indices = [node_mapping[n] for n in mask_nodes if n in node_mapping]
        predict_mask = torch.zeros(len(sorted_nodes), dtype=torch.bool)
        predict_mask[mask_indices] = True

        node_features_masked = node_features.clone()
        node_features_masked[predict_mask] = torch.ones(self.node_classes.shape[1]) / self.node_classes.shape[1]

        node_features_graph_based = compute_graph_based_features(list(range(len(sorted_nodes))), subgraph_edges, node_features_masked, self.node_classes.shape[1])

        node_features = torch.cat([node_features_graph_based, node_features_masked], dim=1)

        if self.split in ['train', 'val']:
            y = torch.tensor([np.argmax(self.node_classes[n]) for n in sorted_nodes], dtype=torch.long)
        else:
            y = None
        
        ibd = subgraph_edges['ibd_sum'].values
        edge_weights = torch.tensor(list(ibd) + list(ibd), dtype=torch.float)

        # masked_node_ids = torch.tensor([mask_nodes[mask_indices.index(i)] for i in mask_indices], dtype=torch.long)
        mask_nodes_filtered = [n for n in mask_nodes if n in node_mapping]
        mask_nodes_in_graph_order = sorted(mask_nodes_filtered, key=lambda n: node_mapping[n])
        masked_node_ids = torch.tensor(mask_nodes_in_graph_order, dtype=torch.long)

        return Data(
            x=node_features,
            edge_index=edge_index,
            y=y,
            weight=edge_weights,
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
        x, edge_index, edge_weight = data.x, data.edge_index, data.weight

        num_nodes = x.size(0)
        adj = to_torch_csr_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes))

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

