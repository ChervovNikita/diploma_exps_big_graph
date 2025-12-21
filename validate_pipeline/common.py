import random
import torch
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.nn import TAGConv, GraphNorm
import torch.nn.functional as F
import numpy as np
import networkx as nx

from torch_geometric.utils import to_networkx, from_networkx
import pandas as pd


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
            print('Dataset len (self.num_samples): ', self.num_samples, self.split)
            return self.num_samples
        else:
            print('Dataset len: ', (len(self.target_nodes) + self.mask_count - 1) // self.mask_count)
            return (len(self.target_nodes) + self.mask_count - 1) // self.mask_count

    def get(self, idx):
        print('IDX', idx)
        nodes_to_include = self.train_nodes.copy()
        nodes_to_include.extend(self.unknown_nodes_subset)
        assert (len(self.train_nodes) + len(self.unknown_nodes_subset)) == len(nodes_to_include) == len(np.unique(nodes_to_include))
        if self.split == 'train':
            mask_nodes = random.sample(self.train_nodes, self.mask_count)
            print('Mask nodes train: ', len(mask_nodes), self.mask_count)
        else:
            start_idx = idx * self.mask_count
            end_idx = min(start_idx + self.mask_count, len(self.target_nodes))
            mask_nodes = self.target_nodes[start_idx:end_idx]
            for n in mask_nodes:
                assert n not in nodes_to_include # ok
                nodes_to_include.append(n)
        
        nodes_set = set(nodes_to_include)
        mask = (self.data_df['node_id1'].isin(nodes_set) & self.data_df['node_id2'].isin(nodes_set))
        subgraph_edges = self.data_df[mask]

        print('Amount of isolated nodes: ', len(nodes_to_include) - len(np.unique(np.concatenate([subgraph_edges['node_id1'].to_numpy(), subgraph_edges['node_id2'].to_numpy()]))))

        if self.split == 'val':
            print('Amount of isolated val nodes: ', len(set(mask_nodes) - set(list(np.unique(np.concatenate([subgraph_edges['node_id1'].to_numpy(), subgraph_edges['node_id2'].to_numpy()])))) ))


        sorted_nodes = sorted(nodes_to_include)
        print('Sorted nodes: ', np.array(sorted_nodes))
        node_mapping = {node: i for i, node in enumerate(sorted_nodes)}

        src = [node_mapping[n] for n in subgraph_edges['node_id1'].values]
        dst = [node_mapping[n] for n in subgraph_edges['node_id2'].values]
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long) # ok

        print('Edge index: ', edge_index, edge_index.shape, subgraph_edges.shape)

        node_features = torch.tensor(self.node_classes[sorted_nodes], dtype=torch.float)

        print('Node features dataset: ', node_features)
        assert not torch.all(node_features == 1/4)

        mask_indices = [node_mapping[n] for n in mask_nodes if n in node_mapping]
        print('Len mask indices: ', len(mask_indices))
        predict_mask = torch.zeros(len(sorted_nodes), dtype=torch.bool)
        predict_mask[mask_indices] = True

        node_features_masked = node_features.clone()
        node_features_masked[predict_mask] = torch.ones(self.node_classes.shape[1]) / self.node_classes.shape[1]

        print('Node classes masked: ', node_features_masked, node_features_masked.shape)
        if not self.split == 'test':
            assert torch.all(node_features[predict_mask] != 1/4)
        else:
            assert torch.all(node_features[predict_mask] == 1/4)
        assert torch.all(node_features_masked[predict_mask] == 1/4)

        if self.split in ['train', 'val']:
            y = torch.tensor([np.argmax(self.node_classes[n]) for n in sorted_nodes], dtype=torch.long)
            print('Node classes: ', self.node_classes.shape)
            print('Y: ', y, y.shape, np.unique(y.numpy(), return_counts=True), np.unique(y[predict_mask].numpy(), return_counts=True), node_features_masked.shape)
        else:
            y = None

        
        ibd = subgraph_edges['ibd_sum'].values
        edge_weights = torch.tensor(list(ibd) + list(ibd), dtype=torch.float) # ok

        # masked_node_ids = torch.tensor([mask_nodes[mask_indices.index(i)] for i in mask_indices], dtype=torch.long)
        mask_nodes_filtered = [n for n in mask_nodes if n in node_mapping] # for "test" masked_node_ids
        assert set(mask_nodes) == set(mask_nodes_filtered)
        mask_nodes_in_graph_order = sorted(mask_nodes_filtered, key=lambda n: node_mapping[n]) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< just sorted()???
        assert np.all(np.array(sorted(mask_nodes_filtered, key=lambda n: node_mapping[n])) == np.array(sorted(mask_nodes_filtered)))
        print('mask_nodes_filtered', mask_nodes_filtered)
        print('mask_nodes_in_graph_order', mask_nodes_in_graph_order)

        masked_node_ids = torch.tensor(mask_nodes_in_graph_order, dtype=torch.long)

        final_graph = Data(
            x=node_features_masked,
            edge_index=edge_index,
            y=y,
            weight=edge_weights,
            num_classes=self.node_classes.shape[1],
            predict_mask=predict_mask,
            masked_node_ids=masked_node_ids
        )

        if self.split == 'val': # 'train'
            # CONTROL
            self.data_df["node_id1"] = self.data_df["node_id1"].astype(int)
            self.data_df["node_id2"] = self.data_df["node_id2"].astype(int)
            self.data_df["ibd_sum"] = self.data_df["ibd_sum"].astype(float)
            self.data_df["ibd_n"] = self.data_df["ibd_n"].astype(int)

            G = nx.from_pandas_edgelist(
                self.data_df,
                source="node_id1",
                target="node_id2",
                edge_attr=["ibd_sum", "ibd_n"],
                create_using=nx.Graph()
            )

            final_graph_nx = nx.Graph(to_networkx(final_graph))
            G_sub = G.subgraph(nodes_to_include)
            print('Checking isomorphism...')
            assert nx.vf2pp_is_isomorphic(final_graph_nx, G_sub, node_label=None)

        return final_graph


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

        x = self.first_linear(x)

        x_ = x.clone()
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = x_ + x
        x = self.n1(x)

        x_ = x.clone()
        x = F.elu(self.conv2(x, edge_index, edge_weight))
        x = x_ + x
        x = self.n2(x)

        x_ = x.clone()
        x = F.elu(self.conv3(x, edge_index, edge_weight))
        x = x_ + x

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

