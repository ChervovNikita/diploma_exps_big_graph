import random
import torch
from torch import nn
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.nn import TAGConv, GraphNorm, GCNConv, SGConv, SAGEConv, LEConv
from torch_geometric.utils import to_torch_csr_tensor
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm


## Datasets


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
        ibd_n = subgraph_edges['ibd_n'].values
        edge_weights = torch.tensor(list(ibd) + list(ibd), dtype=torch.float)
        edge_weights_n = torch.tensor(list(ibd_n) + list(ibd_n), dtype=torch.float)

        # masked_node_ids = torch.tensor([mask_nodes[mask_indices.index(i)] for i in mask_indices], dtype=torch.long)
        mask_nodes_filtered = [n for n in mask_nodes if n in node_mapping]
        mask_nodes_in_graph_order = sorted(mask_nodes_filtered, key=lambda n: node_mapping[n])
        masked_node_ids = torch.tensor(mask_nodes_in_graph_order, dtype=torch.long)

        return Data(
            x=node_features_masked,
            edge_index=edge_index,
            y=y,
            weight=edge_weights,
            weight_n=edge_weights_n,
            num_classes=self.node_classes.shape[1],
            predict_mask=predict_mask,
            masked_node_ids=masked_node_ids
        )


def compute_graph_based_features_local(
    edge_index: torch.Tensor,      # [2, E] local indices, directed or undirected
    edge_w: torch.Tensor,          # [E] W(i,j) = ibd_sum for each directed edge
    edge_k: torch.Tensor | None,   # [E] K_ij = ibd_n for each directed edge (or None -> ones)
    x_masked: torch.Tensor,        # [N, C] one-hot for labeled, uniform for unlabeled/masked
    num_classes: int,
    labeled_thr: float = 0.9999,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Implements GENLINK graph-based features (Eq. 4-7):
      [n_i,c, mean_w_i,c, std_w_i,c, max_w_i,c, IBD_i,c] for c=1..C  => shape [N, 5C]
    Unlabeled nodes never contribute to neighbors' features.
    """
    device = x_masked.device
    N = x_masked.size(0)
    C = num_classes

    # Determine which nodes are labeled (one-hot) vs unlabeled (uniform/masked)
    maxp, arg = x_masked.max(dim=1)
    lbl = torch.where(maxp > labeled_thr, arg, torch.full_like(arg, -1))  # -1 unlabeled

    src, dst = edge_index[0], edge_index[1]
    dst_lbl = lbl[dst]
    valid = dst_lbl >= 0

    src = src[valid]
    c = dst_lbl[valid]
    w = edge_w[valid].to(torch.float32)

    if edge_k is None:
        k = torch.ones_like(w)
    else:
        k = edge_k[valid].to(torch.float32)

    # Flatten (node, class) -> linear index
    lin = src * C + c  # [E_valid]

    flat_cnt  = torch.zeros(N * C, device=device, dtype=torch.float32)
    flat_sumw = torch.zeros_like(flat_cnt)
    flat_sumw2 = torch.zeros_like(flat_cnt)
    flat_sumk = torch.zeros_like(flat_cnt)

    ones = torch.ones_like(w)
    flat_cnt.index_add_(0, lin, ones)
    flat_sumw.index_add_(0, lin, w)
    flat_sumw2.index_add_(0, lin, w * w)
    flat_sumk.index_add_(0, lin, k)

    # max_w via scatter_reduce (torch>=2.0); fallback to a (slower) loop otherwise
    flat_maxw = torch.full((N * C,), float("-inf"), device=device, dtype=torch.float32)
    if hasattr(flat_maxw, "scatter_reduce_"):
        flat_maxw.scatter_reduce_(0, lin, w, reduce="amax", include_self=True)
        flat_maxw[flat_maxw == float("-inf")] = 0.0
    else:
        flat_maxw.fill_(0.0)
        for u in torch.unique(lin):
            m = (lin == u)
            flat_maxw[u] = torch.max(w[m])

    cnt  = flat_cnt.view(N, C)
    sumw = flat_sumw.view(N, C)
    sumw2 = flat_sumw2.view(N, C)
    maxw = flat_maxw.view(N, C)
    sumk = flat_sumk.view(N, C)

    mean = torch.zeros_like(sumw)
    std = torch.zeros_like(sumw)

    mask = cnt > 0
    mean[mask] = sumw[mask] / (cnt[mask] + eps)
    var = torch.zeros_like(sumw2)
    var[mask] = sumw2[mask] / (cnt[mask] + eps) - mean[mask] * mean[mask]
    var = torch.clamp(var, min=0.0)
    std[mask] = torch.sqrt(var[mask] + eps)

    # Order exactly as Eq. (7): counts, mean, std, max, IBD-sum(Kij)
    feats = torch.cat([cnt, mean, std, maxw, sumk], dim=1)  # [N, 5C]
    return feats

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

        ibd_sum = torch.tensor(subgraph_edges["ibd_sum"].values, dtype=torch.float32)
        ibd_n   = torch.tensor(subgraph_edges["ibd_n"].values,   dtype=torch.float32)

        node_features_masked = node_features.clone()
        node_features_masked[predict_mask] = torch.ones(self.node_classes.shape[1]) / self.node_classes.shape[1]

        edge_w = torch.cat([ibd_sum, ibd_sum], dim=0)
        edge_k = torch.cat([ibd_n,   ibd_n],   dim=0)
        C = self.node_classes.shape[1]

        # node_features_graph_based = compute_graph_based_features(list(range(len(sorted_nodes))), subgraph_edges, node_features_masked, self.node_classes.shape[1])
        node_features_graph_based = compute_graph_based_features_local(
            edge_index=edge_index,
            edge_w=edge_w,
            edge_k=edge_k,
            x_masked=node_features_masked,
            num_classes=C,
        )

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

class MaskedGraphDatasetGraphBasedOneNode(Dataset):
    def __init__(self, data_df, unknown_nodes_subset, train_nodes, val_nodes, test_nodes, split, node_classes):
        super().__init__()
        self.data_df = data_df
        self.unknown_nodes_subset = unknown_nodes_subset
        self.train_nodes = train_nodes
        self.val_nodes = val_nodes
        self.test_nodes = test_nodes
        self.split = split
        self.node_classes = node_classes

        if split == 'train':
            self.target_nodes = train_nodes
        elif split == 'val':
            self.target_nodes = val_nodes
        else:
            self.target_nodes = test_nodes

    def len(self):
        return len(self.target_nodes)
    def get(self, idx):
        nodes_to_include = self.train_nodes.copy()
        nodes_to_include.extend(self.unknown_nodes_subset)
        if self.split == 'train':
            mask_nodes = [self.train_nodes[idx]]
        else:
            mask_nodes = [self.target_nodes[idx]]
            assert mask_nodes[0] not in nodes_to_include
            nodes_to_include.append(mask_nodes[0])
        
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

        ibd_sum = torch.tensor(subgraph_edges["ibd_sum"].values, dtype=torch.float32)
        ibd_n   = torch.tensor(subgraph_edges["ibd_n"].values,   dtype=torch.float32)

        node_features_masked = node_features.clone()
        node_features_masked[predict_mask] = torch.ones(self.node_classes.shape[1]) / self.node_classes.shape[1]

        edge_w = torch.cat([ibd_sum, ibd_sum], dim=0)
        edge_k = torch.cat([ibd_n,   ibd_n],   dim=0)
        C = self.node_classes.shape[1]

        node_features_graph_based = compute_graph_based_features_local(
            edge_index=edge_index,
            edge_w=edge_w,
            edge_k=edge_k,
            x_masked=node_features_masked,
            num_classes=C,
        )

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


class MaskedGraphDatasetGraphBasedEgo(MaskedGraphDatasetGraphBased):
    def __init__(self, data_df, unknown_nodes_subset, train_nodes, val_nodes, test_nodes, split, mask_count, num_samples, node_classes):
        super().__init__(data_df, unknown_nodes_subset, train_nodes, val_nodes, test_nodes, split, mask_count, num_samples, node_classes)

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

        ibd_sum = torch.tensor(subgraph_edges["ibd_sum"].values, dtype=torch.float32)
        ibd_n   = torch.tensor(subgraph_edges["ibd_n"].values,   dtype=torch.float32)

        node_features_masked = node_features.clone()
        node_features_masked[predict_mask] = torch.ones(self.node_classes.shape[1]) / self.node_classes.shape[1]

        edge_w = torch.cat([ibd_sum, ibd_sum], dim=0)
        edge_k = torch.cat([ibd_n,   ibd_n],   dim=0)
        C = self.node_classes.shape[1]

        # node_features_graph_based = compute_graph_based_features(list(range(len(sorted_nodes))), subgraph_edges, node_features_masked, self.node_classes.shape[1])
        node_features_graph_based = compute_graph_based_features_local(
            edge_index=edge_index,
            edge_w=edge_w,
            edge_k=edge_k,
            x_masked=node_features_masked,
            num_classes=C,
        )

        node_features_is_target = torch.zeros((len(sorted_nodes), 1), dtype=torch.float)
        node_features_is_target[mask_indices] = 1.0

        node_features = torch.cat([
            node_features_graph_based,
            node_features_masked,
            node_features_is_target
        ], dim=1)

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

class TripletGraphDataset(Dataset):
    def __init__(self, data_df, unknown_nodes_subset, train_nodes, val_nodes, test_nodes,
                 train_nodes_by_class, split='train', num_mask_nodes=64, num_samples=500, 
                 mask_count=64, return_node_ids=False, node_classes=None):
        super().__init__()
        self.data_df = data_df
        self.unknown_nodes_subset = unknown_nodes_subset
        self.train_nodes = train_nodes
        self.val_nodes = val_nodes
        self.test_nodes = test_nodes
        self.train_nodes_by_class = train_nodes_by_class
        self.split = split
        self.num_mask_nodes = num_mask_nodes
        self.num_samples = num_samples
        self.mask_count = mask_count
        self.return_node_ids = return_node_ids
        self.node_classes = node_classes

        if split == 'train':
            self.target_nodes = train_nodes
        elif split == 'train_eval':
            self.target_nodes = train_nodes  # Same nodes but fixed iteration
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
            # Random sampling for training
            mask_nodes = []
            classes = [c for c in self.train_nodes_by_class.keys() if len(self.train_nodes_by_class[c]) >= 2]
            nodes_per_class = max(2, self.num_mask_nodes // len(classes))
            for c in classes:
                available = self.train_nodes_by_class[c]
                sample_size = min(nodes_per_class, len(available))
                mask_nodes.extend(random.sample(available, sample_size))
            random.shuffle(mask_nodes)
            mask_nodes = mask_nodes[:self.num_mask_nodes]
        else:
            start_idx = idx * self.mask_count
            end_idx = min(start_idx + self.mask_count, len(self.target_nodes))
            mask_nodes = self.target_nodes[start_idx:end_idx]
            for n in mask_nodes:
                if n not in nodes_to_include:
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

        if self.split in ['train', 'val', 'train_eval']:
            y = torch.tensor([np.argmax(self.node_classes[n]) for n in sorted_nodes], dtype=torch.long)
        else:
            y = None
        
        ibd = subgraph_edges['ibd_sum'].values
        ibd_n = subgraph_edges['ibd_n'].values
        edge_weights = torch.tensor(list(ibd) + list(ibd), dtype=torch.float)
        edge_weights_n = torch.tensor(list(ibd_n) + list(ibd_n), dtype=torch.float)

        mask_nodes_filtered = [n for n in mask_nodes if n in node_mapping]
        mask_nodes_in_graph_order = sorted(mask_nodes_filtered, key=lambda n: node_mapping[n])
        masked_node_ids = torch.tensor(mask_nodes_in_graph_order, dtype=torch.long)

        return Data(
            x=node_features_masked,
            edge_index=edge_index,
            y=y,
            weight=edge_weights,
            weight_n=edge_weights_n,
            num_classes=self.node_classes.shape[1],
            predict_mask=predict_mask,
            masked_node_ids=masked_node_ids
        )


## Models


class TAGConvModelBase(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=512):
        super().__init__()
        self.conv1 = TAGConv(num_features, hidden_dim)
        self.conv2 = TAGConv(hidden_dim, hidden_dim)
        self.conv3 = TAGConv(hidden_dim, num_features)
        self.n1 = GraphNorm(hidden_dim)
        self.n2 = GraphNorm(hidden_dim)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.weight

        num_nodes = x.size(0)
        adj = to_torch_csr_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes))

        x = F.elu(self.conv1(x, adj))
        x = self.n1(x)
        x = F.elu(self.conv2(x, adj))
        x = self.n2(x)
        x = F.elu(self.conv3(x, adj))

        return x


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


class ArbitaryModel(torch.nn.Module):
    def __init__(self, num_features, num_classes, conv_layer_type, hidden_dim=512):
        super().__init__()
        self.first_linear = torch.nn.Linear(num_features, hidden_dim)
        self.conv1 = conv_layer_type(hidden_dim, hidden_dim)
        self.conv2 = conv_layer_type(hidden_dim, hidden_dim)
        self.conv3 = conv_layer_type(hidden_dim, hidden_dim)
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


class BEBlock(torch.nn.Module):
    def __init__(self, in_shape, out_shape, is_first, device, tabm_inits):
        super().__init__()

        self.R = nn.Parameter(torch.empty(tabm_inits, in_shape, device=device))
        self.S = nn.Parameter(torch.empty(tabm_inits, out_shape, device=device))
        self.B = nn.Parameter(torch.empty(tabm_inits, out_shape, device=device))
        self.W = nn.Linear(in_shape, out_shape, bias=False)

        self.tabm_inits = tabm_inits
        self.is_first = is_first
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        if self.is_first:
            self.R.bernoulli_(0.5).mul_(2).sub_(1)
        else:
            self.R.fill_(1.0)
        self.S.fill_(1.0)
        self.B.zero_()

    def forward(self, x, tabm_seed):
        x = x * self.R[tabm_seed]
        x = self.W(x)
        x = x * self.S[tabm_seed]
        x = x + self.B[tabm_seed]
        return x


class TAGConvModelTABM(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=512, tabm_inits=None, device=None):
        super().__init__()
        self.tabm_inits = tabm_inits

        bes = [
            BEBlock(num_features, num_features, is_first=True, device=device, tabm_inits=tabm_inits),
            BEBlock(num_features, hidden_dim, is_first=False, device=device, tabm_inits=tabm_inits)
        ]
        self.lrs = torch.nn.ModuleList(bes)

        self.conv1 = TAGConv(hidden_dim, hidden_dim)
        self.conv2 = TAGConv(hidden_dim, hidden_dim)
        self.conv3 = TAGConv(hidden_dim, hidden_dim)
        self.n1 = GraphNorm(hidden_dim)
        self.n2 = GraphNorm(hidden_dim)
        self.linear = BEBlock(hidden_dim, num_classes, is_first=False, device=device, tabm_inits=tabm_inits)

    def forward(self, data, tabm_seed):
        x, edge_index, edge_weight = data.x, data.edge_index, data.weight
        for be in self.lrs:
            x = be(x, tabm_seed)
            x = F.elu(x)

        num_nodes = x.size(0)
        adj = to_torch_csr_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes))
        
        x_ = x
        x = F.elu(self.conv1(x, adj))
        x = x_ + x
        x = self.n1(x)
        
        x_ = x
        x = F.elu(self.conv2(x, adj))
        x = x_ + x
        x = self.n2(x)
        
        x_ = x
        x = F.elu(self.conv3(x, adj))
        x = x_ + x

        return self.linear(x, tabm_seed)


class WeightProcessor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(5 + 1, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid(),
        )
    
    def forward(self, edge_weight, edge_weight_n):
        edge_weight_n = torch.minimum(edge_weight_n, torch.tensor(4.0))
        edge_weight_n_one_hot = F.one_hot(edge_weight_n.long(), num_classes=5)
        feats = torch.cat([torch.log1p(edge_weight)[:, None], edge_weight_n_one_hot], dim=-1)
        return self.mlp(feats)


class TAGConvModelTrainableWeights(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=512):
        super().__init__()
        self.first_linear = torch.nn.Linear(num_features, hidden_dim)
        self.conv1 = TAGConv(hidden_dim, hidden_dim)
        self.conv2 = TAGConv(hidden_dim, hidden_dim)
        self.conv3 = TAGConv(hidden_dim, hidden_dim)
        self.n1 = GraphNorm(hidden_dim)
        self.n2 = GraphNorm(hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, num_classes)
        self.weight_processor = WeightProcessor()

    def forward(self, data):
        x, edge_index, edge_weight, edge_weight_n = data.x, data.edge_index, data.weight, data.weight_n

        edge_weight = self.weight_processor(edge_weight, edge_weight_n).squeeze(-1)

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


class TAGConvModelEmbeddings(torch.nn.Module):
    def __init__(self, num_classes, embed_dim=512):
        super().__init__()
        total_features = num_classes
        self.first_linear = torch.nn.Linear(total_features, embed_dim)
        self.conv1 = TAGConv(embed_dim, embed_dim)
        self.conv2 = TAGConv(embed_dim, embed_dim)
        self.conv3 = TAGConv(embed_dim, embed_dim)
        self.n1 = GraphNorm(embed_dim)
        self.n2 = GraphNorm(embed_dim)

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

        x = F.normalize(x, p=2, dim=-1)
        return x


class MoEConvModel(torch.nn.Module):
    def __init__(self, num_features, num_classes, top_k, hidden_dim=512):
        super().__init__()
        self.top_k = top_k
        
        self.first_linear = torch.nn.Linear(num_features, hidden_dim)

        experts_list = [TAGConv, GCNConv, TAGConv, GCNConv]
        
        self.experts1 = torch.nn.ModuleList([experts_type(hidden_dim, hidden_dim) for experts_type in experts_list])
        self.experts2 = torch.nn.ModuleList([experts_type(hidden_dim, hidden_dim) for experts_type in experts_list])
        self.experts3 = torch.nn.ModuleList([experts_type(hidden_dim, hidden_dim) for experts_type in experts_list])
        
        self.gate1 = torch.nn.Linear(hidden_dim, len(experts_list))
        self.gate2 = torch.nn.Linear(hidden_dim, len(experts_list))
        self.gate3 = torch.nn.Linear(hidden_dim, len(experts_list))
        
        self.n1 = GraphNorm(hidden_dim)
        self.n2 = GraphNorm(hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, num_classes)

    def _moe_forward(self, x, adj, experts, gate):
        gate_logits = gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)

        topk_weights, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        expert_outputs = []
        for expert in experts:
            expert_outputs.append(expert(x, adj))
        expert_outputs = torch.stack(expert_outputs, dim=1)

        batch_indices = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand(-1, self.top_k)
        selected_outputs = expert_outputs[batch_indices, topk_indices]

        out = (selected_outputs * topk_weights.unsqueeze(-1)).sum(dim=1)
        return out

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.weight

        num_nodes = x.size(0)
        adj = to_torch_csr_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes))

        x = self.first_linear(x)

        x_ = x.clone()
        x = F.elu(self._moe_forward(x, adj, self.experts1, self.gate1))
        x = x_ + x
        x = self.n1(x)
        del x_

        x_ = x.clone()
        x = F.elu(self._moe_forward(x, adj, self.experts2, self.gate2))
        x = x_ + x
        x = self.n2(x)
        del x_

        x_ = x.clone()
        x = F.elu(self._moe_forward(x, adj, self.experts3, self.gate3))
        x = x_ + x
        del x_, adj

        return self.linear(x)


## Losses


class SemiHardTripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
        
        labels = labels.view(-1, 1)
        same_label = (labels == labels.T).float()
        eye = torch.eye(embeddings.size(0), device=embeddings.device)
        valid_positive_mask = same_label * (1 - eye)
        valid_negative_mask = 1 - same_label
        
        anchor_positive_dist = pairwise_dist * valid_positive_mask
        pos_mask_sum = valid_positive_mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_positive_dist = anchor_positive_dist.sum(dim=1, keepdim=True) / pos_mask_sum
        
        neg_dists = pairwise_dist
        semi_hard_mask = valid_negative_mask * (neg_dists > mean_positive_dist) * (neg_dists < mean_positive_dist + self.margin)
        has_semi_hard = semi_hard_mask.sum(dim=1) > 0
        
        semi_hard_dist = pairwise_dist.clone()
        semi_hard_dist[semi_hard_mask == 0] = float('inf')
        min_semi_hard, _ = semi_hard_dist.min(dim=1)
        
        hard_neg_dist = pairwise_dist + pairwise_dist.max() * same_label
        hardest_neg, _ = hard_neg_dist.min(dim=1)
        
        negative_dist = torch.where(has_semi_hard, min_semi_hard, hardest_neg)
        positive_dist = mean_positive_dist.squeeze()
        
        valid_anchors = (valid_positive_mask.sum(dim=1) > 0)
        loss = F.relu(positive_dist - negative_dist + self.margin)
        loss = loss[valid_anchors]
        
        return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=embeddings.device)


## Utils


class SingleDeviceWrapper(torch.nn.Module):  # so one and multi gpu have the same interface
    def __init__(self, module, device):
        super().__init__()
        self.module = module
        self.device = device

    def forward(self, data_list, **kwargs):
        if isinstance(data_list, list):
            batch = Batch.from_data_list(data_list).to(self.device, non_blocking=True)
        else:
            batch = data_list.to(self.device, non_blocking=True)
        return self.module(batch, **kwargs), batch


def compute_triplet_embeddings(model, loader):
    model.eval()
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Computing embeddings', leave=False):
            embeddings, batch_data = model(batch)
            predict_mask = batch_data.predict_mask

            emb = embeddings[predict_mask].cpu()
            lbl = batch_data.y[predict_mask].cpu()
            all_embeddings.append(emb)
            all_labels.append(lbl)
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def knn_predict(train_emb, train_y, query_emb, k=10, weighted=False):
    sims = torch.matmul(query_emb, train_emb.T)
    topk_vals, topk_indices = torch.topk(sims, k=min(k, train_emb.size(0)), dim=1)
    preds = []
    for i, (neighbors, sim_values) in enumerate(zip(topk_indices, topk_vals)):
        neighbor_labels = train_y[neighbors].numpy()
        if weighted:
            weights = (sim_values + 1) / 2
            label_weights = defaultdict(float)
            for label, weight in zip(neighbor_labels, weights.cpu().numpy()):
                label_weights[int(label)] += weight
            preds.append(max(label_weights.items(), key=lambda x: x[1])[0])
        else:
            classes, counts = np.unique(neighbor_labels, return_counts=True)
            preds.append(int(classes[np.argmax(counts)]))
    
    return preds