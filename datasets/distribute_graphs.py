import torch
import torch_geometric
import torch_geometric.typing
import numpy as np

from torch_geometric.data import Data
from torch_geometric.typing import pyg_lib
from torch_geometric.utils import sort_edge_index
from torch_geometric.nn.pool import radius_graph
from torch_geometric.utils.sparse import index2ptr, ptr2index

from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering


def split_large_graph_random(pos, x, target, vel, attr, radius, world_size, device, special_nodes=None, special_edges=None,):
    assert device == 'cpu'

    if special_nodes is None:
        special_nodes = torch.ones((pos.size(0), ), dtype=torch.bool, device=device)

    pos, x, target, vel, attr = \
        pos.to(device), x.to(device), target.to(device), vel.to(device), attr.to(device)

    node_cnt = pos.size(0)
    indices = torch.randperm(node_cnt, device=device)
    chunk_sizes = [node_cnt // world_size for _ in range(world_size - 1)]
    chunk_sizes.append(node_cnt - sum(chunk_sizes))
    indices_chunk = torch.split(indices, chunk_sizes)
    
    loc_mean = torch.mean(pos, dim=0).unsqueeze(0)  # [1, 3]
    
    data = []
    for i in range(world_size):
        x_i = x[indices_chunk[i]]
        pos_i = pos[indices_chunk[i]]
        vel_i = vel[indices_chunk[i]]
        attr_i = attr[indices_chunk[i]]
        target_i = target[indices_chunk[i]]
        special_nodes_i = special_nodes[indices_chunk[i]]

        edge_index_i = radius_graph(pos_i, r=radius, num_workers=1, max_num_neighbors=pos_i.size(0))
        edge_attr_i = torch.sqrt(torch.sum((pos_i[edge_index_i[0]] - pos_i[edge_index_i[1]]) ** 2, dim=-1)).unsqueeze(-1).repeat(1, 2)

        data.append(Data(
            x=x_i, pos=pos_i, vel=vel_i, attr=attr_i, target=target_i, loc_mean=loc_mean,
            edge_index=edge_index_i, edge_attr=edge_attr_i, special_nodes=special_nodes_i,
        ).to('cpu'))

    return data


def split_large_graph_metis(pos, x, target, vel, attr, outer_radius, inner_radius, world_size, device, special_nodes=None, special_edges=None,):
    assert device == 'cpu'

    if special_nodes is None:
        special_nodes = torch.ones((pos.size(0), ), dtype=torch.bool, device=device)

    pos, x, target, vel, attr = \
        pos.to(device), x.to(device), target.to(device), vel.to(device), attr.to(device)
    
    # get radius graph in CPU because of extremely large graph
    node_cnt = pos.size(0)
    edge_index = radius_graph(pos, r=outer_radius, num_workers=1, max_num_neighbors=node_cnt)  # [2, num_edges]
    cluster = metis(edge_index, node_cnt, world_size, recursive=True) 

    loc_mean = torch.mean(pos, dim=0).unsqueeze(0)  # [1, 3]

    data = []
    for i in range(world_size):
        x_i = x[cluster == i]
        pos_i = pos[cluster == i]
        vel_i = vel[cluster == i]
        attr_i = attr[cluster == i]
        target_i = target[cluster == i]
        special_nodes_i = special_nodes[cluster == i]

        edge_index_i = radius_graph(pos_i, r=inner_radius, num_workers=1, max_num_neighbors=pos_i.size(0))
        edge_attr_i = torch.sqrt(torch.sum((pos_i[edge_index_i[0]] - pos_i[edge_index_i[1]]) ** 2, dim=-1)).unsqueeze(-1).repeat(1, 2)

        data.append(Data(
            x=x_i, pos=pos_i, vel=vel_i, attr=attr_i, target=target_i, loc_mean=loc_mean,
            edge_index=edge_index_i, edge_attr=edge_attr_i, special_nodes=special_nodes_i,
        ).to('cpu'))

    return data


def split_large_graph_spectral(pos, x, target, vel, attr, outer_radius, inner_radius, world_size, device):
    assert device == 'cpu'
    pos, x, target, vel, attr = \
        pos.to(device), x.to(device), target.to(device), vel.to(device), attr.to(device)
    
    cluster = spectral_clustering(pos, world_size, random_state=0)

    loc_mean = torch.mean(pos, dim=0).unsqueeze(0)  # [1, 3]

    data = []
    for i in range(world_size):
        x_i = x[cluster == i]
        pos_i = pos[cluster == i]
        vel_i = vel[cluster == i]
        attr_i = attr[cluster == i]
        target_i = target[cluster == i]

        edge_index_i = radius_graph(pos_i, r=inner_radius, num_workers=1, max_num_neighbors=pos_i.size(0))
        edge_attr_i = torch.sqrt(torch.sum((pos_i[edge_index_i[0]] - pos_i[edge_index_i[1]]) ** 2, dim=-1)).unsqueeze(-1).repeat(1, 2)

        data.append(Data(
            x=x_i, pos=pos_i, vel=vel_i, attr=attr_i, target=target_i, loc_mean=loc_mean,
            edge_index=edge_index_i, edge_attr=edge_attr_i,
        ).to('cpu'))
    
    return data


def split_large_graph_kmeans(pos, x, target, vel, attr, outer_radius, inner_radius, world_size, device):
    assert device == 'cpu'
    pos, x, target, vel, attr = \
        pos.to(device), x.to(device), target.to(device), vel.to(device), attr.to(device)
    
    cluster = kmeans_clustering(pos, world_size)

    loc_mean = torch.mean(pos, dim=0).unsqueeze(0)  # [1, 3]

    data = []
    for i in range(world_size):
        x_i = x[cluster == i]
        pos_i = pos[cluster == i]
        vel_i = vel[cluster == i]
        attr_i = attr[cluster == i]
        target_i = target[cluster == i]

        edge_index_i = radius_graph(pos_i, r=inner_radius, num_workers=1, max_num_neighbors=pos_i.size(0))
        edge_attr_i = torch.sqrt(torch.sum((pos_i[edge_index_i[0]] - pos_i[edge_index_i[1]]) ** 2, dim=-1)).unsqueeze(-1).repeat(1, 2)

        data.append(Data(
            x=x_i, pos=pos_i, vel=vel_i, attr=attr_i, target=target_i, loc_mean=loc_mean,
            edge_index=edge_index_i, edge_attr=edge_attr_i,
        ).to('cpu'))
    
    return data



"""
    From ClusterData: https://pytorch-geometric.readthedocs.io/en/2.4.0/_modules/torch_geometric/loader/cluster.html#ClusterData
    Computes a node-level partition assignment vector via METIS.
"""
def metis(edge_index, num_nodes: int, num_parts: int, recursive):
    if num_parts <= 1:
        return torch.zeros(num_nodes, dtype=torch.long).to(edge_index.device)

    # Calculate CSR representation:
    row, col = sort_edge_index(edge_index, num_nodes=num_nodes)
    rowptr = index2ptr(row, size=num_nodes)
    
    # Compute METIS partitioning:
    cluster = None

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        try:
            cluster = torch.ops.torch_sparse.partition(
                rowptr.cpu(),
                col.cpu(),
                None,
                num_parts,
                recursive,
            ).to(edge_index.device)
        except (AttributeError, RuntimeError):
            pass

    if cluster is None and torch_geometric.typing.WITH_METIS:
        cluster = pyg_lib.partition.metis(
            rowptr.cpu(),
            col.cpu(),
            num_parts,
            recursive=recursive,
        ).to(edge_index.device)

    if cluster is None:
        raise ImportError(f"requires either 'pyg-lib' or 'torch-sparse'")

    return cluster


def kmeans_clustering(pos, num_parts):
    X = pos.detach().cpu().numpy().astype(np.float32)

    kmeans = KMeans(
        n_clusters=num_parts,
        random_state=0,
        n_init="auto",
    )
    labels = kmeans.fit_predict(X)

    return torch.from_numpy(labels).to(torch.long)


def spectral_clustering(pos: torch.Tensor, num_parts: int,
                        sigma: float = None, random_state: int = 0):
    X = pos.detach().cpu().numpy().astype(np.float32)
    N = X.shape[0]

    if sigma is None:
        m = min(N, 2000)
        idx = np.random.RandomState(0).choice(N, size=m, replace=False)
        D = np.linalg.norm(X[idx, None, :] - X[None, idx, :], axis=2)
        sigma = np.median(D[D > 0]) + 1e-12

    gamma = 1.0 / (2.0 * (sigma ** 2))

    sc = SpectralClustering(
        n_clusters=num_parts,
        affinity='rbf',
        gamma=gamma,
        assign_labels='kmeans',
        random_state=random_state,
        eigen_solver='arpack'
    )
    labels = sc.fit_predict(X)
    return torch.from_numpy(labels).to(torch.long)
