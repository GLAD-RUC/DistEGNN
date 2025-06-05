import torch
import torch_geometric
from torch import nn
from torch import distributed as dist
from torch_geometric.nn import global_mean_pool
from torch.autograd import Function
from torch.distributed import group, ReduceOp


class _AllReduce(Function):
    @staticmethod
    def forward(ctx, op, group, tensor):
        ctx.group = group
        ctx.op = op
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None) + (_AllReduce.apply(ctx.op, ctx.group, grad_output.contiguous()),)
    

def all_reduce(tensor, op=ReduceOp.SUM, group=group.WORLD):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.

    After the call the returned tensor is going to be bitwise
    identical in all processes.

    Arguments:
        tensor (Tensor): Input of the collective.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective

    """
    return _AllReduce.apply(op, group, tensor)
    


class GCL_RF_vel(nn.Module):
    """
    Radial Field Convolutional Layer
    """

    def __init__(self, edge_attr_nf, hidden_nf, virtual_channels, world_size, act_fn=nn.LeakyReLU(0.2),):
        super(GCL_RF_vel, self).__init__()
        self.hiddden_nf = hidden_nf
        self.world_size = world_size
        self.epsilon = 1e-8

        # For Fast-RF
        self.virtual_channels = virtual_channels

        ## MLPS
        layer = nn.Linear(hidden_nf, hidden_nf, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi = nn.Sequential(nn.Linear(1 + edge_attr_nf, hidden_nf),
                                 act_fn,
                                 layer,
                                 nn.Tanh())
    
        layer = nn.Linear(hidden_nf, hidden_nf, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi_v = nn.Sequential(nn.Linear(1 + virtual_channels, hidden_nf),
                                   act_fn,
                                   layer,
                                   nn.Tanh())   


        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1),
        )

        self.edge_mlp_rv = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1),
        )

        self.edge_mlp_vr = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1),
        )



        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(1, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1)
        )



    def edge_model(self, radial, edge_attr):
        out = torch.cat([radial, edge_attr], dim=1)
        out = self.phi(out)  # [batch_edge, H]
        return out


    # [batch_edge, 1, C], [batch_edge, C, C] -> [batch_edge, H, C]
    def edge_mode_virtual(self, vitrual_radial, mix_V):
        out = torch.cat([vitrual_radial, mix_V], dim=1)
        out = self.phi_v(out.permute(0, 2, 1)).permute(0, 2, 1)
        return out
    

    def node_model(self, coord, node_vel, edge_index, coord_diff, edge_feat, virtual_edge_feat, virtual_coord_diff):
        row, col = edge_index

        # Messages from real nodes
        trans = coord_diff * self.edge_mlp(edge_feat)
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord = coord + agg

        # Messages from virtual nodes
        trans_v = torch.mean(-virtual_coord_diff * self.edge_mlp_rv(virtual_edge_feat.permute(0, 2, 1)).permute(0, 2, 1), dim=-1)
        coord = coord + trans_v

        # Velocity
        coord = coord + node_vel * self.coord_mlp_vel(torch.norm(node_vel, keepdim=True, dim=-1))
        return coord
    

    def node_model_virtual(self, virtual_coord, virtual_edge_feat, virtual_coord_diff, data_batch):
        # Message from real nodes
        trans = virtual_coord_diff * self.edge_mlp_vr(virtual_edge_feat.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_node, 3, C]
        agg = global_mean_pool(trans.reshape(trans.size(0), -1), data_batch).reshape(-1, 3, self.virtual_channels)  # [B, 3, C]

        if self.world_size > 1:
            batch_total = torch.tensor([torch.sum(data_batch == i).item() for i in range(data_batch.max().item() + 1)]).cuda().unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            agg = weighted_average_reduce(agg, batch_total)

        virtual_coord = virtual_coord + agg
        return virtual_coord


    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        return radial, coord_diff


    def forward(self, edge_index, coord, node_vel, virtual_coord, data_batch, edge_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        virtual_coord_diff = virtual_coord[data_batch] - coord.unsqueeze(-1)  # [batch_node, 3, C]  (X - x)
        vitrual_radial = torch.norm(virtual_coord_diff, p=2, dim=1, keepdim=True)  # [batch_node, 1, C]

        # Edge Model
        edge_feat = self.edge_model(radial, edge_attr)  # [batch_edge, 1]
        
        coord_mean = global_mean_pool(coord, data_batch)  # [B, 3]
        m_X = virtual_coord - coord_mean.unsqueeze(-1)  # [B, 3, C]
        m_X = torch.einsum('bij, bjk -> bik', m_X.permute(0, 2, 1), m_X)  # [B, C, C]
        virtual_edge_feat = self.edge_mode_virtual(vitrual_radial, m_X[data_batch])

        coord = self.node_model(coord, node_vel, edge_index, coord_diff, edge_feat, virtual_edge_feat, virtual_coord_diff)
        virtual_coord = self.node_model_virtual(virtual_coord, virtual_edge_feat, virtual_coord_diff, data_batch)

        return coord, virtual_coord


class FastRF(nn.Module):
    def __init__(self, edge_attr_nf, hidden_nf, virtual_channels, world_size, act_fn=nn.SiLU(), n_layers=4):
        super(FastRF, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.virtual_channels = virtual_channels
        assert virtual_channels > 0, f'Channels of virtual node must greater than 0 (got {virtual_channels})'
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL_RF_vel(edge_attr_nf, hidden_nf, virtual_channels, world_size))


    def forward(self, node_loc, node_vel, loc_mean, edge_index, data_batch, edge_attr=None,):
        # init virtual node feat with multi-channels
        virtual_node_loc = loc_mean.unsqueeze(-1).repeat(1, 1, self.virtual_channels)  # [B, 3] -> [B, 3, C]

        for i in range(0, self.n_layers):
            node_loc, virtual_node_loc = self._modules["gcl_%d" % i](edge_index, node_loc, node_vel, virtual_node_loc, data_batch, edge_attr)
        return node_loc, virtual_node_loc


def weighted_average_reduce(data, weight):
    data = data.clone()
    weight = weight.clone()

    data.mul_(weight)
    data = all_reduce(data, op=ReduceOp.SUM)
    weight = all_reduce(weight, op=ReduceOp.SUM)
    data.div_(weight)

    return data


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)