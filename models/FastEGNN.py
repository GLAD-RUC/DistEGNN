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
    

class E_GCL_vel(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    """

    def __init__(self, node_feat_nf, node_feat_out_nf, node_attr_nf, edge_attr_nf, hidden_nf, virtual_channels, world_size,
                 act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False, gravity=None):
        super(E_GCL_vel, self).__init__()
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.hiddden_nf = hidden_nf
        self.node_feat_out_nf = node_feat_out_nf
        self.tanh = tanh
        self.world_size = world_size
        self.epsilon = 1e-8
        edge_coords_nf = 1

        # For Fast-EGNN
        self.virtual_channels = virtual_channels

        ## MLPS
        self.edge_mlp = nn.Sequential(  # \phi_{e}
            nn.Linear(2 * node_feat_nf + edge_coords_nf + edge_attr_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.edge_mlp_virtual = nn.Sequential(  # \phi_{ev}
            nn.Linear(2 * node_feat_nf + edge_coords_nf + virtual_channels, hidden_nf),  # No edge_feat
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )


        if attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid(),
            )

            self.att_mlp_virtual = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid(),
            )
            

        def get_coord_mlp():
            layer = nn.Linear(hidden_nf, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

            coord_mlp = []
            coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
            coord_mlp.append(act_fn)
            coord_mlp.append(layer)
            if self.tanh:
                coord_mlp.append(nn.Tanh())
            
            return coord_mlp
        

        self.coord_mlp_r = nn.Sequential(*get_coord_mlp())  # \phi_{x}
        self.coord_mlp_r_virtual = nn.Sequential(*get_coord_mlp())  # \phi_{xv}
        self.coord_mlp_v_virtual = nn.Sequential(*get_coord_mlp())  # \phi_{X}

        # Velocity
        self.coord_mlp_vel = nn.Sequential(  # \phi_{v}
            nn.Linear(node_feat_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1)
        )

        # Gravity
        self.gravity = gravity
        if self.gravity is not None:
            self.gravity_mlp = nn.Sequential(
                nn.Linear(node_feat_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, 1)
            )

        self.node_mlp = nn.Sequential(  # \phi_{h}
            # nn.Linear(hidden_nf + hidden_nf + virtual_channels * hidden_nf + node_attr_nf, hidden_nf),
            nn.Linear(hidden_nf + hidden_nf + hidden_nf + node_attr_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, node_feat_out_nf)
        )

        self.node_mlp_virtual = nn.Sequential(  # \phi_{hv}
            nn.Linear(hidden_nf + hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, node_feat_out_nf)
        )


    def edge_model(self, source, target, radial, edge_attr):
        out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out
    

    # [batch_node, H, 1]  [batch_node, H, C], [batch_node, 1, C], [batch_node, C, C] -> [batch_node, H, C]
    def edge_mode_virtual(self, feat_R, feat_V, radial, mix_V):
        feat_R = feat_R.repeat(1, 1, self.virtual_channels)  # [batch_size, H, C]
        out = torch.cat([feat_R, feat_V, radial, mix_V], dim=1)

        out = self.edge_mlp_virtual(out.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_node, H, C]

        if self.attention:
            att_val = self.att_mlp_virtual(out.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_node, 1, C]
            out = out * att_val  # [batch_node, H, C]
        return out


    def coord_model_vel(self, node_feat, coord, node_vel, edge_index, coord_diff, edge_feat, virtual_edge_feat, virtual_coord_diff):
        row, col = edge_index

        trans = coord_diff * self.coord_mlp_r(edge_feat)  # coord_mlp_r: [batch_edge, H] -> [batch_edge, 1]
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        
        coord = coord + agg

        # virtual_edge_feat: [batch_node, H, C], virtual_coord_diff: [batch_node, 3, C]
        trans_v = torch.mean(-virtual_coord_diff * self.coord_mlp_r_virtual(virtual_edge_feat.permute(0, 2, 1)).permute(0, 2, 1), dim=-1)  # [batch_node, 3]
        coord = coord + trans_v

        coord = coord + self.coord_mlp_vel(node_feat) * node_vel  # Velocity

        if self.gravity is not None:
            coord = coord + self.gravity_mlp(node_feat) * self.gravity.to(node_feat.device)  # Gravity

        return coord
    

    def coord_model_virtual(self, virtual_coord, virtual_edge_feat, virtual_coord_diff, data_batch):
        trans = virtual_coord_diff * self.coord_mlp_v_virtual(virtual_edge_feat.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_node, 3, C]
        agg = global_mean_pool(trans.reshape(trans.size(0), -1), data_batch).reshape(-1, 3, self.virtual_channels)  # [B, 3, C]

        if self.world_size > 1:
            batch_total = torch.tensor([torch.sum(data_batch == i).item() for i in range(data_batch.max().item() + 1)]).cuda().unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            agg = weighted_average_reduce(agg, batch_total)

        virtual_coord = virtual_coord + agg
        return virtual_coord
    

    def node_model(self, node_feat, edge_index, edge_feat, virtual_edge_feat, node_attr):
        # node_feat: [batch_node, H], edge_feat: [batch_edge, H], virtual_edge_feat: [batch_node, H, C]
        row, col = edge_index
        agg = unsorted_segment_mean(edge_feat, row, num_segments=node_feat.size(0))  # [batch_node, H]
        agg_v = virtual_edge_feat.mean(dim=-1)  # [batch_node, H]
        # virtual_edge_feat = virtual_edge_feat.reshape(virtual_edge_feat.size(0), -1)
        if node_attr is not None:
            agg = torch.cat([node_feat, agg, agg_v, node_attr], dim=1)
        else:
            agg = torch.cat([node_feat, agg, agg_v], dim=1)
        out = self.node_mlp(agg)

        if self.residual:
            out = node_feat + out
        return out


    def node_model_virtual(self, virtual_node_feat, virtual_edge_feat, data_batch):
        # virtual_node_feat: [B, H, C], virtual_edge_feat: [batch_node, H, C]
        agg = global_mean_pool(virtual_edge_feat.reshape(virtual_edge_feat.size(0), -1), data_batch) \
              .reshape(-1, self.hiddden_nf, self.virtual_channels)  # [B, H, C]
        
        if self.world_size > 1:
            batch_total = torch.tensor([torch.sum(data_batch == i).item() for i in range(data_batch.max().item() + 1)]).cuda().unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            agg = weighted_average_reduce(agg, batch_total)

        out = torch.cat([virtual_node_feat, agg], dim=1)
        out = self.node_mlp_virtual(out.permute(0, 2, 1)).permute(0, 2, 1)
        
        if self.residual:
            out = virtual_node_feat + out
        return out


    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff


    def forward(self, node_feat, edge_index, coord, node_vel, virtual_coord, virtual_node_feat, data_batch, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        virtual_coord_diff = virtual_coord[data_batch] - coord.unsqueeze(-1)  # [batch_node, 3, C]  (X - x)
        vitrual_radial = torch.norm(virtual_coord_diff, p=2, dim=1, keepdim=True)  # [batch_node, 1, C]

        # Edge Model
        edge_feat = self.edge_model(node_feat[row], node_feat[col], radial, edge_attr)  # [batch_edge, H]
        
        coord_mean = global_mean_pool(coord, data_batch)  # [B, 3]
        if self.world_size > 1:
            batch_total = torch.tensor([torch.sum(data_batch == i).item() for i in range(data_batch.max().item() + 1)]).cuda().unsqueeze(-1)  # [B, 1]
            coord_mean = weighted_average_reduce(coord_mean, batch_total)

        m_X = virtual_coord - coord_mean.unsqueeze(-1)  # [B, 3, C]
        m_X = torch.einsum('bij, bjk -> bik', m_X.permute(0, 2, 1), m_X)  # [B, C, C]
        # [batch_node, H, 1]  [batch_node, H, C], [batch_node, 1, C], [batch_node, C, C] -> [batch_node, H, C]
        virtual_edge_feat = self.edge_mode_virtual(node_feat.unsqueeze(-1), virtual_node_feat[data_batch], vitrual_radial, m_X[data_batch])  # [batch_edge, H, C]  # C times memory consumption

        # Coord Model
        coord = self.coord_model_vel(node_feat, coord, node_vel, edge_index, coord_diff, edge_feat, virtual_edge_feat, virtual_coord_diff)
        virtual_coord = self.coord_model_virtual(virtual_coord, virtual_edge_feat, virtual_coord_diff, data_batch)

        # Node Model
        node_feat = self.node_model(node_feat, edge_index, edge_feat, virtual_edge_feat, node_attr)
        virtual_node_feat = self.node_model_virtual(virtual_node_feat, virtual_edge_feat, data_batch)

        return node_feat, coord, virtual_node_feat, virtual_coord


class FastEGNN(nn.Module):
    def __init__(self, node_feat_nf, node_attr_nf, edge_attr_nf, hidden_nf, virtual_channels, world_size, act_fn=nn.SiLU(), 
                 n_layers=4, residual=True, attention=False, normalize=False, tanh=False, gravity=None):
        super(FastEGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.node_attr_nf = node_attr_nf
        self.virtual_channels = virtual_channels
        assert virtual_channels > 0, f'Channels of virtual node must greater than 0 (got {virtual_channels})'
        self.virtual_node_feat = nn.Parameter(data=torch.randn(size=(1, hidden_nf, virtual_channels)), requires_grad=True)
        self.embedding_in = nn.Linear(node_feat_nf, self.hidden_nf)
        if gravity is not None:
            gravity = torch.tensor(gravity)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_vel(hidden_nf, hidden_nf, node_attr_nf, edge_attr_nf, hidden_nf, virtual_channels=virtual_channels, world_size=world_size,
                                                    act_fn=act_fn, residual=residual, attention=attention, normalize=normalize, tanh=tanh, gravity=gravity))

    def forward(self, node_feat, node_loc, node_vel, loc_mean, edge_index, data_batch, edge_attr=None, node_attr=None):
        # init virtual node feat with multi-channels
        batch_size = data_batch[-1].item() + 1
        virtual_node_feat = self.virtual_node_feat.repeat(batch_size, 1, 1)
        virtual_node_loc = loc_mean.unsqueeze(-1).repeat(1, 1, self.virtual_channels)  # [B, 3] -> [B, 3, C]

        node_feat = self.embedding_in(node_feat)
        for i in range(0, self.n_layers):
            node_feat, node_loc, virtual_node_feat, virtual_node_loc = \
                  self._modules["gcl_%d" % i](node_feat, edge_index, node_loc, node_vel, virtual_node_loc, virtual_node_feat,
                                                data_batch, edge_attr=edge_attr, node_attr=node_attr)
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