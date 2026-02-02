import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from einops import rearrange, repeat

from common.data_utils import radius_graph_pbc_e3, repeat_blocks

MAX_ATOMIC_NUM=100

class SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies = 10, n_space = 3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class AtomEmbedding(nn.Module):
    """
    Initial atom embeddings based on the atom type
    Parameters:
        emb_size: int       Atom embeddings size
    """
    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size
        self.embeddings = nn.Embedding(MAX_ATOMIC_NUM, emb_size)
        nn.init.uniform_(
            self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3)
        )

    def forward(self, Z):
        """
        Returns:
            h: torch.Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.embeddings(Z-1)
        return h
    
class FourierFeatures(nn.Module):
    """From VDM"""
    def __init__(self, first=5.0, last=6.0, step=1.0):
        super().__init__()
        self.freqs_exponent = torch.arange(first, last + 1e-8, step)

    @property
    def num_features(self):
        return len(self.freqs_exponent) * 2

    def forward(self, x):
        assert len(x.shape) >= 2

        # Compute (2pi * 2^n) for n in freqs.
        freqs_exponent = self.freqs_exponent.to(dtype=x.dtype, device=x.device)  # (F, )
        freqs = 2.0**freqs_exponent * 2 * torch.pi  # (F, )
        freqs = freqs.view(-1, *([1] * (x.dim() - 1)))  # (F, 1, 1, ...)

        # Compute (2pi * 2^n * x) for n in freqs.
        features = freqs * x.unsqueeze(1)  # (B, F, X1, X2, ...)
        features = features.flatten(1, 2)  # (B, F * C, X1, X2, ...)

        # Output features are cos and sin of above. Shape (B, 2 * F * C, H, W).
        return torch.cat([features.sin(), features.cos()], dim=1)

class E3Layer(nn.Module):
    """ Message passing layer for cspnet."""
    def __init__(
        self,
        hidden_dim=128,
        act_fn=nn.SiLU(),
        dis_emb=None,
        ln=False,
        ip=True
    ):
        super(E3Layer, self).__init__()

        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = True
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def edge_model(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff = None):

        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        if self.ip:
            lattice_ips = lattices @ lattices.transpose(-1,-2)
        else:
            lattice_ips = lattices
        lattice_ips_flatten = lattice_ips.view(-1, 9)
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        edges_input = torch.cat([hi, hj, lattice_ips_flatten_edges, frac_diff], dim=1)
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):

        agg = scatter(edge_features, edge_index[0], dim = 0, reduce='mean', dim_size=node_features.shape[0])
        agg = torch.cat([node_features, agg], dim = 1)
        out = self.node_mlp(agg)
        return out

    def forward(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff = None):

        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff)
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output


class E3GNN(nn.Module):

    def __init__(
        self,
        hidden_dim = 128,           
        time_dim = 256,             
        latent_dim = 128,          
        num_layers = 4,             
        max_atoms = 100,            
        act_fn = 'silu',            
        dis_emb = 'sin',            
        num_freqs = 10,             
        edge_style = 'fc',          
        cutoff = 6.0,               
        max_neighbors = 20,         
        ln = False,                 
        ip = True,                  
        smooth = False,
        pos_emb = 'none',
        fourier_emb = True,
        # 新增两个控制项
        type_from_layer: int = -1,  
        type_grad_alpha: float = 1.0,
    ):
        super(E3GNN, self).__init__()

        self.ip = ip
        self.smooth = smooth
        self.fourier_emb = fourier_emb
        input_dim = 3
        if self.smooth:
            self.node_embedding = AtomEmbedding(hidden_dim)
        else:
            self.node_embedding = nn.Linear(max_atoms, hidden_dim)
        if self.fourier_emb:
            first, last,step = 0,hidden_dim//6-1,1
            fourier_dim = 2*(last-first+1)*3
            self.fourier_pos = FourierFeatures(first, last, step)
        else:
            fourier_dim = 0        
        if pos_emb == 'none':
            pos_dim=0
            self.pos_emb = nn.Linear(input_dim,pos_dim)
        elif pos_emb == 'fc':
            pos_dim = hidden_dim-fourier_dim
            self.pos_emb = nn.Linear(input_dim, pos_dim)
            nn.init.uniform_(
                self.pos_emb.weight, a=-np.sqrt(3), b=np.sqrt(3)
            )    
        self.atom_latent_emb = nn.Linear(hidden_dim+time_dim+latent_dim+pos_dim+fourier_dim, hidden_dim*2)
        if act_fn == 'silu':
            self.act_fn = nn.SiLU()
        if dis_emb == 'sin':
            self.dis_emb = SinusoidsEmbedding(n_frequencies = num_freqs)
        elif dis_emb == 'none':
            self.dis_emb = None
        for i in range(0, num_layers):
            self.add_module(
                "E3_layer_%d" % i, E3Layer(hidden_dim*2, self.act_fn, self.dis_emb, ln=ln, ip=ip)
            )            
        self.time_dim = time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.num_layers = num_layers
        self.coord_out = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim),
                                     #nn.BatchNorm1d(hidden_dim//2),
                                     #nn.LeakyReLU(negative_slope=0.01),
                                     #nn.ReLU(),
                                     nn.PReLU(),
                                     nn.Linear(hidden_dim, 6),
                                     #nn.Tanh()
                                     )
        self.type_out = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim),
                                     nn.PReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(hidden_dim, MAX_ATOMIC_NUM),
                                     )
        """ self.type_out = nn.Sequential(
                                    nn.LayerNorm(hidden_dim*2),
                                    nn.Linear(hidden_dim*2, hidden_dim*2),
                                    nn.PReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(hidden_dim*2, hidden_dim),
                                    nn.PReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(hidden_dim, MAX_ATOMIC_NUM),
                                    ) """
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.ln = ln
        self.edge_style = edge_style
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim*2)
        # 记录控制项
        self.type_from_layer = type_from_layer
        self.type_grad_alpha = type_grad_alpha
        
    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(
        self, edge_index, cell_offsets, neighbors, edge_vector
    ):
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running torch.unique.
        But this does not seem worth it.
        """

        # Generate mask
        mask_sep_atoms = edge_index[0] < edge_index[1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | (
                (cell_offsets[:, 0] == 0)
                & (cell_offsets[:, 1] == 0)
                & (cell_offsets[:, 2] < 0)
            )
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [
                edge_index_new,
                torch.stack([edge_index_new[1], edge_index_new[0]], dim=0),
            ],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device),
            neighbors,
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * torch.bincount(
            batch_edge, minlength=neighbors.size(0)
        )

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(
            cell_offsets, mask, edge_reorder_idx, True
        )
        edge_vector_new = self.select_symmetric_edges(
            edge_vector, mask, edge_reorder_idx, True
        )

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_vector_new,
        )

    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph):

        if self.edge_style == 'fc':
            lis = [torch.ones(n,n, device=num_atoms.device) for n in num_atoms]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.
        elif self.edge_style == 'knn':
            lattice_nodes = lattices[node2graph]
            cart_coords = torch.einsum('bi,bij->bj', frac_coords, lattice_nodes)
            
            edge_index, to_jimages, num_bonds = radius_graph_pbc_e3(
                cart_coords, None, None, num_atoms, self.cutoff, self.max_neighbors,
                device=num_atoms.device, lattices=lattices)

            j_index, i_index = edge_index
            distance_vectors = frac_coords[j_index] - frac_coords[i_index]
            distance_vectors += to_jimages.float()

            edge_index_new, _, _, edge_vector_new = self.reorder_symmetric_edges(edge_index, to_jimages, num_bonds, distance_vectors)

            return edge_index_new, -edge_vector_new
            

    def forward(self, z, t, atom_types, frac_coords, lattices, num_atoms, node2graph):
        
        t = self.time_embedding(t)
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]
        node_features = self.node_embedding(atom_types)

        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pos_features = self.pos_emb(frac_coords)
        if self.fourier_emb:
            extra_features = self.fourier_pos(frac_coords)
            node_features = torch.cat([node_features, z_per_atom, t_per_atom, pos_features, extra_features], dim=1)
        else:
            node_features = torch.cat([node_features, z_per_atom, t_per_atom, pos_features], dim=1)
        node_features = self.atom_latent_emb(node_features)

        # 记录每层特征，供类型分支选择
        feats = []
        for i in range(0, self.num_layers):
            node_features = self._modules["E3_layer_%d" % i](
                node_features, frac_coords, lattices, edges, edge2graph, frac_diff=frac_diff
            )
            feats.append(node_features)

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        # 坐标分支：输出三轴角度
        vec = self.coord_out(node_features).view(-1, 3, 2)   # (N_atoms, 3, 2)
        vec = F.normalize(vec, dim=-1, eps=1e-6)             # 归一化避免数值漂移
        sin_v, cos_v = vec[..., 0], vec[..., 1]
        coord_out = torch.atan2(sin_v, cos_v)
        
        type_out = self.type_out(node_features)

        return coord_out, type_out


