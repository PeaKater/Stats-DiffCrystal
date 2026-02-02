import torch
import torch.nn as nn
from torch.nn import functional as F

from gnn.embeddings import MAX_ATOMIC_NUM


# 构建一个多层感知机模型
def build_mlp(in_dim, 
              hidden_dim, 
              fc_num_layers, 
              out_dim, 
              drop_rate=0, 
              norm=False):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    if drop_rate > 0.:
        mods += [nn.Dropout(drop_rate)]
    for i in range(fc_num_layers-1):
        if norm:
            mods += [nn.Linear(hidden_dim, hidden_dim),nn.BatchNorm1d(hidden_dim),nn.Sigmoid()]
        else:
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        if drop_rate > 0.:
            mods += [nn.Dropout(drop_rate)]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)

class MLPDecodeStats(nn.Module):
    def __init__(self, hidden_dim, latent_dim, fc_num_layers, max_atoms,
                 lattice_scale_method=None, teacher_forcing_lattice=False,
                 drop_rate=0, comp_CNN=False):
        super().__init__()
        self.lattice_scale_method = lattice_scale_method
        self.teacher_forcing_lattice = teacher_forcing_lattice
        self.comp_CNN = comp_CNN
        
        self.fc_num_atoms = build_mlp(latent_dim, hidden_dim,
                                    fc_num_layers, max_atoms+1,
                                    drop_rate=drop_rate)
        self.fc_lattice = build_mlp(latent_dim, hidden_dim,
                                            fc_num_layers, 9, drop_rate)
        self.composition = build_mlp(latent_dim, hidden_dim,
                                        fc_num_layers, MAX_ATOMIC_NUM,
                                        drop_rate,norm=True)         
        
        self.lattice_scaler = None

    # 预测原子数目的多层感知机模型
    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)
    
    
    # 预测原子组成的多层感知机模型
    def predict_composition(self, z, num_atoms, l_and_a=None):
        #z = torch.cat([z, l_and_a], dim=-1)
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        if self.comp_CNN:
            z_per_atom = z_per_atom.unsqueeze(1)
        pred_composition_per_atom = self.composition(z_per_atom)
        #pred_composition_per_atom = torch.clamp(pred_composition_per_atom,min=-10,max=10)
        return pred_composition_per_atom
    
    # 预测晶格参数的多层感知机模型
    """ def predict_lattice(self, z, num_atoms, emb_num_atoms=None):
        self.lattice_scaler.match_device(z)
        if emb_num_atoms is not None:
            z = torch.cat([z, emb_num_atoms], dim=-1)
        pred_lengths_and_angles = self.fc_lattice(z)  # (N, 6)
        scaled_preds = self.lattice_scaler.inverse_transform(
            pred_lengths_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        if self.lattice_scale_method == 'scale_length':
            pred_lengths = pred_lengths * num_atoms.view(-1, 1).float()**(1/3)
        # <pred_lengths_and_angles> is scaled.
        return pred_lengths_and_angles, pred_lengths, pred_angles """
        
    def predict_lattice(self, z, num_atoms, emb_num_atoms=None):
        self.lattice_scaler.match_device(z)
        if emb_num_atoms is not None:
            z = torch.cat([z, emb_num_atoms], dim=-1)
        raw = self.fc_lattice(z)  # (N, 9)
        lengths = raw[:, :3]
        cs = torch.tanh(raw[:, 3:])  # 约束到 [-1, 1]
        cos_alpha, sin_alpha, cos_beta, sin_beta, cos_gamma, sin_gamma = cs.unbind(dim=1)
        angles_rad = torch.stack([
            torch.atan2(sin_alpha, cos_alpha),
            torch.atan2(sin_beta, cos_beta),
            torch.atan2(sin_gamma, cos_gamma),
        ], dim=1)
        pred_angles = angles_rad * (180.0 / torch.pi)
        if self.lattice_scale_method == 'scale_length':
            lengths = lengths * num_atoms.view(-1, 1).float() ** (1 / 3)
        # 返回：九维原始输出（用于损失）+ 物理空间长度/角度（用于矩阵/采样）
        return raw, lengths, pred_angles

    def forward(self, z, gt_num_atoms=None, gt_lengths=None, gt_angles=None,
                teacher_forcing=False):        
        """
        decode key stats from latent embeddings.
        batch is input during training for teach-forcing.
        """
        if gt_num_atoms is not None:
            num_atoms = self.predict_num_atoms(z)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, gt_num_atoms))                        

            composition_per_atom = self.predict_composition(z, gt_num_atoms)
            if self.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
        else:
            num_atoms = self.predict_num_atoms(z).argmax(dim=-1)

            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, num_atoms))            
            
            composition_per_atom = self.predict_composition(z, num_atoms)
        return num_atoms, lengths_and_angles, lengths, angles, composition_per_atom

