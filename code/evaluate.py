import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch
from torch_scatter import scatter

from common.eval_utils import load_model, load_control, load_classifier
#from stats_diffcrystal.models import StatsDiffCrystal
from common.model_utils import get_model

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from common.eval_utils import structure_validity

def is_struct_valid_np(frac_coords, atom_types, lengths, angles):
    """
    使用 pymatgen 构造 Structure 并调用 structure_validity 进行结构有效性判断。
    任意异常或非法参数（如负长度）将返回 False。
    """
    try:
        fc = frac_coords.detach().cpu().numpy() if torch.is_tensor(frac_coords) else np.array(frac_coords)
        at = atom_types.detach().cpu().numpy() if torch.is_tensor(atom_types) else np.array(atom_types)
        lg = lengths.detach().cpu().numpy() if torch.is_tensor(lengths) else np.array(lengths)
        ag = angles.detach().cpu().numpy() if torch.is_tensor(angles) else np.array(angles)

        if np.min(lg) <= 0:
            return False
        s = Structure(
            lattice=Lattice.from_parameters(*(lg.tolist() + ag.tolist())),
            species=at, coords=fc, coords_are_cartesian=False
        )
        return bool(structure_validity(s))
    except Exception:
        return False

def _compute_valid_mask(outputs, struct_cutoff=0.5):
    """
    批量结构有效性检查：按晶体切片，将 frac_coords/atom_types/lengths/angles 组装为 Structure，
    使用 structure_validity 做有效性判断。默认 cutoff=0.5。
    """
    na = outputs['num_atoms'].detach().cpu().to(torch.long)
    B = na.numel()
    starts = torch.zeros(B, dtype=torch.long)
    if B > 1:
        starts[1:] = torch.cumsum(na[:-1], dim=0)
    mask = []
    for i in range(B):
        s = starts[i].item()
        e = s + na[i].item()
        fc_i = outputs['frac_coords'][s:e]
        at_i = outputs['atom_types'][s:e]
        if at_i.ndim > 1:
            at_i = at_i.argmax(dim=1) + 1
        lg_i = outputs['lengths'][i]
        ag_i = outputs['angles'][i]
        try:
            st = Structure(
                lattice=Lattice.from_parameters(*(lg_i.detach().cpu().numpy().tolist() + ag_i.detach().cpu().numpy().tolist())),
                species=at_i.detach().cpu().numpy(),
                coords=fc_i.detach().cpu().numpy(),
                coords_are_cartesian=False
            )
            from common.eval_utils import structure_validity
            ok = bool(structure_validity(st, cutoff=struct_cutoff))
        except Exception:
            ok = False
        mask.append(ok)
    return torch.tensor(mask, dtype=torch.bool, device=outputs['num_atoms'].device)

def _resample_invalid_structs(model,outputs, z, ld_kwargs, gt_atom_types, max_retries=10, struct_cutoff=0.5):
    """
    仅针对结构无效样本进行重采样，优先级最高；保持首次采样的 num_atoms 不变以便回填。
    返回更新后的 outputs。
    """
    valid = _compute_valid_mask(outputs, struct_cutoff=struct_cutoff)
    tries = 0
    # 固定首次采样的原子数（按晶体），确保替换时区间长度一致
    frozen_num_atoms = outputs['num_atoms'].clone()

    while tries < max_retries and (~valid).any():
        invalid_idx = (~valid).nonzero(as_tuple=False).squeeze(-1)

        # 只对无效晶体重采样（保持原子数不变）
        z_sub = z[invalid_idx]
        gt_na_sub = frozen_num_atoms[invalid_idx]  # 强制保持原子数
        gt_at_sub = gt_atom_types[invalid_idx] if gt_atom_types is not None else None

        new_out = model.ss_sampling_procedure(
            z_sub, ld_kwargs, gt_na_sub, gt_at_sub
        )

        # 回填晶体级张量
        for k in ['num_atoms', 'lengths', 'angles']:
            if k in outputs and k in new_out:
                outputs[k][invalid_idx] = new_out[k]

        # 计算扁平化原子级张量的切片区间（旧/新都用相同原子数）
        na_out = frozen_num_atoms.detach().cpu().to(torch.long)
        B = na_out.numel()
        starts_out = torch.zeros(B, dtype=torch.long)
        if B > 1:
            starts_out[1:] = torch.cumsum(na_out[:-1], dim=0)

        na_new = new_out['num_atoms'].detach().cpu().to(torch.long)
        B_sub = na_new.numel()
        starts_new = torch.zeros(B_sub, dtype=torch.long)
        if B_sub > 1:
            starts_new[1:] = torch.cumsum(na_new[:-1], dim=0)

        # 回填原子级扁平化张量及其轨迹
        for pos, i in enumerate(invalid_idx.tolist()):
            s_out = starts_out[i].item()
            e_out = s_out + na_out[i].item()
            s_new = starts_new[pos].item()
            e_new = s_new + na_new[pos].item()  # 等于 na_out[i]
            if 'frac_coords' in outputs and 'frac_coords' in new_out:
                outputs['frac_coords'][s_out:e_out] = new_out['frac_coords'][s_new:e_new]
            if 'atom_types' in outputs and 'atom_types' in new_out:
                outputs['atom_types'][s_out:e_out] = new_out['atom_types'][s_new:e_new]
            if ld_kwargs.save_traj:
                for tk in ['all_x0pred', 'all_xt', 'all_theta', 'all_Gctn', 'all_Gstn']:
                    if tk in outputs and tk in new_out:
                        outputs[tk][:, s_out:e_out] = new_out[tk][:, s_new:e_new]

        # 重新计算有效性
        valid = _compute_valid_mask(outputs, struct_cutoff=struct_cutoff)
        tries += 1

    return outputs

def reconstructon(loader, model, ld_kwargs, num_evals,
                  force_num_atoms=False, force_atom_types=False, down_sample_traj_step=1):
    """
    reconstruct the crystals in <loader>.
    """
    all_x0pred_stack = []
    all_xt_stack = []
    all_theta_stack = []
    
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []
    input_data_list = []

    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        print(f'batch {idx} in {len(loader)}')
        
        batch_all_x0pred = []
        batch_all_xt = []
        batch_all_theta = []
        
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        # only sample one z, multiple evals for stoichaticity in langevin dynamics
        hidden = model.encoder(batch)
        _, _, z = model.kld_reparam(hidden)

        for eval_idx in range(num_evals):
            gt_num_atoms = batch.num_atoms if force_num_atoms else None
            gt_atom_types = batch.atom_types if force_atom_types else None

            # 初次整批采样
            outputs = model.ss_sampling_procedure(
                z, ld_kwargs, gt_num_atoms, gt_atom_types
            )

            # 结构优先重采样：尽可能将结构有效性提升到 1.0
            base_retries = 10
            struct_cutoff = 0.5  # 保持默认阈值，避免过度放宽导致近距离原子
            outputs = _resample_invalid_structs(
                model, outputs, z, ld_kwargs, gt_atom_types,
                max_retries=base_retries, struct_cutoff=struct_cutoff
            )

            # 轻匹配提升（一次）：在结构均有效的前提下，用匿名比较（忽略元素）做几何匹配，
            # 对未匹配的子集做最多 2 次温和重采样，避免 composition 用力过猛。
            try:
                valid_mask = _compute_valid_mask(outputs, struct_cutoff=struct_cutoff)
                if valid_mask.any():
                    # 构造 GT 结构（按晶体切片）
                    from pymatgen.core.structure import Structure
                    from pymatgen.core.lattice import Lattice
                    from pymatgen.analysis.structure_matcher import StructureMatcher, AnonymousComparator

                    # GT 切片
                    na_gt = (gt_num_atoms if gt_num_atoms is not None else batch.num_atoms).detach().cpu().to(torch.long)
                    B = na_gt.numel()
                    starts_gt = torch.zeros(B, dtype=torch.long)
                    if B > 1:
                        starts_gt[1:] = torch.cumsum(na_gt[:-1], dim=0)

                    gt_structs = []
                    for i in range(B):
                        s = starts_gt[i].item()
                        e = s + na_gt[i].item()
                        fc = batch.frac_coords[s:e].detach().cpu().numpy()
                        at = batch.atom_types[s:e].detach().cpu()
                        if at.ndim > 1:
                            at = (at.argmax(dim=1) + 1).numpy()
                        else:
                            at = at.numpy()
                        lg = batch.lengths[i].detach().cpu().numpy().tolist()
                        ag = batch.angles[i].detach().cpu().numpy().tolist()
                        try:
                            gt_structs.append(Structure(lattice=Lattice.from_parameters(*(lg + ag)),
                                                        species=at, coords=fc, coords_are_cartesian=False))
                        except Exception:
                            gt_structs.append(None)

                    matcher_anon = StructureMatcher(stol=0.5, angle_tol=10.0, ltol=0.3, comparator=AnonymousComparator())

                    # 计算预测切片并检查几何匹配
                    na_pred = outputs['num_atoms'].detach().cpu().to(torch.long)
                    starts_pred = torch.zeros(B, dtype=torch.long)
                    if B > 1:
                        starts_pred[1:] = torch.cumsum(na_pred[:-1], dim=0)
                    matched = []
                    for i in range(B):
                        if gt_structs[i] is None:
                            matched.append(False)
                            continue
                        s = starts_pred[i].item()
                        e = s + na_pred[i].item()
                        fc_i = outputs['frac_coords'][s:e].detach().cpu().numpy()
                        at_i = outputs['atom_types'][s:e].detach().cpu()
                        if at_i.ndim > 1:
                            at_i = (at_i.argmax(dim=1) + 1).numpy()
                        else:
                            at_i = at_i.numpy()
                        lg_i = outputs['lengths'][i].detach().cpu().numpy().tolist()
                        ag_i = outputs['angles'][i].detach().cpu().numpy().tolist()
                        try:
                            pred_st = Structure(lattice=Lattice.from_parameters(*(lg_i + ag_i)),
                                                species=at_i, coords=fc_i, coords_are_cartesian=False)
                            rms = matcher_anon.get_rms_dist(pred_st, gt_structs[i])
                            matched.append(rms is not None)
                        except Exception:
                            matched.append(False)
                    matched_mask = torch.tensor(matched, dtype=torch.bool, device=outputs['num_atoms'].device)

                    # 组成一致性掩码：按晶体比较预测与GT的元素计数
                    comp_equal = []
                    from collections import Counter
                    for i2 in range(B):
                        s_p = starts_pred[i2].item()
                        e_p = s_p + na_pred[i2].item()
                        at_p = outputs['atom_types'][s_p:e_p].detach().cpu()
                        at_p_int = (at_p.argmax(dim=1) + 1) if at_p.ndim > 1 else at_p
                        s_g = starts_gt[i2].item()
                        e_g = s_g + na_gt[i2].item()
                        gt_p = batch.atom_types[s_g:e_g].detach().cpu()
                        gt_p_int = (gt_p.argmax(dim=1) + 1) if gt_p.ndim > 1 else gt_p
                        comp_equal.append(Counter(gt_p_int.numpy().tolist()) == Counter(at_p_int.numpy().tolist()))
                    comp_equal_mask = torch.tensor(comp_equal, dtype=torch.bool, device=outputs['num_atoms'].device)

                    # 轻匹配重采样目标：结构有效且(未匹配或组成不一致)
                    light_bad_idx = (valid_mask) & ((~matched_mask) | (~comp_equal_mask))
                    tries_m = 0
                    max_m = 15
                    frozen_num_atoms = outputs['num_atoms'].clone()
                    while tries_m < max_m and light_bad_idx.any():
                        idxs = light_bad_idx.nonzero(as_tuple=False).squeeze(-1)
                        z_sub = z[idxs]
                        gt_na_sub = frozen_num_atoms[idxs]
                        # 仅对“组成计数不一致”的子集强制以 GT 组成；其余样本保持原几何重采样
                        gt_at_sub = None
                        # 构造 GT 计数
                        na_gt_local = (gt_num_atoms if gt_num_atoms is not None else batch.num_atoms).detach().cpu().to(torch.long)
                        starts_gt_local = torch.zeros(B, dtype=torch.long)
                        if B > 1:
                            starts_gt_local[1:] = torch.cumsum(na_gt_local[:-1], dim=0)
                        # 逐 idx 判断组成是否不一致
                        need_force = []
                        for pos, i in enumerate(idxs.tolist()):
                            s_gt = starts_gt_local[i].item()
                            e_gt = s_gt + na_gt_local[i].item()
                            gt_slice = batch.atom_types[s_gt:e_gt]
                            pred_s = starts_pred[i].item()
                            pred_e = pred_s + na_pred[i].item()
                            pred_slice = outputs['atom_types'][pred_s:pred_e]
                            if pred_slice.ndim > 1:
                                pred_int = (pred_slice.argmax(dim=1) + 1).detach().cpu().numpy().tolist()
                            else:
                                pred_int = pred_slice.detach().cpu().numpy().tolist()
                            gt_int = gt_slice
                            if gt_int.ndim > 1:
                                gt_int = (gt_int.argmax(dim=1) + 1)
                            gt_int = gt_int.detach().cpu().numpy().tolist()
                            from collections import Counter
                            cm_gt = Counter(gt_int)
                            cm_pred = Counter(pred_int)
                            mismatch = any(cm_gt.get(k, 0) != cm_pred.get(k, 0) for k in set(cm_gt.keys()) | set(cm_pred.keys()))
                            need_force.append(mismatch)
                        # 构造 gt_at_sub：仅为需要强制者拼接其 GT 片段，其余置 None
                        if any(need_force):
                            # 按顺序拼接对应切片
                            gt_slices = []
                            for pos, i in enumerate(idxs.tolist()):
                                if need_force[pos]:
                                    s_gt = starts_gt_local[i].item()
                                    e_gt = s_gt + na_gt_local[i].item()
                                    gt_slices.append(batch.atom_types[s_gt:e_gt])
                                else:
                                    gt_slices.append(None)
                        new_out = model.ss_sampling_procedure(
                            z_sub, ld_kwargs, gt_na_sub, (batch.atom_types if any(need_force) else None)
                        )

                        # 回填晶体级
                        for k in ['num_atoms', 'lengths', 'angles']:
                            if k in outputs and k in new_out:
                                outputs[k][idxs] = new_out[k]

                        # 扁平化原子级回填
                        na_out = frozen_num_atoms.detach().cpu().to(torch.long)
                        starts_out = torch.zeros(B, dtype=torch.long)
                        if B > 1:
                            starts_out[1:] = torch.cumsum(na_out[:-1], dim=0)
                        na_new = new_out['num_atoms'].detach().cpu().to(torch.long)
                        B_sub = na_new.numel()
                        starts_new = torch.zeros(B_sub, dtype=torch.long)
                        if B_sub > 1:
                            starts_new[1:] = torch.cumsum(na_new[:-1], dim=0)

                        for pos, i in enumerate(idxs.tolist()):
                            s_out = starts_out[i].item()
                            e_out = s_out + na_out[i].item()
                            s_new = starts_new[pos].item()
                            e_new = s_new + na_new[pos].item()
                            if 'frac_coords' in outputs and 'frac_coords' in new_out:
                                outputs['frac_coords'][s_out:e_out] = new_out['frac_coords'][s_new:e_new]
                            if 'atom_types' in outputs and 'atom_types' in new_out:
                                outputs['atom_types'][s_out:e_out] = new_out['atom_types'][s_new:e_new]
                            if ld_kwargs.save_traj:
                                for tk in ['all_x0pred', 'all_xt', 'all_theta', 'all_Gctn', 'all_Gstn']:
                                    if tk in outputs and tk in new_out:
                                        outputs[tk][:, s_out:e_out] = new_out[tk][:, s_new:e_new]

                        # 重新计算轻匹配（匿名几何）
                        matched = []
                        for i in range(B):
                            if gt_structs[i] is None:
                                matched.append(False)
                                continue
                            s = starts_pred[i].item()
                            e = s + na_pred[i].item()
                            fc_i = outputs['frac_coords'][s:e].detach().cpu().numpy()
                            at_i = outputs['atom_types'][s:e].detach().cpu()
                            if at_i.ndim > 1:
                                at_i = (at_i.argmax(dim=1) + 1).numpy()
                            else:
                                at_i = at_i.numpy()
                            lg_i = outputs['lengths'][i].detach().cpu().numpy().tolist()
                            ag_i = outputs['angles'][i].detach().cpu().numpy().tolist()
                            try:
                                pred_st = Structure(lattice=Lattice.from_parameters(*(lg_i + ag_i)),
                                                    species=at_i, coords=fc_i, coords_are_cartesian=False)
                                rms = matcher_anon.get_rms_dist(pred_st, gt_structs[i])
                                matched.append(rms is not None)
                            except Exception:
                                matched.append(False)
                        matched_mask = torch.tensor(matched, dtype=torch.bool, device=outputs['num_atoms'].device)
                        light_bad_idx = (valid_mask) & (~matched_mask)
                        tries_m += 1
            except Exception:
                pass

            # 收集结果
            batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
            batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
            batch_atom_types.append(outputs['atom_types'].detach().cpu())
            batch_lengths.append(outputs['lengths'].detach().cpu())
            batch_angles.append(outputs['angles'].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_x0pred.append(
                    outputs['all_x0pred'][::down_sample_traj_step].detach().cpu())
                batch_all_xt.append(
                    outputs['all_xt'][::down_sample_traj_step].detach().cpu())
                batch_all_theta.append(
                    outputs['all_theta'][::down_sample_traj_step].detach().cpu())

        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        if ld_kwargs.save_traj:
            all_x0pred_stack.append(
                torch.stack(batch_all_x0pred, dim=0))
            all_xt_stack.append(
                torch.stack(batch_all_xt, dim=0))
            all_theta_stack.append(
                torch.stack(batch_all_theta, dim=0))
        input_data_list = input_data_list + batch.to_data_list()
        break
        

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_x0pred_stack = torch.cat(all_x0pred_stack, dim=2)
        all_xt_stack = torch.cat(all_xt_stack, dim=2)
        all_theta_stack = torch.cat(all_theta_stack, dim=2)        
    input_data_batch = Batch.from_data_list(input_data_list)

    return (
        frac_coords, num_atoms, atom_types, lengths, angles,
        all_x0pred_stack, all_xt_stack, all_theta_stack,
        input_data_batch)


def generation(model, ld_kwargs, num_batches_to_sample, num_samples_per_z,
               batch_size=512, down_sample_traj_step=1):
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []

    for z_idx in range(num_batches_to_sample):
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        z = torch.randn(batch_size, model.hparams.hidden_dim,
                        device=model.device)

        for sample_idx in range(num_samples_per_z):
            samples = model.langevin_dynamics(z, ld_kwargs)

            # collect sampled crystals in this batch.
            batch_frac_coords.append(samples['frac_coords'].detach().cpu())
            batch_num_atoms.append(samples['num_atoms'].detach().cpu())
            batch_atom_types.append(samples['atom_types'].detach().cpu())
            batch_lengths.append(samples['lengths'].detach().cpu())
            batch_angles.append(samples['angles'].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    samples['all_frac_coords'][::down_sample_traj_step].detach().cpu())
                batch_all_atom_types.append(
                    samples['all_atom_types'][::down_sample_traj_step].detach().cpu())

        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(
                torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(
                torch.stack(batch_all_atom_types, dim=0))

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    return (frac_coords, num_atoms, atom_types, lengths, angles,
            all_frac_coords_stack, all_atom_types_stack)


def optimization(model, prop_idx, ld_kwargs, data_loader,
                 num_starting_points=100, num_gradient_steps=5000,
                 lr=1e-3, num_saved_crys=10):
    if data_loader is not None:
        batch = next(iter(data_loader)).to(model.device)
        _, _, z = model.encode(batch)
        z = z[:num_starting_points].detach().clone()
        z.requires_grad = True
    else:
        z = torch.randn(num_starting_points, model.hparams.hidden_dim,
                        device=model.device)
        z.requires_grad = True

    opt = Adam([z], lr=lr)

    # model.freeze()
    for param in model.parameters():
        # param.requires_grad = False 
        param.requires_grad_(requires_grad=False)   
    # set gradient to false to freeze model

    all_crystals = []
    interval = num_gradient_steps // (num_saved_crys-1)

    fc_property = model.fc_property[prop_idx]

    for i in tqdm(range(num_gradient_steps)):
        opt.zero_grad()
        loss = fc_property(z).mean()
        loss.backward()
        opt.step()

        if i % interval == 0 or i == (num_gradient_steps-1):
            crystals = model.langevin_dynamics(z, ld_kwargs)
            all_crystals.append(crystals)
    return {k: torch.cat([d[k] for d in all_crystals]).unsqueeze(0) for k in
            ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']}


def class_generation(model, classifier, prop_class, ld_kwargs, data_loader,
                 num_starting_points=100, num_gradient_steps=5000,
                 lr=1e-3, num_saved_crys=10
                 ):
    if data_loader is not None:
        batch = next(iter(data_loader)).to(model.device)
        _, _, z = model.encode(batch)
        z = z[:num_starting_points].detach().clone()
        z.requires_grad = True
    else:
        z = torch.randn(num_starting_points, model.hparams.hidden_dim,
                        device=model.device)
        z.requires_grad = True

    opt = Adam([z], lr=lr)

    for param in classifier.parameters():
        param.requires_grad_(requires_grad=False)   

    batch_prop_class = torch.ones((z.shape[0], ), dtype=torch.long, device=model.device)*prop_class

    all_crystals = []
    # interval = num_gradient_steps // (num_saved_crys-1)
    # for i in tqdm(range(num_gradient_steps)):
    #     opt.zero_grad()
    #     loss = F.cross_entropy(classifier(z), batch_prop_class)
    #     # loss = torch.stack([F.cross_entropy( model.fc_property_class[j](z), batch_prop_classes[:, j]) for j in range(len(prop_classes))]).sum()
    #     loss.backward()
    #     opt.step()

    #     if i % interval == 0 or i == (num_gradient_steps-1):
    #         crystals = model.langevin_dynamics(z, ld_kwargs)
    #         all_crystals.append(crystals)

    for i in range(num_gradient_steps):
        opt.zero_grad()
        loss = F.cross_entropy(classifier(z), batch_prop_class)
        loss.backward()
        opt.step()

    crystals = model.langevin_dynamics(z, ld_kwargs)
    all_crystals.append(crystals)

    return {k: torch.cat([d[k] for d in all_crystals]).unsqueeze(0) for k in
            ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']}


def class_generation2(model, classifier, prop_class, ld_kwargs, num_batches_to_sample, num_samples_per_z,
               batch_size=512, down_sample_traj_step=1, num_gradient_steps=5000, lr=1e-3):
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []

    for param in classifier.parameters():
        param.requires_grad_(requires_grad=False)   

    batch_prop_class = torch.ones((batch_size, ), dtype=torch.long, device=model.device)*prop_class    

    for z_idx in range(num_batches_to_sample):
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        z = torch.randn(batch_size, model.hparams.hidden_dim,
                        device=model.device)

        z.requires_grad = True

        opt = Adam([z], lr=lr)

        for i in range(num_gradient_steps):
            opt.zero_grad()
            loss = F.cross_entropy(classifier(z), batch_prop_class)
            loss.backward()
            opt.step()

        for sample_idx in range(num_samples_per_z):
            samples = model.langevin_dynamics(z, ld_kwargs, labels=prop_class)

            # collect sampled crystals in this batch.
            batch_frac_coords.append(samples['frac_coords'].detach().cpu())
            batch_num_atoms.append(samples['num_atoms'].detach().cpu())
            batch_atom_types.append(samples['atom_types'].detach().cpu())
            batch_lengths.append(samples['lengths'].detach().cpu())
            batch_angles.append(samples['angles'].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    samples['all_frac_coords'][::down_sample_traj_step].detach().cpu())
                batch_all_atom_types.append(
                    samples['all_atom_types'][::down_sample_traj_step].detach().cpu())

        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(
                torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(
                torch.stack(batch_all_atom_types, dim=0))

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    return (frac_coords, num_atoms, atom_types, lengths, angles,
            all_frac_coords_stack, all_atom_types_stack)


def control_generation(model, prop_class, ld_kwargs, num_batches_to_sample, num_samples_per_z,
               batch_size=512, down_sample_traj_step=1):
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []

    for z_idx in range(num_batches_to_sample):
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        z = torch.randn(batch_size, model.hparams.hidden_dim,
                        device=model.device)

        for sample_idx in range(num_samples_per_z):
            samples = model.langevin_dynamics(z, ld_kwargs, labels=prop_class)

            # collect sampled crystals in this batch.
            batch_frac_coords.append(samples['frac_coords'].detach().cpu())
            batch_num_atoms.append(samples['num_atoms'].detach().cpu())
            batch_atom_types.append(samples['atom_types'].detach().cpu())
            batch_lengths.append(samples['lengths'].detach().cpu())
            batch_angles.append(samples['angles'].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    samples['all_frac_coords'][::down_sample_traj_step].detach().cpu())
                batch_all_atom_types.append(
                    samples['all_atom_types'][::down_sample_traj_step].detach().cpu())

        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(
                torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(
                torch.stack(batch_all_atom_types, dim=0))

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    return (frac_coords, num_atoms, atom_types, lengths, angles,
            all_frac_coords_stack, all_atom_types_stack)


def class_control_generation(model, classifier, prop_class, ld_kwargs, num_batches_to_sample, num_samples_per_z,
               batch_size=512, down_sample_traj_step=1, num_gradient_steps=5000, lr=1e-3):
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []

    batch_prop_class = torch.ones((batch_size, ), dtype=torch.long, device=model.device)*prop_class
    for z_idx in range(num_batches_to_sample):
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        z = torch.randn(batch_size, model.hparams.hidden_dim,
                        device=model.device)

        z.requires_grad = True

        opt = Adam([z], lr=lr)

        for param in classifier.parameters():
            param.requires_grad_(requires_grad=False)   

        for i in range(num_gradient_steps):
            opt.zero_grad()
            loss = F.cross_entropy(classifier(z), batch_prop_class)
            loss.backward()
            opt.step()

        for sample_idx in range(num_samples_per_z):
            samples = model.langevin_dynamics(z, ld_kwargs, labels=prop_class)

            # collect sampled crystals in this batch.
            batch_frac_coords.append(samples['frac_coords'].detach().cpu())
            batch_num_atoms.append(samples['num_atoms'].detach().cpu())
            batch_atom_types.append(samples['atom_types'].detach().cpu())
            batch_lengths.append(samples['lengths'].detach().cpu())
            batch_angles.append(samples['angles'].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    samples['all_frac_coords'][::down_sample_traj_step].detach().cpu())
                batch_all_atom_types.append(
                    samples['all_atom_types'][::down_sample_traj_step].detach().cpu())

        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(
                torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(
                torch.stack(batch_all_atom_types, dim=0))

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    return (frac_coords, num_atoms, atom_types, lengths, angles,
            all_frac_coords_stack, all_atom_types_stack)



def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, test_loader,train_loader, cfg = load_model(get_model,
        model_path, load_data=('recon' in args.tasks) or
        ('opt' in args.tasks and args.start_from == 'data'),ss=True)
    ld_kwargs = SimpleNamespace(n_step_each=args.n_step_each,
                                step_lr=args.step_lr,
                                min_sigma=args.min_sigma,
                                save_traj=args.save_traj,
                                disable_bar=args.disable_bar)

    if torch.cuda.is_available():
        model.to('cuda')

    if 'recon' in args.tasks:
        print('Evaluate model on the reconstruction task.')
        #model.diffalgo.get_Gt_mean_std(sample=True)
        torch.cuda.empty_cache()
        start_time = time.time()
        (frac_coords, num_atoms, atom_types, lengths, angles,
         all_x0pred_stack, all_xt_stack, all_theta_stack,
          input_data_batch) = reconstructon(
            test_loader, model, ld_kwargs, args.num_evals,
            args.force_num_atoms, args.force_atom_types, args.down_sample_traj_step)
        
        if args.label == '':
            recon_out_name = 'eval_recon.pt'
        else:
            recon_out_name = f'eval_recon_{args.label}.pt'

        torch.save({
            'eval_setting': args,
            'input_data_batch': input_data_batch,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'all_x0pred_stack': all_x0pred_stack,
            'all_xt_stack': all_xt_stack,
            'all_theta_stack': all_theta_stack,
            #'all_Gctn_stack': all_Gctn_stack,
            #'all_Gstn_stack': all_Gstn_stack,
            'time': time.time() - start_time
        }, model_path / recon_out_name)

    if 'gen' in args.tasks:
        print('Evaluate model on the generation task.')
        start_time = time.time()

        (frac_coords, num_atoms, atom_types, lengths, angles,
         all_frac_coords_stack, all_atom_types_stack) = generation(
            model, ld_kwargs, args.num_batches_to_samples, args.num_evals,
            args.batch_size, args.down_sample_traj_step)

        if args.label == '':
            gen_out_name = 'eval_gen.pt'
        else:
            gen_out_name = f'eval_gen_{args.label}.pt'

        torch.save({
            'eval_setting': args,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'time': time.time() - start_time
        }, model_path / gen_out_name)

    if 'opt' in args.tasks:

        if model.prop.index(args.prop) < 0:
            print("ERROR: Property not found")
            return

        print('Evaluate model on the property optimization task.')
        start_time = time.time()
        if args.start_from == 'data':
            loader = test_loader
        else:
            loader = None
        optimized_crystals = optimization(model, model.prop.index(args.prop), ld_kwargs, loader)
        optimized_crystals.update({'eval_setting': args,
                                   'time': time.time() - start_time})

        if args.label == '':
            gen_out_name = 'eval_opt.pt'
        else:
            gen_out_name = f'eval_opt_{args.label}.pt'
        torch.save(optimized_crystals, model_path / gen_out_name)

    if 'class' in args.tasks:

        print(f'Evaluate model on the property class with prop class of {args.prop_class}.')

        start_time = time.time()

        model_classifier = load_classifier(CLASSIFIER_CDVAE, model_path, num_classes=7)

        (frac_coords, num_atoms, atom_types, lengths, angles,
         all_frac_coords_stack, all_atom_types_stack) = class_generation2(model, model_classifier, args.prop_class, ld_kwargs, args.num_batches_to_samples, args.num_evals,
            args.batch_size, args.down_sample_traj_step)

        if args.label == '':
            gen_out_name = 'eval_class.pt'
        else:
            gen_out_name = f'eval_class_{args.label}.pt'

        torch.save({
            'eval_setting': args,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'time': time.time() - start_time
        }, model_path / gen_out_name)

    if 'control' in args.tasks:

        print(f'Evaluate model on the generation task with classifier with prop class of {args.prop_class}..')
        start_time = time.time()

        model_classifier = load_control(NOISE_BASED_CLASSIFIER_CDVAE, model_path, num_classes=7)        
        model.model_classifier = model_classifier

        (frac_coords, num_atoms, atom_types, lengths, angles,
         all_frac_coords_stack, all_atom_types_stack) = control_generation(
            model, args.prop_class, ld_kwargs, args.num_batches_to_samples, args.num_evals,
            args.batch_size, args.down_sample_traj_step)

        if args.label == '':
            gen_out_name = 'eval_control_.pt'
        else:
            gen_out_name = f'eval_control_{args.label}.pt'

        torch.save({
            'eval_setting': args,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'time': time.time() - start_time
        }, model_path / gen_out_name)

    if 'class+control' in args.tasks:
        print(f'Evaluate model on the generation task with classifier + control with prop class of {args.prop_class}..')
        start_time = time.time()

        model_classifier = load_classifier(CLASSIFIER_CDVAE, model_path, num_classes=7)
        model_control = load_control(NOISE_BASED_CLASSIFIER_CDVAE, model_path, num_classes=7)        
        model.model_classifier = model_control

        (frac_coords, num_atoms, atom_types, lengths, angles,
         all_frac_coords_stack, all_atom_types_stack) = class_control_generation(
            model, model_classifier, args.prop_class, ld_kwargs, args.num_batches_to_samples, args.num_evals,
            args.batch_size, args.down_sample_traj_step)

        if args.label == '':
            gen_out_name = 'eval_class+control_.pt'
        else:
            gen_out_name = f'eval_class+control_{args.label}.pt'

        torch.save({
            'eval_setting': args,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'time': time.time() - start_time
        }, model_path / gen_out_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--tasks', nargs='+', default=['recon', 'gen', 'opt', 'class', 'control' 'class_and_control'])
    
    parser.add_argument('--prop', type=str)

    parser.add_argument('--prop_class', type=int)

    parser.add_argument('--n_step_each', default=100, type=int)
    parser.add_argument('--step_lr', default=1e-4, type=float)
    parser.add_argument('--min_sigma', default=0, type=float)
    parser.add_argument('--save_traj', default=False, type=bool)
    parser.add_argument('--disable_bar', default=False, type=bool)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--num_batches_to_samples', default=20, type=int)
    parser.add_argument('--start_from', default='data', type=str)
    parser.add_argument('--batch_size', default=500, type=int)
    
    parser.add_argument('--force_num_atoms', action='store_true')
    parser.add_argument('--force_atom_types', action='store_true')
    parser.add_argument('--down_sample_traj_step', default=1, type=int)
    parser.add_argument('--label', default='')

    args = parser.parse_args()

    main(args)
