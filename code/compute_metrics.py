from collections import Counter
import argparse
import os
import json

import numpy as np
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map
from scipy.stats import wasserstein_distance

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty

import warnings

from common.eval_utils import (
    smact_validity, structure_validity, CompScaler, get_fp_pdist,
    load_config, load_data, get_crystals_list, prop_model_eval, compute_cov, class_model_eval)

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'mp20_class': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'mp20_class': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}


class Crystal(object):

    def __init__(self, crys_array_dict, get_argmax=False):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        if get_argmax:
            at = self.atom_types
            if isinstance(at, np.ndarray) and at.ndim == 2:
                self.atom_types = at.argmax(-1) + 1
            # 如果是一维，就不再处理，避免把 [N] 变成标量
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict

        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()

    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        else:
            try:
                warnings.filterwarnings("ignore", category=UserWarning)
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
                # 将体积检查放入构造成功后
                if self.structure.volume < 0.1:
                    self.constructed = False
                    self.invalid_reason = 'unrealistically_small_lattice'
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        #print(self.comp_valid, end='\t')
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
            #print(self.struct_valid)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [CrystalNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


class RecEval(object):

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys
        
    def get_match_rate_and_rms(self):
        def process_one(pred, gt):
            # 未通过组成有效性
            if not pred.comp_valid:
                return None, np.array([None]*3), np.array([None]*3), "comp_invalid", pred
            # 晶体构建失败
            if not getattr(pred, "constructed", False):
                reason = f"construction_failed:{getattr(pred, 'invalid_reason', 'unknown')}"
                return None, np.array([None]*3), np.array([None]*3), reason, pred
            # 结构物理有效性未通过
            if not pred.struct_valid:
                return None, np.array([None]*3), np.array([None]*3), "struct_invalid", pred
            # 新增：与 GT 成分是否一致（元素种类与化学计量都需一致）
            if (getattr(pred, "elems", None) != getattr(gt, "elems", None)) or (getattr(pred, "comps", None) != getattr(gt, "comps", None)):
                return None, np.array([None]*3), np.array([None]*3), "comp_mismatch_to_gt", pred
            # 进入匹配过程
            try:
                rms_dist = self.matcher.get_rms_dist(pred.structure, gt.structure)
                if rms_dist is None:
                    return None, np.array([None]*3), np.array([None]*3), "no_match_within_tolerances", pred
                rms_dist = None if rms_dist is None else rms_dist[0]
                pred_lat = pred.structure.lattice
                gt_lat = gt.structure.lattice
                rms_length = np.array(pred_lat.lengths) - np.array(gt_lat.lengths)
                rms_angle = np.array(pred_lat.angles) - np.array(gt_lat.angles)
                return rms_dist, rms_length, rms_angle, None, pred
            except Exception as e:
                return None, np.array([None]*3), np.array([None]*3), f"matcher_exception:{type(e).__name__}", pred

        rms_dists, rms_lengths, rms_angles = [], [], []
        unmatched_details = []
        reason_counts = {}
        for i in tqdm(range(len(self.preds))):
            d, l, a, reason, pred_obj = process_one(self.preds[i], self.gts[i])
            rms_dists.append(d), rms_lengths.append(l), rms_angles.append(a)
            if reason is not None:
                unmatched_details.append({
                    "index": i,
                    "reason": reason,
                    "pred_comp_valid": bool(getattr(pred_obj, "comp_valid", False)),
                    "pred_struct_valid": bool(getattr(pred_obj, "struct_valid", False)),
                    "constructed": bool(getattr(pred_obj, "constructed", False)),
                    "invalid_reason": getattr(pred_obj, "invalid_reason", None) if not getattr(pred_obj, "constructed", False) else None
                })
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        comp_valid = np.array([c.comp_valid for c in self.preds]).mean()
        struct_valid = np.array([c.struct_valid for c in self.preds]).mean()
        rms_dists = np.array(rms_dists)
        rms_lengths = np.concatenate(rms_lengths)
        rms_angles = np.concatenate(rms_angles)
        match_rate = sum(rms_dists != None) / len(self.preds)
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        mean_rms_lengths = np.sqrt((rms_lengths[rms_lengths != None]**2).mean())
        mean_rms_angles = np.sqrt((rms_angles[rms_angles != None]**2).mean())
        # 组成计数偏差：对每个晶体，统计预测与GT在元素计数上的绝对误差，并对批次取平均
        def comp_count_mae(pred, gt):
            pred_map = {int(e): int(c) for e, c in zip(getattr(pred, 'elems', []), getattr(pred, 'comps', []))}
            gt_map = {int(e): int(c) for e, c in zip(getattr(gt, 'elems', []), getattr(gt, 'comps', []))}
            all_keys = set(pred_map.keys()) | set(gt_map.keys())
            if len(all_keys) == 0:
                return 0.0
            err = []
            for k in all_keys:
                pv = pred_map.get(k, 0)
                gv = gt_map.get(k, 0)
                err.append(abs(pv - gv))
            return float(np.mean(err))
        comp_mae_list = [comp_count_mae(self.preds[i], self.gts[i]) for i in range(len(self.preds))]
        mean_comp_count_mae = float(np.mean(comp_mae_list))
        return {
            'match_rate': match_rate,
            'rms_dist': mean_rms_dist,
            'rms_lengths': mean_rms_lengths,
            'rms_angles': mean_rms_angles,
            'comp_valid': comp_valid,
            'struct_valid': struct_valid,
            # 新增：未匹配样本详情与原因计数
            'recon_unmatched_details': unmatched_details,
            'recon_unmatched_reason_counts': reason_counts
            , 'comp_count_mae': mean_comp_count_mae
        }

    def get_metrics(self):
        return self.get_match_rate_and_rms()


class GenEval(object):

    def __init__(self, pred_crys, gt_crys, n_samples=1000, eval_model_name=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name

        valid_crys = [c for c in pred_crys if c.valid]
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}

    def get_comp_diversity(self):
        comp_fps = [c.comp_fp for c in self.valid_samples]
        comp_fps = CompScaler.transform(comp_fps)
        comp_div = get_fp_pdist(comp_fps)
        return {'comp_div': comp_div}

    def get_struct_diversity(self):
        return {'struct_div': get_fp_pdist([c.struct_fp for c in self.valid_samples])}

    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}

    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_prop_wdist(self):
        if self.eval_model_name is not None:
            pred_props = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in self.valid_samples])
            gt_props = prop_model_eval(self.eval_model_name, [
                                       c.dict for c in self.gt_crys])
            wdist_prop = wasserstein_distance(pred_props, gt_props)
            return {'wdist_prop': wdist_prop}
        else:
            return {'wdist_prop': None}
#         return {'wdist_prob': 0.}

    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.crys, self.gt_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        metrics.update(self.get_comp_diversity())
        metrics.update(self.get_struct_diversity())
        metrics.update(self.get_density_wdist())
        metrics.update(self.get_num_elem_wdist())
        metrics.update(self.get_prop_wdist())
        print(metrics)
        metrics.update(self.get_coverage())
        return metrics


class OptEval(object):

    def __init__(self, crys, num_opt=100, eval_model_name=None):
        """
        crys is a list of length (<step_opt> * <num_opt>),
        where <num_opt> is the number of different initialization for optimizing crystals,
        and <step_opt> is the number of saved crystals for each intialzation.
        default to minimize the property.
        """
        step_opt = int(len(crys) / num_opt)
        self.crys = crys
        self.step_opt = step_opt
        self.num_opt = num_opt
        self.eval_model_name = eval_model_name

    def get_success_rate(self):
        valid_indices = np.array([c.valid for c in self.crys])
        valid_indices = valid_indices.reshape(self.step_opt, self.num_opt)
        valid_x, valid_y = valid_indices.nonzero()
        props = np.ones([self.step_opt, self.num_opt]) * np.inf
        valid_crys = [c for c in self.crys if c.valid]
        if len(valid_crys) == 0:
            sr_5, sr_10, sr_15 = 0, 0, 0
        else:
            pred_props = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in valid_crys])
            percentiles = Percentiles[self.eval_model_name]
            props[valid_x, valid_y] = pred_props
            best_props = props.min(axis=0)
            sr_5 = (best_props <= percentiles[0]).mean()
            sr_10 = (best_props <= percentiles[1]).mean()
            sr_15 = (best_props <= percentiles[2]).mean()
        return {'SR5': sr_5, 'SR10': sr_10, 'SR15': sr_15}

    def get_metrics(self):
        return self.get_success_rate()


class ClassEval(object):

    def __init__(self, crys, model_path=None, num_classes=0):

        self.crys = crys
        self.model_path = Path(model_path)
        self.num_classes = num_classes

    def get_success_rate(self):
        correct_class = np.zeros((self.num_classes, ))
        crys = [{"index":i, "composition":str(c.structure.composition).replace(" ", "")} for i, c in enumerate(self.crys) if c.valid]
        valid_crys = [c for c in self.crys if c.valid]
        print('valid=', len(valid_crys), 'total = ', len(self.crys), ' % = ', len(valid_crys)/len(self.crys))
        if len(valid_crys) > 0:
            pred_class = class_model_eval(UNCOND_CDVAE, CLASSIFIER_CDVAE, self.model_path, [c.dict for c in valid_crys], self.num_classes)
            print(pred_class)
            for i, x in enumerate(pred_class):
                correct_class[x] +=1
                crys[i]["class"] = x
            correct_class/=len(valid_crys)
        return {
            "valid_crys":len(valid_crys), "total_crys":len(self.crys), 
            "crys":crys,
            "percent_valid_crys":len(valid_crys)/len(self.crys), "class":{i:correct_class[i] for i in range(self.num_classes)}
        }

    def get_metrics(self):
        return self.get_success_rate()


def get_file_paths(root_path, task, label='', suffix='pt'):
    if args.label == '':
        out_name = f'eval_{task}.{suffix}'
    else:
        out_name = f'eval_{task}_{label}.{suffix}'
    out_name = os.path.join(root_path, out_name)
    return out_name


def get_crystal_array_list(file_path, batch_idx=0):
    data = load_data(file_path)
    crys_array_list = get_crystals_list(
        data['frac_coords'][batch_idx],
        data['atom_types'][batch_idx],
        data['lengths'][batch_idx],
        data['angles'][batch_idx],
        data['num_atoms'][batch_idx])

    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch['frac_coords'], batch['atom_types'], batch['lengths'],
                batch['angles'], batch['num_atoms'])
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types, batch.lengths,
                batch.angles, batch.num_atoms)
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list


def main(args):
    all_metrics = {}

    cfg = load_config(Path(args.root_path))
    eval_model_name = cfg.data.eval_model_name

    if 'recon' in args.tasks:
        recon_file_path = get_file_paths(args.root_path, 'recon', args.label)
        crys_array_list, true_crystal_array_list = get_crystal_array_list(
            recon_file_path)
        print("Get predicted structures")
        pred_crys = p_map(lambda x: Crystal(x, True), crys_array_list)

        print("\nGet ground-truth structures")
        gt_crys = p_map(lambda x: Crystal(x, False), true_crystal_array_list)

        rec_evaluator = RecEval(pred_crys, gt_crys)
        recon_metrics = rec_evaluator.get_metrics()
        all_metrics.update(recon_metrics)

    if 'gen' in args.tasks:
        gen_file_path = get_file_paths(args.root_path, 'gen', args.label)
        recon_file_path = get_file_paths(args.root_path, 'recon', args.label)
        crys_array_list, _ = get_crystal_array_list(gen_file_path)
        #crys_array_list = crys_array_list[:20]
        gen_crys = p_map(lambda x: Crystal(x, True), crys_array_list)
        if 'recon' not in args.tasks:
            _, true_crystal_array_list = get_crystal_array_list(
                recon_file_path)
            #true_crystal_array_list = true_crystal_array_list[:20]
            gt_crys = p_map(lambda x: Crystal(x, False), true_crystal_array_list)

        gen_evaluator = GenEval(
            gen_crys, gt_crys, eval_model_name=eval_model_name, n_samples=1000)
        gen_metrics = gen_evaluator.get_metrics()
        all_metrics.update(gen_metrics)

    if 'opt' in args.tasks:
        opt_file_path = get_file_paths(args.root_path, 'opt', args.label)
        crys_array_list, _ = get_crystal_array_list(opt_file_path)
        opt_crys = p_map(lambda x: Crystal(x), crys_array_list)

        opt_evaluator = OptEval(opt_crys, eval_model_name=eval_model_name)
        opt_metrics = opt_evaluator.get_metrics()
        all_metrics.update(opt_metrics)


    if 'class' in args.tasks:

        opt_file_path = get_file_paths(args.root_path, args.type, args.label)
        crys_array_list, _ = get_crystal_array_list(opt_file_path)

        opt_crys = p_map(lambda x: Crystal(x), crys_array_list)

        opt_evaluator = ClassEval(opt_crys, args.classifier_path, 7)
        opt_metrics = opt_evaluator.get_metrics()
        all_metrics.update(opt_metrics)        

    print(all_metrics)

    if args.label == '':
        metrics_out_file = f'eval_metrics_{args.type}.json'
    else:
        metrics_out_file = f'eval_metrics_{args.type}_{args.label}.json'
    metrics_out_file = os.path.join(args.root_path, metrics_out_file)

    # only overwrite metrics computed in the new run.
    if Path(metrics_out_file).exists():
        with open(metrics_out_file, 'r') as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(all_metrics)
            else:
                with open(metrics_out_file, 'w') as f:
                    json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        if isinstance(written_metrics, dict):
            with open(metrics_out_file, 'w') as f:
                json.dump(written_metrics, f, ensure_ascii=False, indent=2)
    else:
        with open(metrics_out_file, 'w') as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--classifier_path')
    parser.add_argument('--type', default='gen')
    parser.add_argument('--label', default='')
    parser.add_argument('--tasks', nargs='+', default=['recon', 'gen', 'opt'])
    args = parser.parse_args()
    main(args)
