"""Frozen latent conditional moments from the existing static paired-observation cache."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch

from continuation_r14 import create_model
from correlated_observations_r19 import encode
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256

BASE = ROOT/'research/4.temporal-consistency'


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def measures(value):
    return {'signed_mean': float(value.mean()), 'mean_abs': float(value.abs().mean()),
            'rms': float(value.square().mean().sqrt())}


@torch.inference_mode()
def features(net, inputs):
    result = {'head': [], 'body': []}
    for offset in range(0, len(inputs), 8):
        x = inputs[offset:offset+8]
        result['head'].append(net.head.block1(x))
        result['body'].append(net.body.block1(net.head(x)))
    return {k: torch.cat(v) for k, v in result.items()}


def decompose(u, v, reference, a, bias):
    # These FP64 polynomials define moment identities, not FP32 full-network outputs.
    u, v, reference = [x[..., 8:-8, 8:-8].double() for x in [u, v, reference]]
    mu, mv = u.mean(0, keepdim=True), v.mean(0, keepdim=True)
    eu, ev = u-mu, v-mv
    vu, vv, covariance = eu.square().mean(0, keepdim=True), ev.square().mean(0, keepdim=True), (eu*ev).mean(0, keepdim=True)
    clean_q = a*reference.square()+bias
    self_mean = a*(mu.square()-reference.square())
    self_variance = a*vu
    cross_mean = a*(mu*mv-reference.square())
    cross_covariance = a*covariance
    empirical = {'self': (a*u.square()+bias).mean(0, keepdim=True)-clean_q,
                 'cross': (a*u*v+bias).mean(0, keepdim=True)-clean_q,
                 'postmean': (.5*((a*u.square()+bias)+(a*v.square()+bias))).mean(0, keepdim=True)-clean_q,
                 'pre_square': (a*(.5*(u+v)).square()+bias).mean(0, keepdim=True)-clean_q}
    formula = {'self': self_mean+self_variance, 'cross': cross_mean+cross_covariance,
               'postmean': a*(.5*(mu.square()+mv.square())-reference.square()+.5*(vu+vv)),
               'pre_square': a*((.5*(mu+mv)).square()-reference.square()+.25*(vu+vv+2*covariance))}
    scale = max(float(clean_q.abs().max()), *(float(x.abs().max()) for x in empirical.values()), 1e-30)
    closures = {k: float((empirical[k]-formula[k]).abs().max())/scale for k in empirical}
    disagreement = (u-v).square().mean(0, keepdim=True)
    closures['postmean_minus_cross'] = float((empirical['postmean']-empirical['cross']-a/2*disagreement).abs().max())/scale
    closures['pre_square_minus_cross'] = float((empirical['pre_square']-empirical['cross']-a/4*disagreement).abs().max())/scale
    assert max(closures.values()) <= 1e-10
    return {'coefficient_a': float(a), 'reference_q_rms': float(clean_q.square().mean().sqrt()),
            'feature_current_variance': float(vu.mean()), 'feature_previous_variance': float(vv.mean()),
            'feature_paired_covariance': float(covariance.mean()),
            'feature_empirical_correlation': float(covariance.mean()/torch.sqrt(vu.mean()*vv.mean()).clamp_min(1e-30)),
            'feature_current_mean_shift': measures(mu-reference),
            'self_mean_shift_term': measures(self_mean), 'self_variance_term': measures(self_variance),
            'cross_mean_shift_term': measures(cross_mean), 'cross_covariance_term': measures(cross_covariance),
            'empirical_latent_bias': {k: measures(x) for k, x in empirical.items()}, 'closure_scaled': closures}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    weights_path = BASE/'results/exploration-r27-frozen-cpu/checkpoints.json'
    weights = json.loads(weights_path.read_text())
    index = BASE/'results/exploration-r19/cache_index.json'
    conditions = json.loads(index.read_text())['conditions']
    assert len(weights) == 5 and len(conditions) == 16
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in Path(__file__).resolve().parents[1].glob('*.py')}
    sources[str(Path(__file__).relative_to(ROOT))] = sha256(Path(__file__))
    save_json(out/'config.json', {'device': 'cpu', 'threads': 2, 'batch': 8, 'draws': 64, 'margin': 8,
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_sha256': sources, 'protocol_sha256': sha256(BASE/'docs/exploration-r29-protocol.md'),
              'cache_index_sha256': sha256(index), 'weights_index_sha256': sha256(weights_path),
              'scope': 'FP32 frozen latent extraction, FP64 population moments and polynomial identities; no RGB quality or training claim.'})
    models = []
    for record in weights:
        path = ROOT/record['path']
        assert sha256(path) == record['sha256']
        net = create_model('plain', 'cpu').eval()
        if record['label'] != 'official':
            saved = torch.load(path, map_location='cpu', weights_only=False)
            assert saved['step'] == 4000
            net.load_state_dict(saved['model'], strict=True)
        models.append((record, net.backbone.eval()))
    current_hashes, clean_cache, replays, cache_audit = {}, {}, [], []
    start = time.perf_counter()
    count = 0
    for condition in conditions:
        path = ROOT/condition['path']
        assert sha256(path) == condition['sha256']
        cache_audit.append({'path': condition['path'], 'sha256': condition['sha256']})
        with np.load(path, allow_pickle=False) as data:
            assert array_hash(data['current']) == condition['current_array_sha256']
            assert array_hash(data['previous']) == condition['previous_array_sha256']
            x = torch.from_numpy(data['current'].transpose(0, 3, 1, 2).copy())
            y = torch.from_numpy(data['previous'].transpose(0, 3, 1, 2).copy())
            noiseless_sensor = data['target'].astype(np.float64)**2.2*.35
            clean = torch.from_numpy(encode(noiseless_sensor[None], condition['jpeg']).transpose(0, 3, 1, 2).copy())
        assert x.shape == y.shape == (64, 3, 128, 128)
        identity = {k: condition[k] for k in ['scene', 'photons', 'rho', 'jpeg']}
        for record, net in models:
            current, past = features(net, x), features(net, y)
            clean_key = (condition['scene'], condition['jpeg'], record['label'])
            if clean_key not in clean_cache:
                clean_cache[clean_key] = features(net, clean)
            for stage in ['head', 'body']:
                u, v = current[stage], past[stage]
                digest = array_hash(u.numpy())
                key = (condition['scene'], condition['photons'], condition['jpeg'], record['label'], stage)
                if key in current_hashes:
                    assert digest == current_hashes[key]
                    replays.append({'key': list(key), 'current_feature_sha256_exact': True})
                current_hashes[key] = digest
                module = getattr(net, stage)
                a = module.weight1.detach().double()*module.weight2.detach().double()
                row = {**identity, 'weight': record['label'], 'objective': record['objective'], 'train_seed': record['train_seed'],
                       'stage': stage, 'current_feature_sha256': digest, 'previous_feature_sha256': array_hash(v.numpy()),
                       **decompose(u, v, clean_cache[clean_key][stage], a, module.bias.detach().double())}
                with (out/'moments.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                count += 1
        print(json.dumps({**identity, 'rows': count, 'current_feature_replays': len(replays)}), flush=True)
    assert count == 160 and len(replays) == 80 and len(clean_cache)*2 == 40
    assert not torch.cuda.is_initialized()
    save_json(out/'current_replays.json', replays)
    save_json(out/'cache_audit.json', cache_audit)
    save_json(out/'runtime.json', {'rows': count, 'current_feature_replays': len(replays), 'clean_node_cache_entries': len(clean_cache)*2,
              'training_updates': 0, 'cuda_initialized': False, 'wall_seconds': time.perf_counter()-start,
              'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})


if __name__ == '__main__':
    main()
