"""Frozen complete R80 readout experiment; original qualified modules stay unchanged."""
import json

from causal_reference_io_r75 import ROOT, BASE, digest, write, append, lines, stamp
from noise_role_io_r79 import acquire_gpu, owned_cache

PROCESS = BASE/'process/exploration-r80'
PROTOCOL = BASE/'docs/exploration-r80-coefficient-state-protocol.md'
NEW_SOURCES = ['research/4.temporal-consistency/scripts/'+name for name in (
    'coefficient_io_r80.py', 'coefficient_state_r80.py', 'evaluate_coefficient_state_r80.py',
    'evaluate_cvvdp_coefficient_r80.py', 'summarize_coefficient_state_r80.py',
    'render_coefficient_state_r80.py', 'finalize_coefficient_state_r80.py',
    'causal_reference_evaluation_r75.py', 'prepare_cache_r5.py', 'cvvdp_adapter_r66.py',
    'evaluate_cvvdp_references_r66.py', 'exploration_r2_data.py', 'noise_role_io_r79.py')]
NEW_SOURCES += ['scripts/'+name for name in ('prebind_shared_gpu.py', 'shared_gpu_admission.py',
    'shared_training_schedule.py', 'bound_gpu_observer.py')]
LABELS = ('FST_prefix', 'RGB_prefix', 'FST_GT_diagnostic')
SCOPES = ('full', 'tail')


def verify():
    gate = json.loads((PROCESS/'cpu-gate001/gate.json').read_text())
    config = json.loads((PROCESS/'cpu-gate001/config.json').read_text())
    assert gate['passed'] and gate['discarded_ridge_updates'] == 32 and not gate['CUDA_initialized']
    assert digest(PROTOCOL) == config['protocol_sha256']
    assert all(digest(ROOT/path) == sha for path, sha in config['source_sha256'].items())
    expected = json.loads((PROCESS/'source-freeze001.json').read_text())
    actual = {path: digest(ROOT/path) for path in set(NEW_SOURCES) | set(config['source_sha256'])}
    assert expected['source_sha256'] == actual and expected['protocol_sha256'] == digest(PROTOCOL)
    return expected


def key(row):
    return row['seed'], row['scene'], row['label'], row['scope']
