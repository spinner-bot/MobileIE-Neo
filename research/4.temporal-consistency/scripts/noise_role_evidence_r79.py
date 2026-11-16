"""Frozen final-only evidence dependencies, separate from running training sources."""
import json

from noise_role_io_r79 import ROOT, BASE, PROCESS, digest

EVALUATION_SOURCES = ['research/4.temporal-consistency/scripts/'+name for name in (
    'noise_role_evidence_r79.py', 'evaluate_noise_role_r79.py', 'evaluate_cvvdp_noise_role_r79.py',
    'summarize_noise_role_r79.py', 'render_noise_role_r79.py', 'finalize_noise_role_r79.py',
    'causal_reference_evaluation_r75.py', 'cvvdp_adapter_r66.py', 'prepare_cache_r5.py',
    'evaluate_cvvdp_references_r66.py', 'exploration_r2_data.py')]


def verify_evaluation():
    value = json.loads((PROCESS/'evaluation-freeze001.json').read_text())
    assert value['source_sha256'] == {name: digest(ROOT/name) for name in EVALUATION_SOURCES}
    return value
