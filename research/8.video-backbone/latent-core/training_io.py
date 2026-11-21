"""L1 formal source verification and owned runtime assets; no Torch at import."""
import json
from l1_io import BASE, ROOT, append, digest, identity, stamp, verify, write


def read(path):
    return json.loads(path.read_text())


def verify_training():
    verify()
    frozen = read(BASE / "process/training-source-freeze001.json")
    assert all(digest(ROOT / p) == h for p, h in frozen["source_sha256"].items())
    return frozen


def owned_cache(name):
    path = ROOT / ".cache/experiments/topic8-latent-core" / name
    assert path.resolve().is_relative_to("/data/Users/zrh/data/MobileIE-Neo")
    return path
