"""Owned A1 paths and named source verification; no model is loaded."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
SOURCE = BASE.parent / "latent-core"
sys.path.insert(0, str(ROOT / "research/4.temporal-consistency/scripts"))
from causal_reference_io_r75 import append, digest, write


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def identity():
    values = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=os.getpid(), start_ticks=int(values[19]), state=values[0])


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def owned_cache():
    path = ROOT / ".cache/experiments/topic8-affine-cache/screen001"
    assert path.resolve().is_relative_to("/data/Users/zrh/data/MobileIE-Neo")
    return path


def verify():
    frozen = read(BASE / "process/source-freeze001.json")
    assert all(digest(ROOT / p) == h for p, h in frozen["source_sha256"].items())
    return frozen
