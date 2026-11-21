"""Owned paths and explicit small-source scope for L1; importing needs no Torch."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / "research/4.temporal-consistency/scripts"))
from causal_reference_io_r75 import append, digest, write


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def identity():
    values = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=os.getpid(), start_ticks=int(values[19]), state=values[0])


def verify():
    freeze = json.loads((BASE / "process/preflight-source-freeze001.json").read_text())
    assert all(digest(ROOT / p) == h for p, h in freeze["source_sha256"].items())
    return freeze
