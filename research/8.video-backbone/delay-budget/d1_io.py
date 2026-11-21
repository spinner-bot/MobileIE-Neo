"""Owned D1 paths and named-source verification without loading any model."""
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
sys.path.insert(0,str(ROOT/'research/4.temporal-consistency/scripts'))
from causal_reference_io_r75 import append,digest,write
ARMS=('F0','F0_FB','F0_DIS','past3','center3','center_self')
AUDIT=BASE.parent/'evidence-reconciliation/process/delay-source-audit001.json'


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def identity(pid=None):
    pid=os.getpid() if pid is None else pid
    values=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(values[19]),state=values[0])


def read(path):
    return json.loads(Path(path).read_text())


def owned(path):
    path=Path(path)
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    return path


def cache():
    return owned(ROOT/'.cache/experiments/topic8-delay-budget/eval001')


def verify():
    frozen=read(BASE/'process/source-freeze001.json')
    assert all(digest(ROOT/p)==h for p,h in frozen['source_sha256'].items())
    return frozen
