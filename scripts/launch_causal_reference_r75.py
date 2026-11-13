"""Detach the two registered reference seeds; each wrapper independently admits its GPU."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess

from project_assets import mounted_data

ROOT = Path(__file__).resolve().parents[1]
PROCESS = ROOT/'research/4.temporal-consistency/process/exploration-r75'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch',type=int,default=1)
    args=parser.parse_args()
    if args.launch != 1:
        raise SystemExit('Only original registered launch supported; resource recovery is explicit.')
    mounted_data()
    path=PROCESS/'launch001.json'
    if path.exists():
        raise SystemExit('Launch record exists; inspect real PIDs instead of launching duplicates.')
    gate=json.loads((PROCESS/'cpu-gate001/gate.json').read_text())
    assert gate['passed']
    record=dict(at=datetime.now().astimezone().isoformat(),seeds=[],
                scope='Detached original formal launch; no auto retries, no implied admission or completed training.')
    for seed,gpu in [(0,0),(1,1)]:
        parent=PROCESS/f'launch001-seed{seed}-parent.json'
        log=PROCESS/f'launch001-seed{seed}.log'
        assert not parent.exists() and not log.exists()
        command=[str(ROOT/'scripts/project-python'),str(ROOT/'scripts/prebind_shared_gpu.py'),
                 '--gpu',str(gpu),'--record',str(parent),'--',str(ROOT/'scripts/project-python'),
                 str(ROOT/'research/4.temporal-consistency/scripts/train_causal_reference_r75.py'),
                 '--seed',str(seed),'--attempt','1']
        with log.open('x') as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        record['seeds'].append(dict(seed=seed,requested_GPU=gpu,parent_pid=child.pid,
            parent_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],
            command=command,parent_record=str(parent.relative_to(ROOT)),stdout=str(log.relative_to(ROOT))))
        path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
