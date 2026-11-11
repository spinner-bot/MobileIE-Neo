"""Secondary historical GPU replay for the exactly retrained Charb spatial baseline."""
import json

from evaluate_aggregation_r20 import metric_delta
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256

BASE=ROOT/'research/4.temporal-consistency'


def main():
    folder=BASE/'results/exploration-r26-eval-plain';cfg=json.loads((folder/'config.json').read_text())
    assert cfg['device']=='cuda:0' and json.loads((folder/'runtime.json').read_text())['rows']==2944
    sources={};checks=[]
    for filename in ['per_sequence.jsonl','per_frame.jsonl']:
        def key(r,step=None,fusion=None):
            return (r['domain'],r['scene'],r['train_seed'],r['step'] if step is None else step,r['input_reset'],r['fusion'] if fusion is None else fusion,
                *((r['frame'],) if filename=='per_frame.jsonl' else ()))
        path=folder/filename;sources[str(path.relative_to(ROOT))]=sha256(path);actual={}
        with path.open() as stream:
            for line in stream:
                r=json.loads(line)
                if r['objective']=='charb':actual[key(r)]=r
        path=BASE/'results/exploration-r14-eval-spatial'/filename;sources[str(path.relative_to(ROOT))]=sha256(path);count=0;maximum=0.
        with path.open() as stream:
            for line in stream:
                r=json.loads(line)
                if r['train_preparation']=='native_crop' and r['variant']=='plain':
                    maximum=max(maximum,metric_delta(actual[key(r)],r));count+=1
        checks.append({'reference':'R14 native plain raw/recursive, both steps','file':filename,'rows':count,'max_abs_metric':maximum})
        count=0;maximum=0.
        for seed in [0,1]:
            path=BASE/'results'/f'exploration-r20-seed{seed}'/filename;sources[str(path.relative_to(ROOT))]=sha256(path)
            with path.open() as stream:
                for line in stream:
                    r=json.loads(line)
                    if r['method'] in ['postwarp01','postwarp05']:
                        maximum=max(maximum,metric_delta(actual[key(r,4000,r['method'])],r));count+=1
        checks.append({'reference':'R20 finite postwarp .1/.5 step4000','file':filename,'rows':count,'max_abs_metric':maximum})
    save_json(BASE/'process/exploration-r26/plain-historical-metric-replay.json',{'checks':checks,
        'max_abs_metric':max(r['max_abs_metric'] for r in checks),'source_sha256':sources,
        'scope':'Secondary same-GPU-backend historical replay; primary loss comparisons all freshly evaluated on the common backend.'})


if __name__=='__main__':main()
