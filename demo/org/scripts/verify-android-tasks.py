#!/usr/bin/env python3
"""Compare actual Android task outputs against independent CPU references."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'demo/org/desktop'))
import lightlab_tasks as tasks


def max_pixels(a,b):return int(np.abs(np.asarray(a,dtype=int)-np.asarray(b,dtype=int)).max())
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--android-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    refs=Path('/data/Users/zrh/data/MobileIE-Neo/demo/org/three-task/models-v2');records=[]
    for task_id,task in tasks.TASKS.items():
        for index,asset in enumerate(task['sample_assets'],1):
            name=f'{task_id}-sample{index}';prepared=tasks.prepare(tasks.ASSETS/asset,task_id)
            if task_id=='lowlight':_,expected,_=tasks.Engine(task_id).run(prepared)
            else:expected=np.load(refs/(name+'-reference.npy'))
            actual=np.fromfile(a.android_dir/(name+'-android.f32'),dtype='<f4').reshape(task['output_shape'])
            delta=float(np.max(np.abs(actual.astype(float)-expected.astype(float))));finite=bool(np.isfinite(actual).all())
            reference_image=(np.clip(expected[0],0,1)*255).astype(np.uint8);image=Image.open(a.android_dir/(name+'-android.png')).convert('RGB');pixel=max_pixels(image,reference_image)
            baseline=max_pixels(Image.open(a.android_dir/(name+'-baseline.png')).convert('RGB'),tasks.baseline(prepared))
            original=max_pixels(Image.open(a.android_dir/(name+'-input.png')).convert('RGB'),prepared.image)
            raw_input_matches=True
            if task_id=='isp':raw_input_matches=bool(np.array_equal(np.fromfile(a.android_dir/(name+'-input.f32'),dtype='<f4').reshape(task['input_shape']),prepared.tensor))
            records.append({'task_id':task_id,'sample':index,'raw_max_abs':delta,'uint8_max_abs':pixel,'baseline_max_abs':baseline,'original_max_abs':original,'raw_input_exact':raw_input_matches,'all_finite':finite,'pass':delta<=1e-4 and pixel<=1 and baseline<=1 and original==0 and raw_input_matches and finite})
    report={'all_pass':all(c['pass'] for c in records),'cases':records,'gates':{'raw_max_abs':1e-4,'uint8_max_abs':1,'baseline_uint8_max_abs':1,'input_exact':True},'reference':'official PyTorch CPU for UIE/ISP; original published TFLite CPU for lowlight; independently validated fixed baselines'}
    (a.output/'task-numerical-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert report['all_pass'],'Task numerical gate failed'


if __name__=='__main__':main()
