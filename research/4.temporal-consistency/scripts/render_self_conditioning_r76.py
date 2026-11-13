"""Append self4 to all primary contacts without changing any original contact pixels."""
import json
import os
from pathlib import Path
import time
import traceback

import cv2
import matplotlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

from causal_reference_io_r75 import ROOT,BASE,write,digest,stamp
from causal_reference_evaluation_r75 import metas,key
from self_conditioning_evaluation_r76 import require_primary_pipeline,self4_quality
from evaluate_cvvdp_references_r66 import cached
from prepare_cache_r5 import load_one


def display(value):
    return Image.fromarray(np.rint(np.clip(value,0,255)).astype(np.uint8))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise SystemExit('Require an initially empty CUDA mask for CPU contact rendering.')
    torch.set_num_threads(2);cv2.setNumThreads(1)
    require_primary_pipeline()
    _,entries,_=self4_quality();lookup={key(r):r for r in entries}
    primary=BASE/'report/exploration-r75-contacts'
    originals=json.loads((primary/'visual_sources.json').read_text())['contacts']
    assert len(originals)==20
    summary=BASE/'report/exploration-r76-quality/audit.json'
    assert json.loads(summary.read_text())['completed']
    scenes,_=metas();scene_lookup={r['scene']:r for r in scenes}
    out=BASE/'report/exploration-r76-contacts';out.mkdir(exist_ok=False)
    font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),14)
    started,complete,records=time.perf_counter(),False,[]
    try:
        for old in originals:
            seed,scene=old['seed'],old['scene']
            source=ROOT/old['contact'];assert digest(source)==old['sha256']
            with Image.open(source) as original:
                assert original.size==(1600,1468)
                image=Image.new('RGB',(1880,1468),'white')
                image.paste(original,(0,0))
            entry=lookup[seed,scene,'ReBotNet_self4']
            values=cached(entry);data=load_one(scene_lookup[scene])
            assert values.shape==data['gt'].shape
            frames=old['frames'];x,y,width,height=old['native_crop_xywh']
            assert width==height==128 and frames==[0,len(values)//2,len(values)-1]
            draw=ImageDraw.Draw(image);left=1612
            draw.text((left,4),'R76 self4',font=font,fill='black')
            draw.text((left,31),'Same frame / four calls',font=font,fill='black')
            for ordinal,t in enumerate(frames):
                top=60+460*ordinal
                image.paste(display(values[t]).resize((256,144),Image.Resampling.BILINEAR),(left,top+20))
                patch=np.asarray(values[t,y:y+128,x:x+128],dtype=np.float32)
                target=data['gt'][t,y:y+128,x:x+128].astype(np.float32)
                image.paste(display(patch),(left+64,top+184))
                error=np.clip(np.abs(patch-target).mean(axis=-1)/32,0,1)
                heat=np.rint(matplotlib.colormaps['magma'](error)[...,:3]*255).astype(np.uint8)
                image.paste(Image.fromarray(heat),(left+64,top+332))
            draw.text((left,1444),'4x calls / no past frame',font=font,fill='black')
            path=out/f'{scene}--seed{seed}.png';image.save(path)
            records.append(dict(seed=seed,scene=scene,domain=old['domain'],frames=frames,
                native_source_hw=old['native_source_hw'],native_crop_xywh=old['native_crop_xywh'],
                labels=old['labels']+['ReBotNet self4'],primary_contact=old,
                primary_region_preserved=[0,0,1600,1468],self4_source=entry,
                contact=str(path.relative_to(ROOT)),sha256=digest(path),actual_reviewed=False))
            print(json.dumps(dict(seed=seed,scene=scene,contacts=len(records))),flush=True)
            del data,values
        assert len(records)==20 and not torch.cuda.is_initialized();complete=True
        write(out/'visual_sources.json',dict(at=stamp(),contacts=records,primary_files_rewritten=False,
            actual_review_required_separately=True,new_CNN=0,SGD=0,
            scope='Every primary condition retained, fixed display/crop/error scale; no playback or MOS claim.'))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'runtime.json',dict(ended=stamp(),completed_normally=complete,contacts=len(records),
            new_CNN=0,SGD=0,CUDA_initialized=torch.cuda.is_initialized(),seconds=time.perf_counter()-started))


if __name__=='__main__':
    main()
