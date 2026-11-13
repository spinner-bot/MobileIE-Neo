"""All20 fixed contacts: entire cached FOV, native128 crops, and fixed-scale errors."""
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

from causal_reference_io_r75 import ROOT,BASE,write,lines,digest,stamp
from causal_reference_evaluation_r75 import metas,reference_predictions,final_markers
from evaluate_cvvdp_references_r66 import cached
from prepare_cache_r5 import load_one


def display(value):
    return Image.fromarray(np.rint(np.clip(value,0,255)).astype(np.uint8))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise SystemExit('Renderer requires an initially empty CUDA mask.')
    torch.set_num_threads(2);cv2.setNumThreads(1)
    final_markers()
    quality=BASE/'report/exploration-r75-quality/audit.json'
    assert json.loads(quality.read_text())['completed']
    entries=[]
    for seed in (0,1):
        run=BASE/f'results/exploration-r75-eval-seed{seed}'
        assert json.loads((run/'runtime.json').read_text())['completed_normally']
        entries+=lines(run/'prediction_manifest.jsonl')
    references=reference_predictions();scenes,_=metas()
    out=BASE/'report/exploration-r75-contacts';out.mkdir(exist_ok=False)
    font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),14)
    started,complete,records=time.perf_counter(),False,[]
    try:
        for seed in (0,1):
            for meta in scenes:
                data=load_one(meta)
                new=[next(r for r in entries if r['seed']==seed and r['scene']==meta['scene'] and r['history']==route)
                     for route in ('causal','current')]
                old=[next(r for r in references if r['seed']==seed and r['scene']==meta['scene'] and r['label']==label)
                     for label in ('MobileIE_raw','MobileIE_recursive_self05')]
                arrays=[data['low'],data['gt']]+[cached(r) for r in old+new]
                assert all(a.shape==data['gt'].shape for a in arrays)
                frames=[0,len(data['gt'])//2,len(data['gt'])-1]
                height,width=data['gt'].shape[1:3]
                x,y=(width-128)//2,(height-128)//2
                labels=['Input','GT','MobileIE raw','MobileIE recursive .5','ReBotNet causal','ReBotNet current']
                image=Image.new('RGB',(1600,1468),'white');draw=ImageDraw.Draw(image)
                extent='ND native crop' if meta['domain']=='paired_real_ND_DEV' else 'SYN cached FOV'
                draw.text((12,4),f'R75 final64K / {meta["scene"]} / seed{seed} / {extent} {width}x{height} / fixed frames {frames}',font=font,fill='black')
                for col,label in enumerate(labels):
                    draw.text((12+264*col,31),label,font=font,fill='black')
                for ordinal,t in enumerate(frames):
                    top=60+460*ordinal
                    draw.text((12,top),f'Frame {t}: entire cached view (display resize only)',font=font,fill='black')
                    draw.text((12,top+164),f'Native128: x={x}, y={y} (no resize)',font=font,fill='black')
                    draw.text((12,top+312),'Absolute RGB mean error: common 0..32/255 display scale',font=font,fill='black')
                    target=data['gt'][t,y:y+128,x:x+128].astype(np.float32)
                    for col,array in enumerate(arrays):
                        left=12+264*col
                        image.paste(display(array[t]).resize((256,144),Image.Resampling.BILINEAR),(left,top+20))
                        patch=np.asarray(array[t,y:y+128,x:x+128],dtype=np.float32)
                        image.paste(display(patch),(left+64,top+184))
                        error=np.clip(np.abs(patch-target).mean(axis=-1)/32,0,1)
                        heat=np.rint(matplotlib.colormaps['magma'](error)[...,:3]*255).astype(np.uint8)
                        image.paste(Image.fromarray(heat),(left+64,top+332))
                draw.text((12,1444),'Fixed static contacts are not full video playback or human MOS. ND is a cached crop, not full1080p.',font=font,fill='black')
                path=out/f'{meta["scene"]}--seed{seed}.png';image.save(path)
                records.append(dict(seed=seed,scene=meta['scene'],domain=meta['domain'],frames=frames,
                    native_source_hw=[height,width],native_crop_xywh=[x,y,128,128],labels=labels,
                    contact=str(path.relative_to(ROOT)),sha256=digest(path),sources=old+new,
                    actual_reviewed=False,new_CNN=0))
                print(json.dumps(dict(seed=seed,scene=meta['scene'],contacts=len(records))),flush=True)
        assert len(records)==20 and not torch.cuda.is_initialized();complete=True
        write(out/'visual_sources.json',dict(at=stamp(),contacts=records,actual_review_required_separately=True,
             new_CNN=0,SGD=0,scope='All fixed conditions/seeds; no contact chosen from candidate errors.'))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'runtime.json',dict(ended=stamp(),completed_normally=complete,contacts=len(records),
            new_CNN=0,SGD=0,CUDA_initialized=torch.cuda.is_initialized(),seconds=time.perf_counter()-started))


if __name__=='__main__':
    main()
