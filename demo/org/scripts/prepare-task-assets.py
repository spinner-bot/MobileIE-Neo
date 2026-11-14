#!/usr/bin/env python3
"""Package verified task models and attributed RGB / lossless real RAW samples."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import numpy as np
from PIL import Image,ImageOps

REPO=Path(__file__).resolve().parents[3]
ASSETS=REPO/'demo/org/app/app/src/main/assets'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',type=Path,required=True);p.add_argument('--models',type=Path,required=True);a=p.parse_args()
    conversion=json.loads((a.models/'conversion-report.json').read_text());assert len(conversion)==2 and all(x['all_pass'] and x['input_count']==1 and not x['native_control_inputs'] for x in conversion)
    for task in ('underwater','isp'):
        (ASSETS/'samples'/task).mkdir(parents=True,exist_ok=True)
        shutil.copyfile(a.models/(task+'.tflite'),ASSETS/('model_'+task+'.tflite'))
    provenance=[]
    for i,name in enumerate(('0','1','10'),1):
        source=a.sources/f'uieb-{i-1}.png';target=ASSETS/f'samples/underwater/{i:02d}.png'
        ImageOps.fit(Image.open(source).convert('RGB'),(600,400),Image.Resampling.BILINEAR).save(target)
        provenance.append({'asset':str(target.relative_to(ASSETS)),'dataset':'UIEB','source':'https://huggingface.co/datasets/Hikari0608/UIEB','split':'val','row_index':i-1,'source_sha256':sha(source),'asset_sha256':sha(target),'transform':'center 3:2 crop and bilinear resize 600x400','use':'academic demonstration; retain dataset authors attribution'})
        source=a.sources/'zrr'/(name+'.png');raw=np.asarray(Image.open(source),dtype='<u2');assert raw.shape==(448,448) and raw.max()<=1023
        target=ASSETS/f'samples/isp/{i:02d}.rggb';target.write_bytes(struct.pack('<8s5I',b'MIE_RAW1',448,448,4095,64,1023)+raw.tobytes())
        provenance.append({'asset':str(target.relative_to(ASSETS)),'dataset':'Zurich RAW-to-RGB','author_page':'https://aiff22.github.io/pynet.html','archive_member':f'test/huawei_raw/{name}.png','source_sha256':sha(source),'asset_sha256':sha(target),'transform':'lossless uint16 little-endian container; no resize, value edits or generated RAW','sensor':'Huawei P20 RGGB, 10-bit in uint16; official MobileIE divisor 4095'})
    tasks=[]
    for task,title,model,shape,out,baseline,label,paths in [
        ('lowlight','低光增强','model.tflite',[1,400,600,3],[1,400,600,3],'gamma_05','Gamma 0.5',[f'samples/{i:02d}.png' for i in range(1,4)]),
        ('underwater','水下增强','model_underwater.tflite',[1,400,600,3],[1,400,600,3],'gray_world','灰度世界白平衡',[f'samples/underwater/{i:02d}.png' for i in range(1,4)]),
        ('isp','RAW→RGB ISP','model_isp.tflite',[1,224,224,4],[1,448,448,3],'bilinear_isp','双线性传统 ISP',[f'samples/isp/{i:02d}.rggb' for i in range(1,4)])]:
        tasks.append({'id':task,'name':title,'model_asset':model,'model_sha256':sha(ASSETS/model),'input_shape':shape,'output_shape':out,'native_control_inputs':[],'baseline_id':baseline,'baseline_name':label,'sample_assets':paths})
    (ASSETS/'tasks.json').write_text(json.dumps({'schema_version':1,'tasks':tasks},ensure_ascii=False,indent=2)+'\n')
    provenance_text=json.dumps({'samples':provenance,'models':conversion,'RAW_format':'28-byte <8s5I header: MIE_RAW1,width,height,normalization_divisor,black_level,ADC_white; then row-major uint16 LE. Fixed supported profile 448x448 RGGB, 4095/64/1023. No black subtraction (same as upstream).'},ensure_ascii=False,indent=2)+'\n'
    (REPO/'demo/org/app/task-asset-sources.json').write_text(provenance_text)
    (ASSETS/'task-asset-sources.json').write_text(provenance_text)
    print(json.dumps(tasks,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
