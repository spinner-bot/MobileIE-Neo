"""Three frozen task contracts; no adjustable model or processing controls."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import struct
import time

import numpy as np
from PIL import Image,ImageOps,ImageCms

HERE=Path(__file__).resolve().parent
ASSETS=HERE/'assets' if (HERE/'assets').is_dir() else HERE.parent/'app/app/src/main/assets'
TASKS={t['id']:t for t in json.loads((ASSETS/'tasks.json').read_text(encoding='utf-8'))['tasks']}
GAMMA_LUT=[int(255*(i/255)**.5) for i in range(256)]
RAW_HEADER=struct.Struct('<8s5I')


@dataclass
class Prepared:
    task_id:str
    image:Image.Image
    tensor:np.ndarray
    raw:np.ndarray|None=None
    original_bytes:bytes|None=None


def prepare_rgb(path:Path)->Image.Image:
    if path.stat().st_size>32*1024*1024:raise ValueError('图片超过 32 MB，请选择较小的照片')
    with Image.open(path) as image:
        if max(image.size)>50000:raise ValueError('图片尺寸过大，请选择较小的照片')
        image.draft('RGB',(2048,2048));corrected=ImageOps.exif_transpose(image);profile=image.info.get('icc_profile')
        if corrected.mode in ('RGBA','LA') or 'transparency' in corrected.info:
            rgba=corrected.convert('RGBA');corrected=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba).convert('RGB')
        else:corrected=corrected.convert('RGB')
        if profile:corrected=ImageCms.profileToProfile(corrected,ImageCms.ImageCmsProfile(io.BytesIO(profile)),ImageCms.createProfile('sRGB'),outputMode='RGB')
        corrected.thumbnail((2048,2048),Image.Resampling.BILINEAR);w,h=corrected.size
        cw,ch=(max(1,h*3//2),h) if w*2>h*3 else (w,max(1,w*2//3));x,y=(w-cw)//2,(h-ch)//2
        return corrected.crop((x,y,x+cw,y+ch)).resize((600,400),Image.Resampling.BILINEAR)


def prepare(path:Path,task_id:str)->Prepared:
    if task_id not in TASKS:raise ValueError('未知任务')
    if task_id!='isp':
        image=prepare_rgb(path);return Prepared(task_id,image,np.asarray(image,dtype=np.float32)[None]/np.float32(255))
    if path.stat().st_size!=RAW_HEADER.size+448*448*2:raise ValueError('ISP 需要配套 448×448 .rggb RAW 文件；普通 JPG/PNG 不能作为 RAW')
    data=path.read_bytes();magic,w,h,divisor,black,white=RAW_HEADER.unpack_from(data)
    if (magic,w,h,divisor,black,white)!=(b'MIE_RAW1',448,448,4095,64,1023):raise ValueError('RAW 格式或传感器配置不匹配')
    raw=np.frombuffer(data,dtype='<u2',offset=RAW_HEADER.size).reshape(h,w).copy()
    if raw.max()>white:raise ValueError('RAW 数值超过声明的 10-bit 范围')
    tensor=raw.reshape(h//2,2,w//2,2).transpose(0,2,1,3).reshape(1,h//2,w//2,4).astype(np.float32)/np.float32(divisor)
    preview=Image.fromarray((raw.astype(np.float64)/white*255).astype(np.uint8)).convert('RGB')
    return Prepared(task_id,preview,tensor,raw,data)


def gamma(image):return image.point(GAMMA_LUT*3)


def gray_world(image):
    array=np.asarray(image,dtype=np.float64);means=array.mean(axis=(0,1));gains=np.minimum(4,means.mean()/np.maximum(means,1))
    return Image.fromarray(np.clip(array*gains,0,255).astype(np.uint8))


def bilinear_isp(raw):
    h,w=raw.shape;channels=[]
    for color in range(3):
        mask=np.zeros((h,w),np.float64)
        if color==0:mask[::2,::2]=1
        elif color==2:mask[1::2,1::2]=1
        else:mask[::2,1::2]=1;mask[1::2,::2]=1
        values=np.pad(raw.astype(np.float64)*mask,1);known=np.pad(mask,1);total=np.zeros_like(mask);weight=np.zeros_like(mask)
        kernel=((1,2,1),(2,4,2),(1,2,1)) if color!=1 else ((0,1,0),(1,4,1),(0,1,0))
        for y in range(3):
            for x in range(3):
                if kernel[y][x]:total+=values[y:y+h,x:x+w]*kernel[y][x];weight+=known[y:y+h,x:x+w]*kernel[y][x]
        channels.append(total/np.maximum(weight,1)/4095)
    rgb=np.stack(channels,-1);means=rgb.mean(axis=(0,1));rgb*=np.minimum(4,means.mean()/np.maximum(means,1e-8))
    rgb=np.clip(rgb,0,1);encoded=np.where(rgb<=.0031308,rgb*12.92,1.055*np.maximum(rgb,0)**(1/2.4)-.055)
    return Image.fromarray(np.clip(encoded*255,0,255).astype(np.uint8))


def baseline(value:Prepared):
    if value.task_id=='lowlight':return gamma(value.image)
    if value.task_id=='underwater':return gray_world(value.image)
    return bilinear_isp(value.raw)


class Engine:
    def __init__(self,task_id='lowlight'):
        from ai_edge_litert.interpreter import Interpreter,OpResolverType
        self.task_id=task_id;self.task=TASKS[task_id];source=ASSETS/self.task['model_asset'];start=time.perf_counter()
        if hashlib.sha256(source.read_bytes()).hexdigest()!=self.task['model_sha256']:raise ValueError('模型校验失败，请重新获取演示包')
        self.interpreter=Interpreter(model_path=str(source),num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES);self.interpreter.allocate_tensors()
        inputs=self.interpreter.get_input_details();outputs=self.interpreter.get_output_details();assert len(inputs)==len(outputs)==1
        self.input,self.output=inputs[0],outputs[0]
        assert self.input['shape'].tolist()==self.task['input_shape'] and self.output['shape'].tolist()==self.task['output_shape']
        assert self.input['dtype']==self.output['dtype']==np.float32
        self.initialization_ms=(time.perf_counter()-start)*1000

    def run(self,value):
        if isinstance(value,Image.Image):
            if self.task_id=='isp':raise ValueError('ISP 需要 RAW，不能使用普通 RGB 图片')
            value=Prepared(self.task_id,value,np.asarray(value,dtype=np.float32)[None]/np.float32(255))
        if value.task_id!=self.task_id or list(value.tensor.shape)!=self.task['input_shape']:raise ValueError('输入与本次任务不匹配')
        self.interpreter.set_tensor(self.input['index'],value.tensor);start=time.perf_counter();self.interpreter.invoke();milliseconds=(time.perf_counter()-start)*1000
        raw=self.interpreter.get_tensor(self.output['index']).copy()
        if not np.isfinite(raw).all():raise ValueError('模型输出异常')
        return Image.fromarray((np.clip(raw[0],0,1)*np.float32(255)).astype(np.uint8)),raw,milliseconds
