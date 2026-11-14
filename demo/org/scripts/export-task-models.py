#!/usr/bin/env python3
"""Serialize official frozen MobileIE graphs to FP32 TFLite with builtin ops.

No fitting, weight edits, TensorFlow installation, delegates or user controls.
Numerical acceptance uses separate official PyTorch forward references.
"""
import argparse
import hashlib
import json
from pathlib import Path

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s
from ai_edge_litert.interpreter import Interpreter,OpResolverType


class Graph:
    def __init__(self):
        self.model=s.ModelT();self.model.version=3;self.model.buffers=[s.BufferT()];self.model.operatorCodes=[]
        self.graph=s.SubGraphT();self.graph.name='main';self.graph.tensors=[];self.graph.operators=[];self.model.subgraphs=[self.graph];self.codes={}

    def tensor(self,name,shape,data=None,dtype=np.float32):
        tensor=s.TensorT();tensor.name=name;tensor.shape=list(shape);tensor.hasRank=True;tensor.type=s.TensorType.FLOAT32 if dtype==np.float32 else s.TensorType.INT32
        if data is not None:
            buffer=s.BufferT();buffer.data=np.frombuffer(np.asarray(data,dtype=dtype).tobytes(),dtype=np.uint8);tensor.buffer=len(self.model.buffers);self.model.buffers.append(buffer)
        i=len(self.graph.tensors);self.graph.tensors.append(tensor);return i

    def constant(self,name,data,dtype=np.float32):
        x=np.asarray(data,dtype=dtype);return self.tensor(name,x.shape,x,dtype)

    def op(self,code,inputs,shape,name,options=None):
        if code not in self.codes:
            entry=s.OperatorCodeT();entry.builtinCode=code;entry.deprecatedBuiltinCode=min(code,127);self.codes[code]=len(self.model.operatorCodes);self.model.operatorCodes.append(entry)
        out=self.tensor(name,shape);operation=s.OperatorT();operation.opcodeIndex=self.codes[code];operation.inputs=list(inputs);operation.outputs=[out]
        if options is not None:
            operation.builtinOptions=options;operation.builtinOptionsType=getattr(s.BuiltinOptions,type(options).__name__[:-1])
        self.graph.operators.append(operation);return out

    def binary(self,code,x,y,shape,name):
        option=s.MulOptionsT() if code==s.BuiltinOperator.MUL else s.AddOptionsT()
        return self.op(code,[x,y],shape,name,option)

    def conv(self,x,weights,key,shape):
        w=weights[key+'.weight'].transpose(0,2,3,1).copy();b=weights[key+'.bias']
        option=s.Conv2DOptionsT();option.padding=s.Padding.SAME;option.strideW=option.strideH=1
        return self.op(s.BuiltinOperator.CONV_2D,[x,self.constant(key+'/kernel',w),self.constant(key+'/bias',b)],shape,key,option)


def build(weights,task):
    g=Graph();h,w,channels=(200,300,4) if task=='isp' else (400,600,3);shape=[1,h,w,12]
    inp=g.tensor('image',[1,h,w,channels]);g.graph.inputs=[inp]
    x=g.conv(inp,weights,'head.block1.0',shape)
    alpha=g.constant('head/PReLU',weights['head.block1.1.weight'].reshape(1,1,12))
    x=g.op(s.BuiltinOperator.PRELU,[x,alpha],shape,'head/prelu')
    x=g.conv(x,weights,'head.block1.2',shape)
    def fst(x,key):
        a=g.binary(s.BuiltinOperator.MUL,x,g.constant(key+'/weight1',weights[key+'.weight1']),shape,key+'/weighted1')
        b=g.binary(s.BuiltinOperator.MUL,x,g.constant(key+'/weight2',weights[key+'.weight2']),shape,key+'/weighted2')
        product=g.binary(s.BuiltinOperator.MUL,a,b,shape,key+'/product')
        return g.binary(s.BuiltinOperator.ADD,product,g.constant(key+'/bias',weights[key+'.bias'].transpose(0,2,3,1)),shape,key+'/output')
    x=fst(x,'head');x=g.conv(x,weights,'body.block1',shape);x=fst(x,'body')
    option=s.ReducerOptionsT();option.keepDims=True
    mean=g.op(s.BuiltinOperator.MEAN,[x,g.constant('spatial_axes',[1,2],np.int32)],[1,1,1,12],'attention/mean',option)
    attention=g.conv(mean,weights,'att.1',[1,1,1,12]);attention=g.op(s.BuiltinOperator.LOGISTIC,[attention],[1,1,1,12],'attention/sigmoid')
    weighted=g.binary(s.BuiltinOperator.MUL,attention,x,shape,'attention/weighted')
    option=s.ReducerOptionsT();option.keepDims=True
    maximum=g.op(s.BuiltinOperator.REDUCE_MAX,[weighted,g.constant('channel_axis',[3],np.int32)],[1,h,w,1],'attention/max',option)
    spatial=g.conv(maximum,weights,'att1.0',shape);spatial=g.op(s.BuiltinOperator.LOGISTIC,[spatial],shape,'spatial/sigmoid')
    gate=g.binary(s.BuiltinOperator.MUL,spatial,attention,shape,'dual_attention')
    x=g.binary(s.BuiltinOperator.MUL,gate,x,shape,'attention/output')
    if task=='isp':
        # PyTorch PixelShuffle uses CRD order; explicit reshape/transpose preserves it.
        option=s.ReshapeOptionsT();option.newShape=[1,h,w,3,2,2]
        x=g.op(s.BuiltinOperator.RESHAPE,[x],[1,h,w,3,2,2],'shuffle/split',option)
        x=g.op(s.BuiltinOperator.TRANSPOSE,[x,g.constant('shuffle/permutation',[0,1,4,2,5,3],np.int32)],[1,h,2,w,2,3],'shuffle/interleave',s.TransposeOptionsT())
        option=s.ReshapeOptionsT();option.newShape=[1,h*2,w*2,3]
        x=g.op(s.BuiltinOperator.RESHAPE,[x],[1,h*2,w*2,3],'shuffle/output',option)
        out=g.conv(x,weights,'tail.1',[1,400,600,3])
    else:out=g.conv(x,weights,'tail',[1,400,600,3])
    g.graph.outputs=[out];g.model.description=f'Official MobileIE {task}; frozen FP32; image input only'.encode()
    builder=flatbuffers.Builder(32768);offset=g.model.Pack(builder);builder.Finish(offset,file_identifier=b'TFL3');return bytes(builder.Output())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--weights-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);results=[]
    for task in ('underwater','isp'):
        weights=np.load(a.weights_dir/(task+'-weights.npz'));data=build(weights,task);dest=a.output/(task+'.tflite');dest.write_bytes(data)
        runtime=Interpreter(model_content=data,num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES);runtime.allocate_tensors()
        inputs=runtime.get_input_details();outputs=runtime.get_output_details();assert len(inputs)==len(outputs)==1
        cases=[]
        for name in ('black','white','random','ramp'):
            x=np.load(a.weights_dir/f'{task}-{name}-input.npy');expected=np.load(a.weights_dir/f'{task}-{name}-reference.npy')
            runtime.set_tensor(inputs[0]['index'],x);runtime.invoke();actual=runtime.get_tensor(outputs[0]['index'])
            raw=float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64))));pixels=np.abs((np.clip(actual,0,1)*255).astype(np.uint8).astype(int)-(np.clip(expected,0,1)*255).astype(np.uint8).astype(int));maximum=int(pixels.max())
            assert np.isfinite(actual).all() and raw<=1e-4 and maximum<=1,(task,name,raw,maximum)
            cases.append({'case':name,'raw_max_abs':raw,'uint8_max_abs':maximum,'pass':True})
        results.append({'task_id':task,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'input_shape':inputs[0]['shape'].tolist(),'output_shape':outputs[0]['shape'].tolist(),'input_count':len(inputs),'native_control_inputs':[],'parameter_panel':False,'reference':'official PyTorch forward, CPU','cases':cases,'all_pass':True})
    (a.output/'conversion-report.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))


if __name__=='__main__':main()
