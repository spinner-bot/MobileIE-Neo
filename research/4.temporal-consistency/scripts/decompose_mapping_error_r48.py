"""Post-hoc exact DC-error decomposition; no output correction or new score gate."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'research/4.temporal-consistency'


def main():
    out=BASE/'report/exploration-r48-joint-recursion'
    assert json.loads((out/'audit.json').read_text())['processing_complete']
    assert not (out/'raw_bias_decomposition.json').exists()
    source=out/'raw_mapping_sources.json'
    selection=json.loads(source.read_text());assert len(selection)==8
    rows,frames=[],[]
    for chosen in selection:
        with np.load(ROOT/chosen['GT_cache'],allow_pickle=False) as archive:gt=archive['gt']
        for name,record in zip(['R44_square_raw','plain_raw','joint_raw'],chosen['raw_records']):
            prediction=np.load(ROOT/record['path'],mmap_mode='r',allow_pickle=False)
            assert prediction.shape==gt.shape and len(gt)==16
            error=prediction.astype(np.float64)-gt.astype(np.float64)
            channel_bias=error.mean(axis=(1,2))
            mse=np.mean(error**2,axis=(1,2,3))
            dc=np.mean(channel_bias**2,axis=1)
            centered=np.mean((error-channel_bias[:,None,None,:])**2,axis=(1,2,3))
            closure=float(np.max(np.abs(mse-dc-centered)))
            assert closure<=1e-10*max(1.,float(mse.max()))
            identity={'scene':chosen['scene'],'seed':chosen['seed'],'model':name}
            rows.append({**identity,'mean_MSE':float(mse.mean()),'mean_DC_MSE':float(dc.mean()),
                         'mean_centered_MSE':float(centered.mean()),
                         'DC_fraction_of_MSE':float(dc.mean()/mse.mean()) if mse.mean()>0 else None,
                         'identity_max_abs_error':closure})
            for t in range(16):
                frames.append({**identity,'frame':t,'MSE':float(mse[t]),'DC_MSE':float(dc[t]),
                     'centered_MSE':float(centered[t]),'mean_RGB_error_0_to_255':channel_bias[t].tolist()})
    assert len(rows)==24 and len(frames)==384
    (out/'raw_bias_decomposition.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'raw_bias_per_frame.json').write_text(json.dumps(frames,indent=2)+'\n')
    text=['# R48 raw空间误差的DC/中心化分解（事后诊断）','',
          '全部四ND源、双seed、三组raw空间输出、全16帧；0新CNN/0GPU/0输出修正。',
          '逐帧逐通道误差mean为DC项，MSE = mean_channel(mean_error²) + mean((error-mean_error)²)。',
          '它不是重新拟合GT后的性能，也不替换主要PSNR/时序指标或改变A/B门槛。','',
          '|源|seed|raw模型|MSE|DC MSE|中心化MSE|DC占比|','|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"|{r['scene']}|{r['seed']}|{r['model']}|{r['mean_MSE']:.6f}|{r['mean_DC_MSE']:.6f}|{r['mean_centered_MSE']:.6f}|{r['DC_fraction_of_MSE']:.6f}|")
    text+=['','DC占比只描述空间平均通道偏差，不能证明误差由BN、GAP、曝光或ISP某一因素造成，也不是感知闪烁质量指标。']
    (out/'raw_bias_tables.md').write_text('\n'.join(text)+'\n')
    (out/'raw_bias_sources.json').write_text(json.dumps({'at':datetime.now().astimezone().isoformat(),
        'source':str(source.relative_to(ROOT)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'post_hoc':True,'output_correction':False,'changes_primary_gate':False,'new_neural_sequences':0,
        'scope':'Unfused raw mapping only; exact identity, not attribution or improved evaluation.'},indent=2)+'\n')
    print(json.dumps([r for r in rows if r['scene'] in ['video_121','video_47']]))


if __name__=='__main__':main()
