"""Every fixed native-DRV source/reference; no training, GT fitting or frame selection."""
from datetime import datetime
import json
import subprocess
import time
import traceback

import cv2
import numpy as np
import torch

from drv_references_r59 import NETWORKS, PRIMARY, load_network, infer, postprocess, metrics
from dynamic_training_r34 import BASE, ROOT, digest
from fastdvdnet_adapter_r12 import window_indices


def write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def append(path,value):
    with path.open('a') as stream:
        stream.write(json.dumps(value,allow_nan=False)+'\n')


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    gate_path = BASE/'process/exploration-r59/cpu-gate001/gate.json'
    gate = json.loads(gate_path.read_text()); assert gate['passed']
    assert gate['source_sha256']['drv_references_r59.py'] == digest(BASE/'scripts/drv_references_r59.py')
    index_path = BASE/'results/exploration-r58-drv-render-attempt001/index.json'
    assert json.loads((index_path.parent/'runtime.json').read_text())['completed_normally']
    metas = json.loads(index_path.read_text())['scenes']
    assert [m['scene'] for m in metas] == ['0001','0072','0138','0202']
    out = BASE/'results/exploration-r59-drv-native-attempt001'; out.mkdir(exist_ok=False)
    cache = ROOT/'.cache/experiments/exploration-r59/drv-native-attempt001'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True,exist_ok=False)
    names = ['evaluate_drv_native_r59.py','drv_references_r59.py','response_cue_r45.py',
             'multinex_reference_r55.py','tiny_adapted_reference_r56.py','task_fastdvd_r47.py',
             'fastdvdnet_adapter_r12.py','aligned_raw_history_r11.py','half_motion_r4.py','video_metrics.py']
    frozen = {n:digest(BASE/'scripts'/n) for n in names}
    write(out/'config.json',dict(at=datetime.now().astimezone().isoformat(),networks=NETWORKS,
          data_controls=['native_input','nominal_EV_data_control'], primary_frames=PRIMARY,
          CNN_input_view='low_native only',nominal_EV_input_to_CNN=False, GT_fitted_scaling=False,
          source_index_sha256=digest(index_path),gate_sha256=digest(gate_path),
          protocol_sha256=digest(BASE/'docs/exploration-r59-drv-native-frozen-reference-protocol.md'),
          source_sha256=frozen, future_wait_max={'fastdvd':2,'others':0},
          five_frame_indices=window_indices(8,'window').tolist(),CPU_threads=2,OpenCV_threads=1,
          revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          expected_quality_rows=88,expected_per_frame_rows=704,expected_CNN_sequences=32,
          expected_CNN_center_outputs=256,unique_native_observations=32,
          scope='Static train pilot custom ISP, repeated longGT; no method, true-flicker, standard/final/phone claim.'))
    start,complete,count,neural,centers = time.perf_counter(),False,0,0,0
    try:
        cases = []
        for meta in metas:
            path = ROOT/meta['path']; assert digest(path) == meta['sha256']
            with np.load(path,allow_pickle=False) as source:
                low, nominal, gt = source['low_native'],source['low_ev_matched'],source['gt']
            assert low.dtype == nominal.dtype == gt.dtype == np.uint8
            assert low.shape == nominal.shape == gt.shape == (8,428,640,3)
            assert np.array_equal(gt,np.broadcast_to(gt[0],gt.shape))
            cases.append(dict(meta=meta,low=low,nominal=nominal,gt=gt))
        for name in ['native_input','nominal_EV_data_control']+NETWORKS:
            model,control,source = (None,None,dict(name=name,parameters=0,future_wait_max=0,
                role='input data control; nominal EV requires paired exposure metadata')) if name not in NETWORKS else load_network(name)
            append(out/'model_sources.jsonl',source)
            for case in cases:
                begin = time.perf_counter()
                if model is None:
                    raw = (case['low'] if name == 'native_input' else case['nominal']).astype(np.float32)
                else:
                    raw = infer(name,model,case['low']); neural += 1; centers += 8
                seconds = time.perf_counter()-begin
                identity = dict(domain='DRV_static_train_native_custom',scene=case['meta']['scene'],
                                reference=name,seed=source.get('seed'),future_wait_max=source['future_wait_max'])
                append(out/'neural_stage_seconds.jsonl',dict(**identity,seconds=seconds,is_CNN=model is not None,
                       scope='CPU2 prediction stage only, excludes post/metrics/IO; not independent production/phone latency.'))
                outputs,stats = postprocess(raw,control)
                for method,output in outputs.items():
                    summary,frames = metrics(output,case['gt'])
                    row_id = dict(**identity,method=method,
                                  parameters=source['parameters']+(41 if method == 'rgb' else 0))
                    append(out/'per_sequence.jsonl',dict(**row_id,**summary,**stats))
                    for frame in frames: append(out/'per_frame.jsonl',dict(**row_id,**frame))
                    path = cache/f"{name}--{case['meta']['scene']}--{method}.npy"
                    np.save(path,output,allow_pickle=False)
                    append(out/'prediction_manifest.jsonl',dict(**row_id,path=str(path.relative_to(ROOT)),
                        sha256=digest(path),shape=list(output.shape),dtype=str(output.dtype),
                        source_cache=case['meta']['path'],input_view='native' if name != 'nominal_EV_data_control' else 'nominal_EV_data_control',
                        online_RGB_post_history_bytes=0 if method == 'raw' else 2*output[0].nbytes,
                        note='FastDVD input window/future buffer additional; post history is not whole-model storage.'))
                    count += 1
                print(json.dumps(dict(**identity,quality_rows=count,CNN_sequences=neural)),flush=True)
            del model,control
        assert count == 88 and neural == 32 and centers == 256 and not torch.cuda.is_initialized()
        assert all(digest(BASE/'scripts'/n) == value for n,value in frozen.items())
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json',dict(completed_normally=complete,quality_rows=count,
              CNN_sequences=neural,CNN_center_outputs=centers,unique_native_observations=32,
              SGD=0,cuda_initialized=torch.cuda.is_initialized(),seconds=time.perf_counter()-start,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
