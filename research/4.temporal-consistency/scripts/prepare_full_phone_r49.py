"""Full duration of the same three registered phone DEV clips, no inference."""
import argparse
import hashlib
import json
import re
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'research/4.temporal-consistency'


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,required=True)
    args=parser.parse_args();assert args.attempt>=1
    source=BASE/'data/external-phone/manifest.json'
    previous=BASE/'results/exploration-r31-data/index.json'
    manifest=json.loads(source.read_text());old=json.loads(previous.read_text())
    assert len(manifest['videos'])==len(old['scenes'])==3
    old_by_phone={r['phone']:r for r in old['scenes']}
    out=BASE/f'results/exploration-r49-full-phone-attempt{args.attempt:03d}'
    cache=ROOT/f'.cache/experiments/exploration-r49/full-phone/attempt{args.attempt:03d}'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    out.mkdir(exist_ok=False);cache.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    write(out/'config.json',{'at':datetime.now().astimezone().isoformat(),'attempt':args.attempt,
          'opencv_version':cv2.__version__,'opencv_threads':1,'decoder_threads':1,
          'source_manifest_sha256':digest(source),'previous_index_sha256':digest(previous),
          'protocol_sha256':digest(BASE/'docs/exploration-r49-full-phone-data-protocol.md'),
          'script_sha256':digest(Path(__file__)),'frames_expected':[329,327,296],
          'resize':'Long side640; short side nearest even; INTER_AREA, full FOV',
          'format':'uint8 RGB, original brightness','model_calls':0,'GPU':False,
          'scope':'Same three DEV videos, no GT, no independent scene or phone latency claim.'})
    start,complete,rows=time.perf_counter(),False,[]
    try:
        for video,expected in zip(manifest['videos'],[329,327,296]):
            path=ROOT/video['path'];prior=old_by_phone[video['phone']]
            assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
            assert digest(path)==video['sha256']==prior['source_sha256']
            assert int(prior['video_metadata']['reported_frame_count'])==expected
            capture=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
            assert capture.isOpened() and capture.get(cv2.CAP_PROP_N_THREADS)==1
            metadata={'reported_fps':capture.get(cv2.CAP_PROP_FPS),
                      'reported_frame_count':capture.get(cv2.CAP_PROP_FRAME_COUNT),
                      'reported_width':capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                      'reported_height':capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                      'orientation_metadata_degrees':capture.get(cv2.CAP_PROP_ORIENTATION_META),
                      'orientation_auto_property':capture.get(cv2.CAP_PROP_ORIENTATION_AUTO),
                      'backend':capture.getBackendName(),'decoder_threads':capture.get(cv2.CAP_PROP_N_THREADS)}
            frames,timestamps,shape=[],[],None
            try:
                while True:
                    ok,image=capture.read()
                    if not ok:break
                    if shape is None:shape=list(image.shape)
                    assert list(image.shape)==shape
                    h,w=image.shape[:2]
                    size=(640,max(2,int(round(h*640/w/2))*2)) if w>=h else (max(2,int(round(w*640/h/2))*2),640)
                    value=cv2.cvtColor(cv2.resize(image,size,interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
                    frames.append(value);timestamps.append(capture.get(cv2.CAP_PROP_POS_MSEC))
            finally:capture.release()
            assert len(frames)==expected,(video['phone'],len(frames),expected)
            images=np.stack(frames);times=np.asarray(timestamps,dtype=np.float64)
            assert np.isfinite(times).all() and np.all(np.diff(times)>0)
            old_path=ROOT/prior['path'];assert digest(old_path)==prior['sha256']
            with np.load(old_path,allow_pickle=False) as archive:
                prefix_equal=bool(np.array_equal(images[:64],archive['low']))
            assert prefix_equal,'Preparation differs from already used DEV frames.'
            label=re.sub(r'[^A-Za-z0-9]+','_',video['phone']+'_'+path.stem).strip('_')
            target=cache/(label+'.npz')
            np.savez_compressed(target,low=images,decoded_timestamp_ms=times)
            row={'domain':'lliv_phone_full_DEV','scene':label,'phone':video['phone'],
                 'source_path':video['path'],'source_sha256':video['sha256'],
                 'path':str(target.relative_to(ROOT)),'sha256':digest(target),'shape':list(images.shape),
                 'dtype':str(images.dtype),'video_metadata':metadata,'native_decoded_shape':shape,
                 'source_frames_decoded':len(images),'extension_beyond_previous64':len(images)-64,
                 'first64_exactly_match_prior':prefix_equal,'previous_cache_sha256':prior['sha256'],
                 'decoded_timestamp_ms':times.tolist(),'timestamps_strictly_increasing':True,
                 'duration_first_to_last_timestamp_seconds':float((times[-1]-times[0])/1000),
                 'ground_truth_available':False,'timestamp_scope':'Decoder positions, not independent sensor timing.'}
            write(out/(label+'.json'),row);rows.append(row)
            print(json.dumps({'scene':label,'frames':len(images),'first64_match':prefix_equal}),flush=True)
        assert sum(r['source_frames_decoded'] for r in rows)==952
        assert sum(r['extension_beyond_previous64'] for r in rows)==760
        write(out/'index.json',{'scenes':rows,'source_manifest_sha256':digest(source),
              'previous_index_sha256':digest(previous),'model_calls':0,'ground_truth_available':False,
              'scope':'Complete same three phone DEV clips; no new independent video, GT or model quality.'})
        complete=True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'runtime.json',{'completed_normally':complete,'videos_completed':len(rows),
              'source_frames_decoded':sum(r['source_frames_decoded'] for r in rows),
              'extended_frames':sum(r['extension_beyond_previous64'] for r in rows),
              'model_calls':0,'GPU':False,'seconds':time.perf_counter()-start,
              'ended':datetime.now().astimezone().isoformat()})


if __name__=='__main__':main()
