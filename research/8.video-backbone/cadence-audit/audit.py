"""Source-index and reported-time evidence only; no image/video/model imports."""
import csv
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
OLD=ROOT/'research/4.temporal-consistency'
OUT=BASE/'results/audit001'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    began=time.perf_counter();OUT.mkdir(parents=True,exist_ok=False)
    split_path=BASE/'process/davis-split-read001.json';split=read(split_path)
    assert split['completed_normally'] and split['image_members_read']==0
    original2016=set().union(*(set(v['scenes']) for v in split['splits'].values()))
    assert len(original2016)==50
    sources=[BASE/'scope.md',Path(__file__).resolve(),split_path]
    rows=[]
    for relative,role,count in [('data/exploration-r4b/manifest.json','train',12),('data/exploration-r9-dev/manifest.json','dev',4)]:
        path=OLD/relative;manifest=read(path);sources.append(path)
        assert manifest['revision']=='291ae19d061dedcff8e62f9b6d9d33d7ceeedbaf'
        selection=[r for r in manifest['selection'] if r['split']==role]
        assert len(selection)==count
        for item in selection:
            scene=item['scene'];ids=[]
            for side in ('input','gt'):
                files=sorted([r for r in manifest['files'] if r['scene']==scene and r['side']==side],key=lambda r:r['source_path'])
                ids.append([int(Path(r['source_path']).stem) for r in files])
            assert ids[0]==ids[1]==list(range(16))
            rows.append(dict(dataset='OLiVES',role=role,scene=scene,frames=16,first_source_index=0,last_source_index=15,
                index_step=1,local_temporal_resampling=False,source_documented_fps=25,
                fps_evidence='OLiVES v1 section3.1 reported acquisition; export PTS not independently verified',
                source_manifest=str(path.relative_to(ROOT))))
    for relative,role,count in [('data/exploration-r32-training/manifest.json','train',57),('data/external-davis/manifest.json','dev',3)]:
        path=OLD/relative;manifest=read(path);sources.append(path)
        groups={}
        for r in manifest['frames']:
            scene=r.get('scene',Path(r['name']).parent.name)
            groups.setdefault(scene,[]).append(int(Path(r['name']).stem))
        assert len(groups)==count
        for scene,ids in sorted(groups.items()):
            assert ids==sorted(ids)==list(range(len(ids))) and len(ids)<=64
            rows.append(dict(dataset='DAVIS2016' if scene in original2016 else 'DAVIS2017_extra',role=role,scene=scene,
                frames=len(ids),first_source_index=ids[0],last_source_index=ids[-1],index_step=1,
                local_temporal_resampling=False,source_documented_fps=24 if scene in original2016 else None,
                fps_evidence='CVPR2016 section3 reported24fps and official2016 split membership' if scene in original2016 else 'No per-source rate established in this audit',
                source_manifest=str(path.relative_to(ROOT))))
    assert len(rows)==76 and sum(r['frames'] for r in rows if r['dataset'].startswith('DAVIS') and r['role']=='train')==3428
    assert not {'video_43','video_71','video_83'}&{r['scene'] for r in rows}
    for relative,key,manifest_relative in [('results/exploration-r10-cache/index.json','manifest_sha256','data/exploration-r9-dev/manifest.json'),
                                         ('results/exploration-r30-cache/index.json','source_manifest_sha256','data/external-davis/manifest.json')]:
        p=OLD/relative;sources.append(p);assert read(p)[key]==sha(OLD/manifest_relative)
    windows=ROOT/'research/8.video-backbone/latent-core/results/held-seed0/windows.jsonl';sources.append(windows)
    draws=[]
    for line in windows.read_text().splitlines():
        record=json.loads(line)
        for i,w in enumerate(record['windows']):
            indices=[int(Path(s).stem) for s in w['members']] if w['source']=='davis_synthetic' else w['indices']
            assert len(indices)==4 and indices==list(range(indices[0],indices[0]+4))
            draws.append(dict(draw_key=f"step{record['data_step']}-window{i}",source=w['source'],scene=w['video'],source_indices=indices))
    assert len(draws)==32
    write('source-index-mapping.json',dict(sources=rows,L1_held_draws=draws,
        local_no_temporal_decimation_verified=True,source_sensor_timestamps_independently_verified=False))
    with (OUT/'source-index-summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    phone=OLD/'results/exploration-r49-full-phone-attempt001/index.json'
    phone_runtime=phone.parent/'runtime.json';sources.extend([phone,phone_runtime])
    assert read(phone_runtime)['completed_normally']
    phones=[]
    for r in read(phone)['scenes']:
        times=r['decoded_timestamp_ms'];delta=[b-a for a,b in zip(times,times[1:])]
        assert len(times)==r['source_frames_decoded'] and all(d>0 for d in delta)
        median=statistics.median(delta)
        phones.append(dict(phone=r['phone'],frames=len(times),reported_fps=r['video_metadata']['reported_fps'],
            first_timestamp_ms=times[0],last_timestamp_ms=times[-1],minimum_interval_ms=min(delta),median_interval_ms=median,
            maximum_interval_ms=max(delta),median_derived_fps=1000/median,
            maximum_absolute_interval_deviation_from_median_ms=max(abs(d-median) for d in delta),
            evidence='Previously recorded OpenCV FFMPEG decoder positions; not independent sensor timing or model latency'))
    assert sum(r['frames'] for r in phones)==952
    write('existing-phone-timestamps.json',dict(videos=phones,new_video_decodes=0,new_packet_probes=0))
    counts={key:sum(r['dataset']==key for r in rows) for key in ('OLiVES','DAVIS2016','DAVIS2017_extra')}
    write('time-contract.json',dict(default_time_unit='registered_source_frame_index',
        local_selected_index_step=1,source_groups=counts,
        source_nominal_rates=dict(OLiVES=dict(fps=25,level='paper_reported_acquisition',sensor_or_export_PTS_verified=False),
            DAVIS2016=dict(fps=24,level='paper_reported_for_2016_sources_confirmed_by_split',sensor_or_export_PTS_verified=False),
            DAVIS2017_extra=dict(fps=None,level='not_established_for_added_sources')),
        conditional_one_frame_ms=dict(OLiVES=40.,DAVIS2016=1000/24,DAVIS2017_extra=None),
        conversion_assumption='Only if released source-frame indexing preserves reported acquisition cadence; arithmetic is not measured end-to-end handset latency',
        phone_timestamp_evidence='Existing decoder positions only; use per-source intervals, do not assume every clip is exactly30fps',
        old_nominal30fps_JOD_or_quality_values_changed=False,new_metrics_computed=False,
        independent_final_pixels_read=False))
    text=['# 帧索引/时间基准核查','',
        '|资料|源数|本地索引步长|作者报告帧率|证据边界|','|---|---:|---:|---|---|',
        f"|OLiVES训练/DEV|{counts['OLiVES']}|1|25fps|原PNG前16，无本地时间重采样；源端PTS未验|",
        f"|DAVIS2016训练/DEV|{counts['DAVIS2016']}|1|24fps|官方2016清单成员/原JPEG前缀；非独立PTS|",
        f"|DAVIS2017新增训练/DEV|{counts['DAVIS2017_extra']}|1|未确立|不能把2016的24fps推广到新增源|",'',
        'D1：bear/lucia属2016 train，walking不在原2016清单。L1全部32个四帧draw保持来源编号连续；并不因此证明传感器逐帧等间隔。','',
        '|手机原R49记录|帧数|reported fps|最小/中位/最大间隔ms|','|---|---:|---:|---|']
    for r in phones:
        text.append(f"|{r['phone']}|{r['frames']}|{r['reported_fps']:.9f}|{r['minimum_interval_ms']:.9f} / {r['median_interval_ms']:.9f} / {r['maximum_interval_ms']:.9f}|")
    text+=['','无新解码/像素/模型/flow/训练/GPU；旧质量/名义30fps JOD保持。条件式40/41.667ms只是依据报告帧率的换算，不是实际手机端到端时延。']
    (OUT/'tables.md').write_text('\n'.join(text)+'\n')
    scripts=['fetch_olives_r4.py','fetch_development_r9.py','prepare_native_r10.py','inspect_davis_external.py',
             'fetch_davis_pilot.py','prepare_davis_pilot.py','fetch_davis_training.py','dynamic_training_r34.py','mixed_anchor_r40.py']
    sources.extend(OLD/'scripts'/s for s in scripts)
    write('runtime.json',dict(at=datetime.now(timezone(timedelta(hours=8))).isoformat(),completed_normally=True,
        seconds=time.perf_counter()-began,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        source_groups=76,held_draws=32,existing_phone_timestamp_rows=952,
        new_image_or_video_decodes=0,new_model_or_flow_or_optimizer_calls=0,GPU_used=False,
        limits='Metadata + stated source-code reading only; numeric contiguity does not establish physical PTS or source-side frame export fidelity'))
    print(json.dumps(dict(completed=True,source_groups=counts,phone_videos=phones)))


if __name__=='__main__':main()
