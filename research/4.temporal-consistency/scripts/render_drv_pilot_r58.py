"""Render all frozen train RAW pairs after normal acquisition, never run a CNN."""
from datetime import datetime
from enum import Enum
import importlib.metadata
import json
import time
import traceback

import cv2
import numpy as np
from PIL import Image, ImageDraw
import rawpy
import torch

from drv_render_r58 import exif_metadata, raw_metadata, parameters, make_view
from dynamic_training_r34 import BASE, ROOT, digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def contact(scene, native, matched, reference, path):
    h = round(native.shape[1]*320/native.shape[2]); step = h+22
    sheet = Image.new('RGB', (960, step*8+24), 'white'); draw = ImageDraw.Draw(sheet)
    draw.text((5, 4), f'DRV train {scene} | 8 static RAW observations, one long reference | no model', fill='black')
    for t in range(8):
        for k, (image, label) in enumerate([(native[t], 'native exposure'),
                                          (matched[t], 'EXIF EV compensation (nominal)'),
                                          (reference, 'same single long reference')]):
            y = 24+t*step
            sheet.paste(Image.fromarray(image).resize((320, h), Image.Resampling.BILINEAR), (k*320, y+20))
            draw.text((k*320+3, y+3), f'{t}: {label}', fill='black')
    sheet.save(path)


def main():
    cv2.setNumThreads(1); torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    acquire = BASE/'process/external-drv/raw-acquisition001'
    runtime = json.loads((acquire/'runtime.json').read_text())
    manifest = json.loads((acquire/'manifest.json').read_text())
    assert runtime['completed_normally'] and runtime['complete_members'] == len(manifest['members']) == 36
    gate_path = BASE/'process/external-drv/render-gate001/gate.json'; gate = json.loads(gate_path.read_text())
    assert gate['passed'] and gate['source_sha256']['drv_render_r58.py'] == digest(BASE/'scripts/drv_render_r58.py')
    plan_path = BASE/'data/external-drv/raw-pilot-plan-r58.json'; plan = json.loads(plan_path.read_text())
    protocol = BASE/'docs/exploration-r58-drv-static-raw-pilot-protocol.md'
    assert digest(protocol) == plan['protocol_sha256']
    out = BASE/'results/exploration-r58-drv-render-attempt001'; out.mkdir(exist_ok=False)
    report = BASE/'report/exploration-r58-drv-raw-pilot'; report.mkdir(exist_ok=False)
    cache = ROOT/'.cache/datasets/DRV/render-pilot/attempt001'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    source_names = ['render_drv_pilot_r58.py', 'drv_render_r58.py']
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(),
          protocol_sha256=digest(protocol), plan_sha256=digest(plan_path), gate_sha256=digest(gate_path),
          acquisition_manifest_sha256=digest(acquire/'manifest.json'), views=['native', 'ev_matched'],
          WB='first short camera WB fixed for whole scene and long reference',
          image_orientation='raw sensor orientation; user_flip0', threads={'OpenMP':2, 'OpenCV':1},
          versions={'rawpy':rawpy.__version__, 'LibRaw':rawpy.libraw_version,
                    'ExifRead':importlib.metadata.version('ExifRead'), 'numpy':np.__version__},
          source_sha256={n:digest(BASE/'scripts'/n) for n in source_names},
          scope='Fixed custom rendered static train pilot, no VBM4D/CNN or standard/independent test claim.'))
    start, complete, raw_count, scenes = time.perf_counter(), False, 0, []
    try:
        metadata = []
        for row in manifest['members']:
            path = ROOT/row['path']; assert digest(path) == row['sha256']
            metadata.append(dict(**row, EXIF=exif_metadata(path)))
        write(out/'exif-metadata.json', metadata)
        # Verify all pairs before producing any RGB image, so no source is chosen by outcome.
        for scene in plan['scenes']:
            members = [r for r in metadata if r['scene'] == scene]
            long = next(r for r in members if r['role'] == 'long')['EXIF']
            assert len(members) == 9
            for r in members:
                e = r['EXIF']
                assert e['model'] == long['model'], (scene, 'camera mismatch')
                assert np.isclose(e['ISO'], long['ISO'], rtol=1e-6, atol=0), (scene, 'ISO mismatch', e, long)
                assert np.isclose(e['f_number'], long['f_number'], rtol=1e-6, atol=0), (scene, 'aperture mismatch', e, long)
        for scene in plan['scenes']:
            members = [r for r in metadata if r['scene'] == scene]
            short = sorted([r for r in members if r['role'] == 'short'], key=lambda r:r['ordinal'])
            long = next(r for r in members if r['role'] == 'long')
            native, matched, sensor, wb, reference = [], [], None, None, None
            for row in short+[long]:
                with rawpy.imread(str(ROOT/row['path'])) as raw:
                    raw_count += 1; sensor_meta = raw_metadata(raw)
                    if sensor is None:
                        sensor, wb = sensor_meta, sensor_meta['camera_whitebalance']
                        kwargs = parameters(wb)
                        write(out/f'{scene}-render-parameters.json',
                              {k: v.name if isinstance(v, Enum) else v for k,v in kwargs.items()})
                    else:
                        for key in ['raw_pattern', 'color_desc', 'rgb_xyz_matrix']:
                            assert sensor_meta[key] == sensor[key], (scene, key)
                        for key in ['height', 'width', 'flip']:
                            assert sensor_meta['sizes'][key] == sensor['sizes'][key], (scene, key)
                    linear16 = raw.postprocess(**kwargs)
                    assert linear16.dtype == np.uint16 and linear16.ndim == 3 and linear16.shape[-1] == 3
                native_frame, native_stats = make_view(linear16, 1)
                if row['role'] == 'short':
                    gain = long['EXIF']['exposure_seconds']/row['EXIF']['exposure_seconds']
                    matched_frame, matched_stats = make_view(linear16, gain)
                    native.append(native_frame); matched.append(matched_frame)
                else:
                    reference = native_frame; matched_stats = None
                detail = dict(scene=scene, role=row['role'], ordinal=row['ordinal'], source=row['path'],
                              sensor=sensor_meta, linear16_shape=list(linear16.shape), native=native_stats,
                              ev_matched=matched_stats, fixed_first_short_WB=wb)
                with (out/'per_raw.jsonl').open('a') as stream:
                    stream.write(json.dumps(detail,allow_nan=False)+'\n')
            native, matched = np.stack(native), np.stack(matched)
            assert native.shape == matched.shape and native.shape[1:] == reference.shape and len(native) == 8
            path = cache/f'{scene}.npz'
            np.savez_compressed(path, low_native=native, low_ev_matched=matched,
                                gt=np.repeat(reference[None],8,axis=0))
            contact(scene,native,matched,reference,report/f'{scene}.png')
            scenes.append(dict(scene=scene, frames=8, unique_long_references=1, official_split='train',
                               shape=list(native.shape), path=str(path.relative_to(ROOT)), sha256=digest(path),
                               input_members=[r['path'] for r in short], long_member=long['path'],
                               fixed_WB_source=short[0]['path'], views=['native','ev_matched'],
                               source_fps='approximately16–18 author claim; actual per-frame interval unknown'))
            print(json.dumps(dict(scene=scene,frames=8,shape=list(native.shape))),flush=True)
        assert raw_count == 36 and len(scenes) == 4 and not torch.cuda.is_initialized()
        write(out/'index.json', dict(scenes=scenes, CNN=0, GPU=False,
              scope='custom rendering, static sparse32 train observations, no standard/final/phone claim'))
        write(report/'audit.json', dict(at=datetime.now().astimezone().isoformat(), complete_sources=4,
              raw_members=36, input_observations=32, views=['native','ev_matched'], CNN=0, GPU=False,
              fixed_contacts=4, visual_review_required=True, complete_benchmark=False, independent_final=False))
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, opened_raw_members=raw_count,
              rendered_scenes=len(scenes), CNN=0, cuda_initialized=torch.cuda.is_initialized(),
              seconds=time.perf_counter()-start, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
