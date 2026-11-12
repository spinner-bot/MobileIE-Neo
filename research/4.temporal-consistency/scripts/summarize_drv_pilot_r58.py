"""Describe fixed inputs and correct plot semantics without rerendering RAW pixels."""
from datetime import datetime
import json

import numpy as np

from dynamic_training_r34 import BASE, ROOT, digest
from render_drv_pilot_r58 import contact


def main():
    run = BASE/'results/exploration-r58-drv-render-attempt001'
    assert json.loads((run/'runtime.json').read_text())['completed_normally']
    assert json.loads((BASE/'process/external-drv/raw-finalize001/runtime.json').read_text())['completed_normally']
    index = json.loads((run/'index.json').read_text())
    metadata = json.loads((run/'exif-metadata.json').read_text())
    records = [json.loads(line) for line in (run/'per_raw.jsonl').read_text().splitlines()]
    report = BASE/'report/exploration-r58-drv-raw-pilot'; rows = []
    for scene in index['scenes']:
        path = ROOT/scene['path']; assert digest(path) == scene['sha256']
        with np.load(path,allow_pickle=False) as source:
            native, ev, gt = source['low_native'], source['low_ev_matched'], source['gt']
        assert native.shape == ev.shape == gt.shape == tuple(scene['shape'])
        assert np.array_equal(gt,np.broadcast_to(gt[0],gt.shape))
        short = next(r for r in metadata if r['scene'] == scene['scene'] and r['role'] == 'short')['EXIF']
        long = next(r for r in metadata if r['scene'] == scene['scene'] and r['role'] == 'long')['EXIF']
        values = [r for r in records if r['scene'] == scene['scene'] and r['role'] == 'short']
        rows.append(dict(scene=scene['scene'], frames=8, shape=list(native.shape), ISO=short['ISO'],
                         f_number=short['f_number'], short_exposure_seconds=short['exposure_seconds'],
                         long_exposure_seconds=long['exposure_seconds'],
                         actual_short_gain_values=sorted(set(r['ev_matched']['exposure_gain'] for r in values)),
                         mean_native_RGB=float(native.mean()), mean_nominal_EV_RGB=float(ev.mean()),
                         mean_reference_RGB=float(gt[0].mean()),
                         nominal_EV_to_reference_mean_ratio=float(ev.mean()/gt[0].mean()),
                         source_exif_datetime=short['datetime_original'],
                         actual_frame_intervals='unknown: no EXIF subsecond values',
                         zero_native_channel_fraction=float(np.mean(native == 0))))
        contact(scene['scene'],native,ev,gt[0],report/f'{scene["scene"]}.png')
    (report/'input-statistics.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    text = ['# R58 真实相机极低光静态资料：固定双视图', '',
            '36原始RAW已完整取得；四官方train源×8连续短曝光，单一长曝光参考逐帧重复仅作配对便利。',
            '不是完整标准/独立最终动态夜间测试，也不是手机成本或增强模型结果。0202拍摄于16:44，不能把全部四源称自然夜间。', '',
            '**EV视图仅按EXIF比值作名义补偿，实测亮度并不匹配GT。** 内部键ev_matched保留原登记名称；不得据名字当成纯去噪基准。native保留原曝光。',
            '图中“EXIF EV compensation (nominal)”纠正原标签，来自原缓存，未重做RAW处理、改gain/选帧或用GT拟合修正。', '',
            '|源|ISO|EXIF倍率|native均值|名义EV均值|参考均值|EV/参考均值|',
            '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"|{r['scene']}|{r['ISO']:.0f}|{r['actual_short_gain_values']}|{r['mean_native_RGB']:.4f}|"
                    f"{r['mean_nominal_EV_RGB']:.4f}|{r['mean_reference_RGB']:.4f}|{r['nominal_EV_to_reference_mean_ratio']:.4f}|")
    text += ['', '均值为RGB通道0–255数值的全帧平均，只描述输入，不是质量/噪声参数/物理因果或时间一致性。',
             '0138的名义EV视图严重偏亮偏紫；其余源也有颜色/噪声差异。保留所有源和视图，不校正到GT均值。',
             '固定第一短帧WB，长短ISO/光圈/相机/几何与色矩阵资格均通过；这不证明逐像素精配准/辐照度一致或正确相机噪声标定。',
             'rawpy0.25.1/LibRaw0.21.4半尺寸线性RGB16→固定缩放→sRGB→uint8，不含VBM4D，也不是作者原ISP复现。',
             '各图8行依次为前8张短曝光；三列为native、名义EV补偿、同一长参考，不假定30FPS。', '']
    text += [f'![固定源{s}全部观察]({s}.png)' for s in ['0001','0072','0138','0202']]
    (report/'README.md').write_text('\n'.join(text)+'\n')
    record = dict(at=datetime.now().astimezone().isoformat(), original_render_complete=True,
                  input_observations=32, unique_long_references=4, RAW_members_redecoded=0, CNN=0, GPU=False,
                  native_and_nominal_EV_tasks_distinct=True, nominal_EV_not_empirically_brightness_matched=True,
                  GT_fitted_scaling=False, corrected_contact_labels_only=True,
                  source_sha256={str(p.relative_to(ROOT)):digest(p) for p in
                      [run/'index.json',run/'per_raw.jsonl',run/'exif-metadata.json',
                       BASE/'scripts/summarize_drv_pilot_r58.py',BASE/'scripts/render_drv_pilot_r58.py']})
    (report/'interpretation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(rows))


if __name__ == '__main__': main()
