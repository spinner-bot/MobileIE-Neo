"""Complete-only table and fixed-frame inspection of every native-DRV output."""
from collections import defaultdict
from datetime import datetime
import json

import numpy as np
from PIL import Image, ImageDraw

from dynamic_training_r34 import BASE, ROOT, digest


def lines(path): return [json.loads(x) for x in path.read_text().splitlines()]


def write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def fmt(value): return '+inf' if value is None else f'{value:.4f}'


def main():
    run = BASE/'results/exploration-r59-drv-native-attempt001'
    runtime = json.loads((run/'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['quality_rows'] == 88
    assert runtime['CNN_sequences'] == 32 and runtime['CNN_center_outputs'] == 256
    rows,frames,manifest = [lines(run/n) for n in ['per_sequence.jsonl','per_frame.jsonl','prediction_manifest.jsonl']]
    assert len(rows) == len(manifest) == 88 and len(frames) == 704
    groups = defaultdict(list)
    for r in rows: groups[r['reference'],r['method']].append(r)
    assert len(groups) == 22 and all(len(v) == 4 for v in groups.values())
    summary = []
    keys = ['psnr_rgb','ssim_rgb','edge_mae','full8_psnr_rgb','full8_ssim_rgb','full8_edge_mae',
            'same_coordinate_time_std_RMS','same_coordinate_adjacent_RGB_MAE','mean_RGB','target_mean_RGB']
    for (reference,method), selected in groups.items():
        value = {k: None if any(r[k] is None for r in selected) else float(np.mean([r[k] for r in selected])) for k in keys}
        summary.append(dict(reference=reference,method=method,parameters=selected[0]['parameters'],
                            future_wait_max=selected[0]['future_wait_max'],**value))
    out = BASE/'report/exploration-r59-drv-native'; out.mkdir(exist_ok=False)
    write(out/'domain-summary.json',summary); write(out/'conditions.json',rows)
    table = ['# R59 固定原曝光DRV冻结参考：完整结果', '',
             '四官方train静态源，32真实观察/4唯一长参考；非标准完整/独立最终/动态夜间或手机实验。',
             'CNN只读native，名义EV为使用配对曝光元数据的数据控制，未GT拟合、未实测亮度匹配。',
             '主空间指标所有方法用帧2–5，避免五帧参考的边界反射；全8PSNR同时保留。',
             '变化代理为全8同坐标相邻RGB MAE，含真实变化/噪声/边界，不是GT校正闪烁；暗且稳定也可能很低。', '',
             '|参考|输出|参数|未来帧|主PSNR|主SSIM|全8PSNR|同坐标变化代理↓|',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        table.append(f"|{r['reference']}|{r['method']}|{r['parameters']}|{r['future_wait_max']}|"
                     f"{fmt(r['psnr_rgb'])}|{fmt(r['ssim_rgb'])}|{fmt(r['full8_psnr_rgb'])}|{fmt(r['same_coordinate_adjacent_RGB_MAE'])}|")
    table += ['', '同目标/更新数不是同预训练/训练参数化/计算或充分收敛。RGB gate还额外4000控制器更新。',
              'recursive/RGB有motion与历史成本，FastDVDnet有五帧/未来输入；列出参数不代表手机速度。',
              '没有本轮方法采用门槛，不把这些输入或源/seed选优用于新增训练。']
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    conditions = ['# 每源完整空间结果（共同主帧2–5）', '',
                  '|源|参考|输出|主PSNR|全8PSNR|主SSIM|变化代理|', '|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        conditions.append(f"|{r['scene']}|{r['reference']}|{r['method']}|{fmt(r['psnr_rgb'])}|{fmt(r['full8_psnr_rgb'])}|"
                          f"{fmt(r['ssim_rgb'])}|{fmt(r['same_coordinate_adjacent_RGB_MAE'])}|")
    (out/'conditions.md').write_text('\n'.join(conditions)+'\n')
    index_path = BASE/'results/exploration-r58-drv-render-attempt001/index.json'
    metas = json.loads(index_path.read_text())['scenes']
    for meta in metas:
        with np.load(ROOT/meta['path'],allow_pickle=False) as data: gt = data['gt'][4]
        images = [(gt,'single long reference')]
        for item in manifest:
            if item['scene'] != meta['scene']: continue
            array = np.load(ROOT/item['path'],mmap_mode='r',allow_pickle=False)
            assert list(array.shape) == item['shape'] and array.dtype == np.float32
            value = np.clip(np.rint(array[4]),0,255).astype(np.uint8)
            images.append((value,item['reference']+' / '+item['method']))
        assert len(images) == 23
        sheet = Image.new('RGB',(1280,24+6*236),'#eeeeee'); draw = ImageDraw.Draw(sheet)
        draw.text((5,4),f"DRV train {meta['scene']} | fixed primary frame4 | all22 outputs, no selection",fill='black')
        for i,(image,label) in enumerate(images):
            x,y=(i%4)*320,24+(i//4)*236
            draw.text((x+3,y+3),label,fill='black')
            sheet.paste(Image.fromarray(image).resize((320,214),Image.Resampling.BILINEAR),(x,y+20))
        sheet.save(out/f"{meta['scene']}--frame4.png")
    write(out/'sources.json',{str(p.relative_to(ROOT)):digest(p) for p in
          [run/'runtime.json',run/'config.json',run/'model_sources.jsonl',run/'per_sequence.jsonl',
           run/'prediction_manifest.jsonl',index_path,BASE/'scripts/summarize_drv_native_r59.py']})
    write(out/'audit.json',dict(at=datetime.now().astimezone().isoformat(),completed_normally=True,
          quality_rows=88,per_frame_rows=704,CNN_sequences=32,CNN_center_outputs=256,
          unique_native_observations=32,unique_long_references=4,primary_unique_targets=16,
          all22_outputs_retained=True,fixed_frame=4,contacts=4,visual_review_required=True,
          new_method=False,true_flicker_evaluated=False,independent_final=False,phone_measured=False))
    print(json.dumps(dict(complete=True,quality_rows=88,outputs=22,contacts=4)))


if __name__ == '__main__': main()
