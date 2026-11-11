"""Assemble existing evidence into a paper-table draft; never turn transfer into a SOTA rank."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parents[1]
BASE = REPO/'research/4.temporal-consistency'


def main():
    out = BASE/'report/paper-main-table'
    out.mkdir(parents=True, exist_ok=True)
    sources = {}
    def read(path):
        path = BASE/path
        sources[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    original = read(Path('report/exploration-r26/domain_summary.json'))
    spatial = read(Path('report/exploration-r26b/domain_summary.json'))
    dvd = read(Path('report/exploration-r12/domain_summary.json'))
    stable = read(Path('report/exploration-r15/domain_summary.json'))
    frozen = read(Path('report/exploration-r27-frozen/domain_summary.json'))
    cost = read(Path('results/exploration-r16-cost-counterbalanced/cost.json'))
    conv = {r['conv_MACs_per_frame'] for r in cost['rows'] if r['variant'] == 'official' and r['HW'] == [1080, 1920]}
    assert len(conv) == 1
    macs = next(iter(conv))
    stable_params = read(Path('results/exploration-r15-eval/config.json'))['parameters']
    dvd_params = read(Path('report/exploration-r12/audit.json'))['denoiser_parameters']
    rows = []
    def add(name, evidence, metrics, params=4075, flow=False, state=None, future=0, mobileie=True, source=''):
        rows.append({'method': name, 'evidence': evidence, 'protocol': '4 native DEV clips, 16 frames, exploratory repeats',
                     **{k: None if metrics is None else metrics[k] for k in ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']},
                     'predictor_params': params, 'fixed_motion_teacher_params': 4075 if flow else 0,
                     'predictor_conv_MACs_1080p': macs if mobileie else None,
                     'history_state': state or '0', 'flow_at_inference': flow, 'future_frames': future,
                     'phone_latency_ms': None, 'phone_status': '未测', 'source': source})
    for objective in ['charb', 'mse']:
        for fusion in ['raw', 'postwarp05']:
            r = next(r for r in original if r['domain'] == 'dev_native' and r['architecture'] == 'plain' and r['objective'] == objective
                     and r['step'] == 4000 and r['history'] == 'causal' and not r['input_reset'] and r['fusion'] == fusion)
            add('MobileIE plain/'+objective+'/'+fusion, 'matched task, CUDA', r, flow=fusion != 'raw',
                state='RGB history 23.73 MiB@1080p + motion state' if fusion != 'raw' else '0', source='R26')
    for fusion in ['raw', 'postwarp05']:
        r = next(r for r in spatial if r['domain'] == 'dev_native' and r['architecture'] == 'pre_fixedhalf' and r['objective'] == 'mse'
                 and r['step'] == 4000 and not r['input_reset'] and r['fusion'] == fusion)
        add('R26 pre/MSE backbone/current/'+fusion, 'matched task, CUDA', r, flow=fusion != 'raw',
            state='RGB history 23.73 MiB@1080p + motion state' if fusion != 'raw' else '0', source='R26b')
    for sigma in [5, 15, 30]:
        r = next(r for r in dvd if r['domain'] == 'dev_native' and r['mode'] == 'window' and r['sigma'] == sigma)
        add('R9 resize1000 + FastDVDnet sigma'+str(sigma), 'published-weight transfer, CUDA; different base training', r,
            params=4075+dvd_params, future=2, state='5-frame window; peak state unmeasured', mobileie=False, source='R12')
    r = next(r for r in stable if r['domain'] == 'dev_native' and r['method'] == 'StableLLVE_clean_raw' and not r['input_reset'])
    add('StableLLVE official clean', 'published-weight transfer, CUDA; not same-domain retraining', r,
        params=stable_params, mobileie=False, source='R15; single-frame inference, no temporal state')
    for mode, fusion in [('copy', 'postwarp05'), ('post_transport', 'raw')]:
        selected = [r for r in frozen if r['objective'] == 'charb' and r['mode'] == mode and r['history'] == 'causal' and r['fusion'] == fusion]
        metrics = {k: statistics.mean(r[k] for r in selected) for k in ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']}
        add('R27 frozen/'+mode+'/'+fusion, 'frozen mechanism, CPU; compare only this CPU pair', metrics, flow=True,
            state=('RGB history 23.73 MiB' if mode == 'copy' else '12ch feature history 94.92 MiB')+'@1080p + motion state', source='R27')
    for objective in ['charb', 'mse']:
        add('R28 post_transport/'+objective+'/current', 'candidate pending; single-frame deployment', None, source='R28 focused protocol')
        add('R28 post_transport/'+objective+'/causal', 'candidate pending; temporal deployment', None, flow=True,
            state='12ch feature history 94.92 MiB@1080p + motion state', source='R28 focused protocol')
    with (out/'evidence.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 论文主表工作稿：已有证据与缺口', '',
             '这是旧探索DEV证据清单，尚非标准基准投稿主表。不能跨证据组把异域权重迁移或CPU微小差当作公平SOTA排名。所有手机延迟均未测。', '',
             '| 方法 | 证据组 | PSNR | SSIM | 时序误差↓ | 边缘误差↓ |', '|---|---|---:|---:|---:|---:|']
    for r in rows:
        values = ['待完成' if r[k] is None else f'{r[k]:.6f}' for k in ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']]
        lines.append('|'+r['method']+'|'+r['evidence']+'|'+'|'.join(values)+'|')
    lines += ['', '| 方法 | 预测参数 | 运动teacher参数 | 预测Conv GMAC@1080p | 额外历史状态 | Flow | 未来帧 | 手机ms |',
              '|---|---:|---:|---:|---|---|---:|---|']
    for r in rows:
        value = '未计' if r['predictor_conv_MACs_1080p'] is None else f"{r['predictor_conv_MACs_1080p']/1e9:.3f}"
        lines.append(f"|{r['method']}|{r['predictor_params']}|{r['fixed_motion_teacher_params']}|{value}|{r['history_state']}|{r['flow_at_inference']}|{r['future_frames']}|未测|")
    lines += ['', '运动列对应当前共享流场评价的固定官方MobileIE + 半分辨率Farneback/FB。预计算flow不等于部署免费；需要独立teacher时要计入其权重/前向、CPUflow和warp。MAC列只计预测Conv2d，不含FST/attention/激活/warp/运动teacher；历史状态是对应方法的最低历史张量，不含完整峰值内存。', '',
              'R27 post_transport真实推理依赖过去特征与flow；其约27.06/2.71是CPU冻结探针，不是无flow的单帧结果，也不能由空间重参数化消除独立过去输入。单帧current是另一个部署映射，R28须单独评价。', '',
              '外部有效性缺口：标准公开视频协议、真实手机视频、同域训练/适配的已发表对照、手机端完整延迟及独立保留测试仍待。FastDVDnet/StableLLVE已有迁移评价，尚不能替代这些缺口。', '',
              '完整机器表：[evidence.csv](evidence.csv)；数据来源：[sources.json](sources.json)。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    (out/'sources.json').write_text(json.dumps({'sources': sources, 'rows': len(rows),
        'scope': 'Evidence draft only. Missing values are explicit; no new inference, training or phone timing.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
