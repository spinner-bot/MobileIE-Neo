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
    transport = read(Path('report/exploration-r28/domain_seed_summary.json'))
    transport_audit = read(Path('report/exploration-r28/audit.json'))
    assert transport_audit['passed'] and transport_audit['rows'] == 4416
    external_audit = read(Path('report/exploration-r30/audit.json'))
    assert external_audit['complete_registered_model_set'] and external_audit['rows'] == 312
    cost = read(Path('results/exploration-r16-cost-counterbalanced/cost.json'))
    external = read(Path('results/exploration-r30-cache/index.json'))
    phones = read(Path('results/exploration-r31-data/index.json'))
    prior_overlap = read(Path('data/external-davis/fastdvdnet-training-overlap.json'))
    heldout_metadata = read(Path('data/external-davis/test-dev-metadata-reserve.json'))
    assert external['source_video_count'] == 3 and external['input_conditions'] == 6 and len(phones['scenes']) == 3
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
        for history, fusion in [('current', 'raw'), ('current', 'postwarp05'), ('causal', 'raw')]:
            selected = [r for r in transport if r['domain'] == 'dev_native' and r['mode'] == 'post_transport'
                        and r['objective'] == objective and r['history'] == history and r['fusion'] == fusion and not r['input_reset']]
            assert len(selected) == 2
            metrics = {k: statistics.mean(r[k] for r in selected) for k in ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']}
            temporal = history == 'causal'
            post = fusion != 'raw'
            evidence = 'same-weight reference, CUDA; classical post' if post else 'trained candidate, CUDA; registered screening failed'
            add('R28 post_transport/'+objective+'/'+history+'/'+fusion, evidence, metrics, flow=temporal or post,
                state=('12ch feature history 94.92 MiB@1080p + motion state' if temporal else 'RGB history 23.73 MiB@1080p + motion state' if post else '0'),
                source='R28 fixed4000, dual seed; current/raw is single-frame')
    with (out/'evidence.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    external_rows = []
    external_summary = next((p for p in [Path('report/external-baselines/domain_summary.json'),
                                         Path('report/external-baselines-self-motion/domain_summary.json')]
                             if (BASE/p).exists()), None)
    if external_summary is not None:
        summaries = read(external_summary)
        audit = read(external_summary.with_name('audit.json'))
        assert audit['passed']
        pipeline_path = BASE/'results/exploration-r30-self-motion/pipeline_cost.jsonl'
        sources[str(pipeline_path.relative_to(REPO))] = hashlib.sha256(pipeline_path.read_bytes()).hexdigest()
        pipeline = [json.loads(line) for line in pipeline_path.read_text().splitlines()]
        assert len(pipeline) == 12 and all(r['frames'] == 64 for r in pipeline)
        pipeline_ms = 1000*sum(r['pipeline_seconds'] for r in pipeline)/sum(r['frames'] for r in pipeline)
        for r in summaries:
            method = r['method']
            published = method.startswith('fastdvd_')
            shared_motion = method == 'shared_motion_post05'
            self_motion = method == 'self_motion_post05'
            external_rows.append({'method': method, 'protocol': 'R30: 3 DEV videos, 2 noise conditions, 64 frames, 2 training seeds, CPU',
                  **{m: r[m] for m in ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']},
                  'predictor_params': 4075+dvd_params if published else 4075,
                  'fixed_quality_motion_teacher_params': 4075 if shared_motion else 0,
                  'predictor_conv_MACs_1080p': None if published else macs,
                  'flow_at_inference': shared_motion or self_motion,
                  'future_frames': 2 if method.endswith('_window') else 0,
                  'cpu_resident_pipeline_ms_per_frame_480x854': pipeline_ms if self_motion else None,
                  'phone_latency_ms': None,
                  'documented_pretraining_video_overlap': 'all three DEV IDs' if published else 'not established by this audit',
                  'scope': 'Transfer DEV only' if published else 'Matched R26 plain MSE base; custom synthetic DEV'})
        with (out/'external_evidence.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(external_rows[0]))
            writer.writeheader()
            writer.writerows(external_rows)
    lines = ['# 论文主表工作稿：已有证据与缺口', '',
             '这是旧探索DEV证据清单，尚非标准基准投稿主表。不能跨证据组把异域权重迁移或CPU微小差当作公平SOTA排名。所有手机延迟均未测。', '',
             '| 方法 | 证据组 | PSNR | SSIM | 时序误差↓ | 边缘误差↓ |', '|---|---|---:|---:|---:|---:|']
    for r in rows:
        values = ['待完成' if r[k] is None else f'{r[k]:.6f}' for k in ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']]
        lines.append('|'+r['method']+'|'+r['evidence']+'|'+'|'.join(values)+'|')
    lines += ['', '| 方法 | 预测参数 | 共享质量评价teacher参数 | 预测Conv GMAC@1080p | 额外历史状态 | Flow | 未来帧 | 手机ms |',
              '|---|---:|---:|---:|---|---|---:|---|']
    for r in rows:
        value = '未计' if r['predictor_conv_MACs_1080p'] is None else f"{r['predictor_conv_MACs_1080p']/1e9:.3f}"
        lines.append(f"|{r['method']}|{r['predictor_params']}|{r['fixed_motion_teacher_params']}|{value}|{r['history_state']}|{r['flow_at_inference']}|{r['future_frames']}|未测|")
    lines += ['', '运动列对应当前共享流场质量评价的固定官方MobileIE + 半分辨率Farneback/FB，并非经典post所有部署都需要第二个CNN。自驱运动post可复用自身增强输出，见下面外部同底座表。预计算flow不等于部署免费；需要独立teacher时要计入其权重/前向、CPUflow和warp。MAC列只计预测Conv2d，不含FST/attention/激活/warp/运动teacher；历史状态是对应方法的最低历史张量，不含完整峰值内存。', '',
              'R27 post_transport真实推理依赖过去特征与flow；其约27.06/2.71是CPU冻结探针，不是无flow的单帧结果，也不能由空间重参数化消除独立过去输入。单帧current是另一个部署映射，R28须单独评价。', '',
              'R28的4416行共同CUDA评价已完成，current/raw与causal/raw均未通过注册质量—时序筛查。causal需要flow和12ch历史状态，主路线冻结；同权重current+post保留完整，不能把训练后空间映射变化归为真实历史收益。', '',
              '外部开发评价：[R30 DAVIS](../../docs/exploration-r30-protocol.md)为3段官方TRAIN来源、6个合成低光条件、每段64帧；双seed312行[共同CPU结果](../exploration-r30/tables.md)已完成，仍未建立候选联合优势，与上表CUDA原生域分开。[R31真实手机](../../docs/exploration-r31-protocol.md)为3设备各64帧，无同步GT，[原亮度预览](../exploration-r31-data/phone_inputs.png)，尚无方法质量结论。', '',
              '外部有效性缺口：完整标准公开视频验证、真实手机域方法评价、同域训练/适配的已发表对照、手机端完整延迟及独立保留测试仍待。FastDVDnet/StableLLVE已有迁移评价，尚不能替代这些缺口。', '',
              '完整机器表：[evidence.csv](evidence.csv)；数据来源：[sources.json](sources.json)。']
    if external_rows:
        lines += ['', '## 外部同底座CPU证据（独立表）', '',
                  '以下均为R26 plain MSE双seed、同一R30六条件。与上方native/CUDA表分开；仅自驱post已测CPU驻留输入流水线，不能当手机延迟。', '',
                  '|方法|PSNR|SSIM|时序误差↓|边缘误差↓|CPU流水线ms/帧@480×854|',
                  '|---|---:|---:|---:|---:|---:|']
        for r in external_rows:
            cpu = r['cpu_resident_pipeline_ms_per_frame_480x854']
            lines.append('|'+r['method']+'|'+'|'.join(f'{r[m]:.6f}' for m in ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae'])+'|'+('未测' if cpu is None else f'{cpu:.3f}')+'|')
        lines += ['', '自驱post没有独立teacher前向，状态为一张raw增强RGB；该CPU循环总时间含输出留存指针开销，分帧计时不含后续指标/文件保存。未测完整峰值内存或相机到屏幕延迟。[外部机器表](external_evidence.csv)。']
    lines += ['', '预训练来源限制：官方FastDVDnet文档训练清单覆盖R30三开发片及DAVIS TRAIN60/VAL30，因此新的R35迁移结果不能声称未见视频泛化；旧R12的ND原生表不因该DAVIS重叠自动变成训练片。已仅按元数据保留另30段test-dev来源/1788帧，与该90训练ID及项目训练/开发ID不重叠，JPEG成员尚未读取。这仍不是官方LLVE标准基准，其他公开权重来源需分别核验。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    (out/'sources.json').write_text(json.dumps({'sources': sources, 'rows': len(rows),
        'external_data_prepared': {'DAVIS_source_videos': 3, 'DAVIS_synthetic_conditions': 6,
                                   'real_phone_videos': 3, 'phone_reference_GT_available': False,
                                   'frames_per_clip': 64, 'method_results_in_this_table': False},
        'R28_screening_status': 'failed; route frozen as primary method, not universal impossibility',
        'R30_external_pilot_status': 'complete312 rows, common CPU; separate from native CUDA table; candidate screening failed',
        'external_same_base_rows': len(external_rows),
        'FastDVDnet_documented_R30_DEV_overlap': prior_overlap['overlap_R30_DEV'],
        'heldout_metadata_only': {'video_count': heldout_metadata['video_count'], 'future_frames': heldout_metadata['future_total_frames'],
                                  'JPEG_members_read': heldout_metadata['JPEG_members_read']},
        'scope': 'Evidence draft only. Missing values are explicit; no new inference, training or phone timing.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
