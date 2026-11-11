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
    dynamic_rows, dynamic_audit, dynamic_cost_audit = [], None, None
    if (BASE/'report/exploration-r34/audit.json').exists():
        dynamic_audit = read(Path('report/exploration-r34/audit.json'))
        assert dynamic_audit['passed'] and dynamic_audit['rows'] == 120
        summaries = read(Path('report/exploration-r34/domain_summary.json'))
        comparisons = read(Path('report/exploration-r34/paired_domain_seed.json'))
        assert len(summaries) == 10
        training = read(Path('results/exploration-r34-seed0/config.json'))
        candidate_cost = read(Path('report/exploration-r33-prototype/checks.json'))
        cost_lookup = {}
        if (BASE/'report/exploration-r34-runtime/audit.json').exists():
            dynamic_cost_audit = read(Path('report/exploration-r34-runtime/audit.json'))
            assert dynamic_cost_audit['passed'] and dynamic_cost_audit['runs'] == 42
            measured = read(Path('report/exploration-r34-runtime/cost.json'))
            cost_lookup = {(r['model'], r['history'], r['fusion']): r for r in measured}
            assert len(cost_lookup) == 7
        def gate_against(model):
            values = [r for r in comparisons if r['reference_model'] == model and r['reference_fusion'] == 'self_motion_post05']
            assert len(values) == 2 and {r['seed'] for r in values} == {0, 1}
            return all(r['descriptive_quality_temporal_gate'] for r in values)
        for r in summaries:
            measured = cost_lookup.get((r['model'], r['history'], r['fusion']))
            actual = r['history'] == 'causal'
            shared = r['fusion'] == 'postwarp05'
            post = r['fusion'] != 'raw'
            sampling = 'none' if r['model'] == 'original' else 'learned causal low-resolution transport' if actual else 'learned single-frame sampling'
            if post:
                sampling += ' + '+('shared teacher' if shared else 'own enhancement')+' half-FB/warp'
            dynamic_rows.append({'model': r['model'], 'history': r['history'], 'fusion': r['fusion'],
                'protocol': 'R34 final4000; 57 training videos, 3 disjoint DEV videos x 2 noise conditions x 2 seeds; CPU',
                **{m: r[m] for m in ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']},
                'predictor_params': training['parameters'][r['model']],
                'predictor_conv_MACs_1080p': macs if r['model'] == 'original' else candidate_cost['conv_MACs_1080p'],
                'fixed_quality_motion_teacher_params': 4075 if shared else 0, 'sampling_path': sampling,
                'future_frames': 0, 'online_history_bytes_480x854': None if measured is None else measured['online_history_storage_bytes'],
                'cpu_median_pipeline_ms_per_frame_480x854': None if measured is None else measured['median_pipeline_ms_per_frame'],
                'cpu_pipeline_min_run_average_ms': None if measured is None else measured['min_pipeline_ms_per_frame'],
                'cpu_pipeline_max_run_average_ms': None if measured is None else measured['max_pipeline_ms_per_frame'],
                'phone_latency_ms': None,
                'registered_spatial_self_post_gate_both_seeds': dynamic_audit['registered_spatial_post_gate_passed_both_seeds'] if actual else None,
                'original_self_post_gate_both_seeds': gate_against('original') if actual else None,
                'same_weight_self_post_gate_both_seeds': gate_against('causal') if actual else None})
        with (out/'dynamic_evidence.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dynamic_rows[0]))
            writer.writeheader()
            writer.writerows(dynamic_rows)
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
    if dynamic_rows:
        lines += ['', '## R34公共动态训练后的固定双seed对照', '',
                  '以下三控制均同资料/同更新预算，采用最终4000权重。质量是R30六条件/三源视频的共同CPU均值；与上面R26旧训练资料/原生CUDA表分开。成本若已测，仅来自固定单片前16帧、两seed各三轮的CPU驻留输入流水线，不能当手机速度。', '',
                  '|模型|历史|输出|PSNR|SSIM|时序↓|边缘↓|CPU中位ms/帧|',
                  '|---|---|---|---:|---:|---:|---:|---:|']
        for r in dynamic_rows:
            cpu = r['cpu_median_pipeline_ms_per_frame_480x854']
            lines.append('|'+ '|'.join(r[f] for f in ['model', 'history', 'fusion'])+'|'+'|'.join(f'{r[m]:.6f}' for m in ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae'])+'|'+('未测' if cpu is None else f'{cpu:.3f}')+'|')
        actual = next(r for r in dynamic_rows if r['history'] == 'causal')
        lines += ['', f"actual相对独立空间+self-post的注册联合门槛（双seed均过）：{actual['registered_spatial_self_post_gate_both_seeds']}；相对原始模型+self-post：{actual['original_self_post_gate_both_seeds']}；相对自身current+self-post：{actual['same_weight_self_post_gate_both_seeds']}。门槛仅为DEV筛查，不能替代全部质量/成本、新颖性、手机和独立测试证据。", '',
                  '同权重current保留训练后的空间路径；actual内部仍有学习偏移/采样，不能称完全无对齐成本。shared-motion post仅用原共享teacher质量控制，未实测其完整部署成本。[动态机器表](dynamic_evidence.csv)、[配对与分seed全表](../exploration-r34/tables.md)。']
    else:
        lines += ['', 'R34固定动态训练与120行双seed评价尚未产生完整汇总，当前不填入候选质量或成本，不由部分训练loss推断效果。']
    classical_rows = []
    if (BASE/'report/exploration-r36-classical/audit.json').exists():
        classical_audit = read(Path('report/exploration-r36-classical/audit.json'))
        assert classical_audit['passed'] and classical_audit['rows'] == 48
        assert classical_audit['reference_checks'] == 24
        classical = read(Path('report/exploration-r36-classical/domain_summary.json'))
        classical_gates = read(Path('report/exploration-r36-classical/gates.json'))
        assert len(classical) == 4 and len(classical_gates) == 4
        classical_cost = {}
        if (BASE/'report/exploration-r36-runtime/audit.json').exists():
            classical_cost_audit = read(Path('report/exploration-r36-runtime/audit.json'))
            assert classical_cost_audit['passed'] and classical_cost_audit['runs'] == 18
            measured = read(Path('report/exploration-r36-runtime/cost.json'))
            classical_cost = {r['method']: r for r in measured}
            assert set(classical_cost) == {'raw', 'finite_self05', 'recursive_self05'}
        for r in classical:
            measured = classical_cost.get(r['method'])
            if measured:
                assert measured['online_history_storage_bytes'] == r['online_history_storage_bytes']
            classical_rows.append({**r, 'predictor_params': 4075,
                'protocol': 'R36 same R34 original/final4000 weights; 3 DEV videos x 2 noise conditions x 2 seeds; CPU',
                'flow_at_inference': r['method'] != 'raw', 'future_frames': 0,
                'extra_motion_teacher_params': 0, 'phone_latency_ms': None,
                'standalone_cpu_median_pipeline_ms_480x854': None if measured is None else measured['median_pipeline_ms_per_frame'],
                'standalone_cpu_min_run_average_ms': None if measured is None else measured['min_pipeline_ms_per_frame'],
                'standalone_cpu_max_run_average_ms': None if measured is None else measured['max_pipeline_ms_per_frame'],
                'gate_vs_finite_both_seeds': classical_audit['gate_passed_both_seeds'].get(r['method']),
                'scope': 'Fixed traditional control, not a novel method or calibrated covariance; no shared-stage speed rank.'})
        with (out/'classical_evidence.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(classical_rows[0]))
            writer.writeheader()
            writer.writerows(classical_rows)
        lines += ['', '## R36强原始底座上的传统递归对照', '',
                  '同R34原始final4000权重与同R30输入，两seed完整48行。有限post只使用上一raw，递归EMA另保存上一融合输出；三post共享同一own-raw半尺度FB运动，部署不需要第二CNN。', '',
                  '|方法|PSNR|SSIM|时序↓|边缘↓|在线历史MiB@480×854|同轮CPU中位ms/帧|',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for r in classical_rows:
            cpu = r['standalone_cpu_median_pipeline_ms_480x854']
            lines.append('|'+r['method']+'|'+'|'.join(f'{r[m]:.6f}' for m in ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae'])
                         +f"|{r['online_history_storage_bytes']/2**20:.3f}|"+('未测' if cpu is None else f'{cpu:.3f}')+'|')
        lines += ['', f"固定递归EMA相对有限post双seed联合筛查：{classical_audit['gate_passed_both_seeds']['recursive_self05']}；启发式adaptive：{classical_audit['gate_passed_both_seeds']['adaptive_recursive']}。递归是既有传统机制，不算创新。后续模型必须同时对照这个更强参考；不能只打败有限两帧post。", '',
                  '这些为三开发源上的固定控制。自适应方差未经校准，不等价复现BNLK/EMVD；共享质量阶段计时不作独立流水线速度。若CPU栏已填，来自R36独立raw/finite/recursive同轮18次流水线，固定单片前16帧，六个run均值取中位；不能和上方R34旧轮次混排，min–max见机器表。手机仍未测。[传统机器表](classical_evidence.csv)、[逐seed与限制](../exploration-r36-classical/tables.md)。']
    preserved_rows = []
    if (BASE/'report/exploration-r37/audit.json').exists():
        audit = read(Path('report/exploration-r37/audit.json'))
        assert audit['passed'] and audit['rows'] == 72 and audit['new_neural_sequences'] == 24
        values = read(Path('report/exploration-r37/domain_summary.json'))
        assert len(values) == 6
        for r in values:
            temporal = r['method'].endswith('alignment')
            preserved_rows.append({**r, 'protocol': 'R37 final4000, same frozen R34 anchor; R30 3 DEV videos x 2 noise conditions x 2 seeds; common CPU',
                 'fixed_motion_teacher_params': 0, 'future_frames': 0,
                 'external_FB_at_inference': not temporal and r['method'] != 'raw',
                 'learned_sampling_at_inference': temporal,
                 'primary_gate_both_seeds_both_references': audit['primary_passed_both_seeds_both_references'].get(r['method']),
                 'phone_latency_ms': None,
                 'scope': 'Fixed negative candidate; finite and recursive strong references preserved, no new candidate speed claim.'})
        with (out/'preserved_evidence.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(preserved_rows[0]))
            writer.writeheader()
            writer.writerows(preserved_rows)
        lines += ['', '## R37保留强空间映射的全分辨率状态：完整负结果', '',
                  '冻结同一R34原始骨干、仅训练512时间参数；final4000两个seed完整72行，含48行同anchor强传统参考复用。current显式返回不可变原骨干。', '',
                  '|方法|PSNR|SSIM|时序↓|边缘↓|在线历史MiB@480×854|',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for r in preserved_rows:
            lines.append('|'+r['method']+'|'+'|'.join(f'{r[m]:.6f}' for m in ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae'])
                         +f"|{r['online_history_storage_bytes']/2**20:.3f}|")
        lines += ['', '两个时间控制对finite/recursive的双seed主要门槛全部False，已停止该设计的局部网格。缓存减少是真实取舍，不能充当质量联合胜出；未为失败候选追加手机/CPU成本调优。[机器表](preserved_evidence.csv)、[全条件配对](../exploration-r37/tables.md)、[止损决策](../../docs/exploration-r37-decision.md)。']
    transfer_rows = []
    if (BASE/'report/exploration-r39-anchor-transfer/audit.json').exists():
        audit = read(Path('report/exploration-r39-anchor-transfer/audit.json'))
        assert audit['passed'] and audit['rows'] == 60 and audit['neural_sequences'] == 20
        transfer_rows = read(Path('report/exploration-r39-anchor-transfer/domain_summary.json'))
        assert len(transfer_rows) == 9
        with (out/'paired_real_anchor_transfer.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(transfer_rows[0]))
            writer.writeheader()
            writer.writerows(transfer_rows)
        lines += ['', '## R39固定空间底座的既有配对实拍DEV转移', '',
                  '四个既有白天ND实拍native crop，各16帧；全部同CPU2，官方一份权重、R26/R34各两个final4000 seed。训练方案不同，不能归于单个数据/优化因素；与R30合成域分开，不是自然夜景或独立最终确认。', '',
                  '|底座|输出|PSNR|SSIM|时序↓|边缘↓|', '|---|---|---:|---:|---:|---:|']
        for r in transfer_rows:
            lines.append('|'+r['anchor_family']+'|'+r['method']+'|'+'|'.join(f'{r[m]:.6f}' for m in ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae'])+'|')
        lines += ['', '[完整共同CPU转移表](../exploration-r39-anchor-transfer/tables.md)、[机器表](paired_real_anchor_transfer.csv)。R34合成DEV较强，配对实拍却明显弱于R26；后续必须处理跨域映射及数据假设，不能只优化合成均值。']
    phone_reference_complete = False
    if (BASE/'report/exploration-r38-phone/audit.json').exists():
        phone_audit = read(Path('report/exploration-r38-phone/audit.json'))
        assert phone_audit['passed'] and phone_audit['reference_rows'] == 27
        phone_reference_complete = True
        lines += ['', '## R38真实手机输入上的参考检查', '',
                  '三设备各64帧、官方及R34双seed底座的raw/finite/recursive共27输出序列已完成。固定0/32/63接触表与全部64帧预览均保留，出现明显截断/色偏；没有同步GT，不提供伪PSNR/SSIM或真实闪烁排名。它们是参考行为诊断，不是R37候选结果或手机时延。[固定预览与限制](../exploration-r38-phone/README.md)。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    (out/'sources.json').write_text(json.dumps({'sources': sources, 'rows': len(rows),
        'external_data_prepared': {'DAVIS_source_videos': 3, 'DAVIS_synthetic_conditions': 6,
                                   'real_phone_videos': 3, 'phone_reference_GT_available': False,
                                   'frames_per_clip': 64, 'method_results_in_this_table': False},
        'R28_screening_status': 'failed; route frozen as primary method, not universal impossibility',
        'R30_external_pilot_status': 'complete312 rows, common CPU; separate from native CUDA table; candidate screening failed',
        'external_same_base_rows': len(external_rows),
        'R34_dynamic_rows': len(dynamic_rows),
        'R34_quality_complete': dynamic_audit is not None,
        'R34_complete_CPU_cost': dynamic_cost_audit is not None,
        'R36_classical_rows': len(classical_rows),
        'R37_preserved_rows': len(preserved_rows),
        'R38_real_phone_reference_complete': phone_reference_complete,
        'R39_paired_real_anchor_rows': len(transfer_rows),
        'FastDVDnet_documented_R30_DEV_overlap': prior_overlap['overlap_R30_DEV'],
        'heldout_metadata_only': {'video_count': heldout_metadata['video_count'], 'future_frames': heldout_metadata['future_total_frames'],
                                  'JPEG_members_read': heldout_metadata['JPEG_members_read']},
        'scope': 'Evidence draft only. Missing values are explicit; no new inference, training or phone timing.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
