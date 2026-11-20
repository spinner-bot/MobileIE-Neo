"""Registered O1 CPU implementation/gradient qualification, all updates discarded."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import time
import traceback
from study_io import BASE, write, append, stamp, verify, identity
from mixed_anchor_r40 import MixedPool
from motion_transport_r74 import load_spatial
from video_metrics import warp_previous
from core import models, Streaming, mix
from objectives import prepare_teacher, teacher_linear, losses, optimizer
import torch
import cv2
import numpy as np

OUT = BASE / 'process/preflight001'


def check_operator(teacher):
    errors = []
    for probe, expected in zip(teacher['probes'], teacher['responses']):
        for i, (state, target) in enumerate(zip(probe, expected)):
            flow = teacher['flow'][i].numpy().transpose(1, 2, 0)
            warped, bounds = warp_previous(state.numpy().transpose(1, 2, 0), flow)
            original = .5*(bounds & teacher['mask'][i, 0].numpy())[..., None]*warped
            error = float(np.max(np.abs(original-target.numpy().transpose(1, 2, 0))))
            assert error <= 1e-5
            errors.append(error)
    zero = teacher_linear(torch.zeros_like(teacher['probes'][0]), teacher['flow'], teacher['mask'])
    assert torch.count_nonzero(zero) == 0
    return dict(passed=True, OpenCV_comparisons=len(errors), max_difference=max(errors),
                zero_states=len(zero), zero_response_exact=True,
                accepted_fraction=float(teacher['mask'].float().mean()))


def check_stream(model, images):
    model.eval().requires_grad_(False)
    values, encoded = [], []
    with torch.inference_mode():
        for image in images.unbind(1):
            raw, desc = model.encode(image)
            values.append(raw.clamp(0, 1)); encoded.append(desc)
        wanted, changed = [values[0]], None
        for t in range(1, 4):
            fields = model.fields(encoded[t-1], encoded[t], images.shape[-2:])
            next_value = mix(values[t], wanted[-1], fields).clamp(0, 1)
            if t == 1:
                substitute = mix(values[t], values[t], fields).clamp(0, 1)
                changed = float((next_value-substitute).abs().max())
            wanted.append(next_value)
        stream = Streaming(model)
        observed = [stream.step(image) for image in images.unbind(1)]
        error = float((torch.cat(wanted)-torch.cat(observed)).abs().max())
        assert error <= 2e-6 and changed > 1e-8
        assert torch.equal(observed[0], values[0])
        height, width = images.shape[-2:]
        expected_bytes = 4*(3*height*width + 8*((height+3)//4)*((width+3)//4))
        assert stream.state_bytes() == expected_bytes
        buffers = (stream.previous_descriptor, stream.previous_fused)
        assert all(v.is_contiguous() and v.untyped_storage().nbytes() == v.numel()*v.element_size() for v in buffers)
        assert len({v.untyped_storage().data_ptr() for v in buffers}) == 2
        assert images.untyped_storage().data_ptr() not in {v.untyped_storage().data_ptr() for v in buffers}
    return dict(passed=True, stream_max_difference=error, first_frame_exact=True,
        history_substitution_max_difference=changed, stored_bytes=stream.state_bytes(),
        compact_separate_buffers=True, extra_backbone_images=8, field_pairs=6)


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); cv2.setNumThreads(1); os.nice(5)
    frozen = verify()
    OUT.mkdir(parents=True, exist_ok=False)
    started, complete = time.perf_counter(), False
    counts = dict(teacher_images=0, teacher_FB_pairs=0, teacher_probe_states=0,
        teacher_state_actions=0, teacher_zero_state_actions=0, OpenCV_probe_comparisons=0,
        student_learning_images=0, student_stream_images=0, student_learning_field_pairs=0,
        student_stream_field_pairs=0, student_response_training_actions=0, discarded_optimizer_updates=0)
    write(OUT / 'config.json', dict(at=stamp(), process=identity(), source_freeze=frozen,
        device='cpu', source_seed_step=[[0, 8000], [1, 8000]], B=4, T=4, crop=128,
        formal_training=False, all_updates_discarded=True, new_DEV=False, independent_final_opened=False,
        inference_uses_GT=False, GT_not_a_qualification_target=True, wall_clock_cutoff=False))
    try:
        pool = MixedPool()
        write(OUT / 'data.json', pool.info)
        current_errors, operator_checks, stream_checks = [], [], []
        for seed in (0, 1):
            low, unused_GT, metadata = pool.batch(seed, 8000)
            del unused_GT
            append(OUT / 'windows.jsonl', dict(seed=seed, data_step=8000, windows=metadata))
            teacher_net, provenance = load_spatial(seed)
            teacher_weights = {k: v.clone() for k, v in teacher_net.state_dict().items()}
            with torch.inference_mode():
                target = teacher_net(low.flatten(0, 1)).unflatten(0, (4, 4)).clone()
            counts['teacher_images'] += 16
            teacher = prepare_teacher(target, seed)
            counts['teacher_FB_pairs'] += 12; counts['teacher_probe_states'] += 24
            counts['teacher_state_actions'] += 24
            checked = check_operator(teacher)
            operator_checks.append(dict(seed=seed, **checked))
            counts['teacher_zero_state_actions'] += checked['zero_states']
            counts['OpenCV_probe_comparisons'] += checked['OpenCV_comparisons']
            append(OUT / 'teacher.jsonl', dict(seed=seed, source=provenance, operator=checked,
                probe_kernels=[5, 17], random_states_are_not_white=True))
            students = models(teacher_net, seed)
            assert all(torch.equal(v, students['response'].state_dict()[k]) for k, v in students['coordinate'].state_dict().items())
            for arm, model in students.items():
                model.train(); opt = optimizer(model)
                for step in range(2):
                    opt.zero_grad(set_to_none=True)
                    raw, desc, fields = model.batch(low)
                    counts['student_learning_images'] += 16; counts['student_learning_field_pairs'] += 12
                    assert sum(p.numel() for p in model.parameters()) == 5049
                    if step == 0:
                        error = float((raw-target).abs().max())
                        assert error <= 2e-6
                        current_errors.append(dict(seed=seed, arm=arm, max_difference=error))
                    loss, details = losses(arm, raw, target, fields, teacher)
                    if arm == 'response': counts['student_response_training_actions'] += 24
                    assert torch.isfinite(loss) and torch.isfinite(raw).all()
                    loss.backward()
                    parameters = tuple(model.parameters())
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
                    shared_norm = sum(float(p.grad.square().sum()) for p in model.backbone.head.block1.parameters())**.5
                    descriptor_norm = sum(float(p.grad.square().sum()) for p in model.descriptor.parameters())**.5
                    assert shared_norm > 0 and descriptor_norm > 0
                    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1))
                    opt.step(); counts['discarded_optimizer_updates'] += 1
                    assert all(torch.isfinite(p).all() for p in parameters)
                    append(OUT / 'discarded-learning.jsonl', dict(seed=seed, arm=arm, discarded_update=step+1,
                        **details, shared_head_gradient_norm=shared_norm, descriptor_gradient_norm=descriptor_norm,
                        total_gradient_norm_before_clip=norm, parameters=5049))
                checked = check_stream(model, low[:1])
                counts['student_stream_images'] += checked['extra_backbone_images']
                counts['student_stream_field_pairs'] += checked['field_pairs']
                stream_checks.append(dict(seed=seed, arm=arm, **checked))
                print(dict(seed=seed, arm=arm, **counts), flush=True)
            assert all(torch.equal(v, teacher_weights[k]) for k, v in teacher_net.state_dict().items())
        assert counts == dict(teacher_images=32, teacher_FB_pairs=24, teacher_probe_states=48,
            teacher_state_actions=48, teacher_zero_state_actions=24, OpenCV_probe_comparisons=48,
            student_learning_images=128, student_stream_images=32, student_learning_field_pairs=96,
            student_stream_field_pairs=24, student_response_training_actions=96, discarded_optimizer_updates=8)
        write(OUT / 'qualification.json', dict(at=stamp(), passed=True, all_updates_discarded=True,
            no_qualification_weights_exported=True, current_path=current_errors,
            operator=operator_checks, stream=stream_checks, counts=counts,
            parameters=5049, additional_parameters=974, analytical_1080p_Conv_GMAC=(3887.25*1080*1920+144)/1e9,
            analytical_1080p_persistent_state_bytes=4*(3*1080*1920+8*270*480),
            allowed_formal_protocol_design=True, established_video_method=False))
        verify(); assert not torch.cuda.is_initialized(); complete = True
    except BaseException:
        (OUT / 'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(OUT / 'runtime.json', dict(at=stamp(), process=identity(), completed_normally=complete,
            seconds=time.perf_counter()-started, CUDA_initialized=torch.cuda.is_initialized(), **counts))


if __name__ == '__main__':
    main()
