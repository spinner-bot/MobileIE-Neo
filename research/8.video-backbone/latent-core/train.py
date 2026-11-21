"""Fixed L1 learning; bind UUID and CuBLAS environment before importing Torch."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import copy
import math
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
parser.add_argument("--attempt", type=int, required=True)
parser.add_argument("--qualify", action="store_true")
parser.add_argument("--resume", type=Path)
ARGS = parser.parse_args()
from prebind_shared_gpu import actual_device, require_initial_binding
BOUND = require_initial_binding()
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "2"

import cv2
import numpy as np
import torch
from training_io import BASE, append, digest, identity, owned_cache, read, stamp, verify_training, write
from mixed_anchor_r40 import MixedPool
from motion_transport_r74 import load_spatial
from core import models, optimizer, sequence
from gpu import acquire_gpu
from shared_training_schedule import ResourcePressure


def available_kib():
    return next(int(r.split()[1]) for r in Path("/proc/meminfo").read_text().splitlines()
                if r.startswith("MemAvailable:"))


def main(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    os.nice(5)
    frozen = verify_training()
    assert read(BASE / "process/preflight001/qualification.json")["passed"]
    assert not torch.cuda.is_initialized()
    if args.qualify:
        assert args.seed == 0
    else:
        qualifications = sorted((BASE / "process").glob("cuda-qualification*/qualification.json"))
        assert qualifications and read(qualifications[-1])["passed"]
        assert read(qualifications[-1].parent / "config.json")["source_freeze"] == frozen
    limit = 2 if args.qualify else 8000
    label = f"cuda-qualification{args.attempt:03d}" if args.qualify else f"train-seed{args.seed}-attempt{args.attempt:03d}"
    out, assets = BASE / "process" / label, owned_cache(label)
    out.mkdir(parents=True, exist_ok=False)
    assets.mkdir(parents=True, exist_ok=False)
    began, complete, paused = time.perf_counter(), False, False
    counts = dict(common_steps=0, optimizer_updates=0, student_images=0, F0_CPU_anchor_images=0)
    lock = schedule = latest = optimizers = inherited = None
    initial_errors = {}
    try:
        if available_kib() < 8 * 1024**2:
            raise ResourcePressure("Host preparation reserve below8GiB")
        pool = MixedPool()
        original, provenance = load_spatial(args.seed)
        students = models(original, args.seed)
        if args.resume:
            assert args.resume.resolve().is_relative_to("/data/Users/zrh/data/MobileIE-Neo")
            inherited = torch.load(args.resume, map_location="cpu", weights_only=False)
            assert inherited["resource_pause"] and inherited["source_freeze"] == frozen
            assert (inherited["seed"], inherited["qualification"]) == (args.seed, args.qualify)
            counts = copy.deepcopy(inherited["counts"])
            initial_errors = inherited["initial_errors"]
            for arm, model in students.items():
                model.load_state_dict(inherited["models"][arm], strict=True)
            latest = dict(path=str(args.resume.relative_to(ROOT)), sha256=digest(args.resume), inherited=True,
                          counts=copy.deepcopy(counts))
        prepared = 0 if args.qualify else min(counts["common_steps"], limit - 1)
        first = pool.batch(args.seed, 8000 + prepared)
        anchor = None
        if args.qualify and counts["common_steps"] == 0:
            with torch.inference_mode():
                anchor = torch.stack([original(x) for x in first[0].unbind(1)], 1).clone()
            counts["F0_CPU_anchor_images"] += 16
        del original
        write(out / "config.json", dict(at=stamp(), process=identity(), source_freeze=frozen, seed=args.seed,
              qualification=args.qualify, inherited_counts=copy.deepcopy(counts), common_step_budget=limit,
              data=pool.info, F0_source=provenance, parameters=12431, prepared_data_step=8000 + prepared,
              first_windows=first[-1], CUBLAS_WORKSPACE_CONFIG=os.environ["CUBLAS_WORKSPACE_CONFIG"],
              qualified_weight_or_Adam_inheritance=False, formal_teacher_images=0,
              resume=None if args.resume is None else dict(path=str(args.resume.relative_to(ROOT)), sha256=digest(args.resume)),
              wall_clock_cutoff=False, automatic_scientific_retry=False))
        lock, schedule, _ = acquire_gpu(BOUND, out)
        for model in students.values():
            model.to("cuda").train()
        optimizers = {arm: optimizer(model) for arm, model in students.items()}
        if inherited is not None:
            for arm, opt in optimizers.items():
                opt.load_state_dict(inherited["optimizers"][arm])
            torch.set_rng_state(inherited["torch_rng"])
            torch.cuda.set_rng_state_all(inherited["cuda_rng"])
            random.setstate(inherited["python_rng"])
            np.random.set_state(inherited["numpy_rng"])
        torch.cuda.reset_peak_memory_stats()

        def checkpoint(resource_pause=False):
            nonlocal latest
            path = assets / f'step{counts["common_steps"]:05d}{"-resource" if resource_pause else ""}.pt'
            temporary = path.with_suffix(".tmp")
            torch.save(dict(seed=args.seed, qualification=args.qualify, resource_pause=resource_pause,
                       source_freeze=frozen, counts=copy.deepcopy(counts), initial_errors=initial_errors,
                       models={k: m.state_dict() for k, m in students.items()},
                       optimizers={k: o.state_dict() for k, o in optimizers.items()},
                       torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                       python_rng=random.getstate(), numpy_rng=np.random.get_state()), temporary)
            temporary.replace(path)
            latest = dict(path=str(path.relative_to(ROOT)), sha256=digest(path), counts=copy.deepcopy(counts),
                          seed=args.seed, qualification=args.qualify, resource_pause=resource_pause)
            write(out / (path.stem + ".json"), latest)

        for step in range(counts["common_steps"], limit):
            if available_kib() < 4 * 1024**2:
                raise ResourcePressure("Host running reserve below4GiB")
            low, target, metadata = first if args.qualify or step == prepared else pool.batch(args.seed, 8000 + step)
            schedule.check_memory()
            low, target = low.to("cuda"), target.to("cuda")
            multiplier = .01 + .495 * (1 + math.cos(math.pi * step / 7999))
            torch.cuda.synchronize()
            work_began, details = time.perf_counter(), {}
            for arm, model in students.items():
                opt = optimizers[arm]
                opt.param_groups[0]["lr"] = 1e-5 * multiplier
                opt.param_groups[1]["lr"] = 1e-3 * multiplier
                opt.zero_grad(set_to_none=True)
                raw, state = sequence(model, low, "current" if arm == "spatial" else "causal")
                counts["student_images"] += 16
                assert torch.isfinite(raw).all() and torch.isfinite(state).all()
                assert float(state.detach().abs().max()) <= 1 + 1e-6
                if args.qualify and step == 0:
                    initial_errors[arm] = float((raw.detach().cpu() - anchor).abs().max())
                    assert initial_errors[arm] <= 1e-4
                loss = (raw - target).square().mean()
                assert torch.isfinite(loss)
                loss.backward()
                parameters = tuple(model.parameters())
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
                norms = {}
                if args.qualify:
                    norms = {name: sum(float(p.grad.square().sum()) for p in model.get_submodule(name).parameters())**.5
                             for name in ["backbone.head.block1", "project", "gates", "candidate", "readout"]}
                    if step == 1:
                        assert all(v > 0 for v in norms.values())
                norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1))
                assert math.isfinite(norm)
                opt.step()
                counts["optimizer_updates"] += 1
                assert all(torch.isfinite(p).all() for p in parameters)
                details[arm] = dict(GT_MSE=float(loss.detach()), gradient_norm_before_clip=norm,
                                    qualification_path_gradient_norms=norms)
            torch.cuda.synchronize()
            work = time.perf_counter() - work_began
            counts["common_steps"] = step + 1
            data_step = 8000 + (prepared if args.qualify else step)
            append(out / "training.jsonl", dict(at=stamp(), data_step=data_step, lr_multiplier=multiplier,
                   controls=details, synchronized_student_seconds=work, **counts))
            append(out / "windows.jsonl", dict(common_step=step + 1, data_step=data_step, windows=metadata))
            if not args.qualify and (step + 1) % 1000 == 0 and step + 1 < limit:
                checkpoint()
            if (step + 1) % 100 == 0 or args.qualify:
                print(dict(seed=args.seed, **counts), flush=True)
            if step + 1 < limit:
                schedule.yield_after_work(work)
            else:
                schedule.work_seconds += work
        assert counts["common_steps"] == limit and counts["optimizer_updates"] == 2 * limit
        assert counts["student_images"] == 32 * limit
        verify_training()
        if args.qualify:
            device = actual_device(BOUND)
            own = [p for g in device["sample"]["GPUs"] for p in g["processes"]
                   if int(p.split(",")[1].strip()) == os.getpid()]
            assert device["matched"] and len(own) == 1 and int(own[0].split(",")[2].strip()) <= 4096
            assert torch.cuda.max_memory_allocated() <= 2 * 1024**3
            write(out / "qualification.json", dict(at=stamp(), passed=True, all_updates_discarded=True,
                  no_qualification_weights_exported=True, counts=counts, initial_F0_CPU_difference=initial_errors,
                  own_final_NVML_mib=int(own[0].split(",")[2].strip()), actual_UUID_matched=True,
                  maximum_allocated_bytes=torch.cuda.max_memory_allocated(), maximum_reserved_bytes=torch.cuda.max_memory_reserved()))
        else:
            for arm, model in students.items():
                path = assets / (arm + "-final.pt")
                torch.save(model.state_dict(), path)
                exported = torch.load(path, map_location="cpu", weights_only=True)
                assert all(torch.isfinite(v).all() for v in exported.values())
                write(out / (arm + "-final.json"), dict(at=stamp(), seed=args.seed, arm=arm, qualification=False,
                      unique_final=True, updates=8000, parameters=12431, all_reloaded_finite=True,
                      path=str(path.relative_to(ROOT)), sha256=digest(path)))
        complete = True
    except ResourcePressure as error:
        paused = True
        if optimizers is not None and counts["common_steps"] > (0 if inherited is None else inherited["counts"]["common_steps"]):
            checkpoint(True)
        write(out / "resource-pause.json", dict(at=stamp(), reason=str(error), resume=latest))
        raise SystemExit(75)
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        final_device, own_memory = None, None
        if torch.cuda.is_initialized():
            try:
                final_device = actual_device(BOUND)
                own = [p for g in final_device["sample"]["GPUs"] for p in g["processes"]
                       if int(p.split(",")[1].strip()) == os.getpid()]
                if len(own) == 1:
                    own_memory = int(own[0].split(",")[2].strip())
            except Exception as error:
                final_device = dict(observation_error=str(error))
        write(out / "runtime.json", dict(at=stamp(), process=identity(), seed=args.seed, qualification=args.qualify,
              completed_normally=complete, resource_pause=paused, counts=counts, latest_checkpoint=latest,
              count_scope="Actual completed forwards/updates; common_steps counts complete two-arm steps only",
              seconds=time.perf_counter() - began, CUDA_initialized=torch.cuda.is_initialized(),
              GPU=None if schedule is None else schedule.summary(), final_actual_device=final_device,
              own_final_NVML_mib=own_memory, own_NVML_scope="End sample, not lifetime own process peak",
              maximum_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
              maximum_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0))
        if lock:
            lock.close()


if __name__ == "__main__":
    main(ARGS)
