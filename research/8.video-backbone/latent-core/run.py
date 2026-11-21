"""Qualify once, run independent L1 seeds on authorized GPUs1/2, then summarize.

Resource exits resume the same algorithmic budget. Other failures are retained;
one failed seed never causes the other seed's live worker to be killed/restarted.
"""
import argparse
import itertools
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from training_io import BASE, ROOT, append, digest, identity, owned_cache, read, stamp, verify_training, write
sys.path.insert(0, str(ROOT / "scripts"))
from shared_gpu_admission import assess, sustained_samples


def child_identity(pid):
    value = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=pid, start_ticks=int(value[19]), state=value[0])


def main(seed):
    frozen = verify_training()
    label = "chain001" if seed is None else f"seed{seed}-chain001"
    out = BASE / "process" / label
    out.mkdir(parents=True, exist_ok=False)
    logs = owned_cache(label)
    logs.mkdir(parents=True, exist_ok=False)
    python = str(ROOT / "scripts/project-python")
    began, complete = time.perf_counter(), False
    state = dict(at=stamp(), process=identity(), source_freeze=frozen, stage="initial", children={})

    def start_child(command, name, cpu=False):
        env = os.environ.copy()
        if cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        with (logs / (name + ".log")).open("x") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
        observed = child_identity(child.pid)
        state["children"][name] = observed
        state.update(at=stamp(), stage=name)
        write(out / "state.json", state)
        append(out / "events.jsonl", dict(at=stamp(), action="start", label=name, child=observed, command=command))
        return child

    def wait_child(child, name):
        code = child.wait()
        state["children"][name]["exit_code"] = code
        state.update(at=stamp())
        write(out / "state.json", state)
        append(out / "events.jsonl", dict(at=stamp(), action="exit", label=name, code=code))
        return code

    def run_child(command, name, cpu=False):
        return wait_child(start_child(command, name, cpu), name)

    def training(model_seed, qualification):
        prefix = "cuda-qualification" if qualification else f"train-seed{model_seed}-attempt"
        desired_gpu = 1 if qualification or model_seed == 0 else 2
        resume = None
        for attempt in itertools.count(1):
            name = f"{prefix}{attempt:03d}"
            samples = sustained_samples()
            assessment = assess(samples)
            selected = next(r for r in assessment["candidates"] if r["index"] == desired_gpu)
            write(out / (name + "-selection.json"), dict(at=stamp(), samples=samples, assessment=assessment,
                  desired_gpu=desired_gpu, selected=selected if selected["allowed"] else None,
                  scope="Read-only current headroom, not a reservation; CPU child must recheck and lock"))
            if not selected["allowed"]:
                state.update(at=stamp(), stage="resource_wait", pending=name, desired_gpu=desired_gpu)
                write(out / "state.json", state)
                time.sleep(60)
                continue
            command = [python, str(BASE / "train.py"), "--seed", str(model_seed), "--attempt", str(attempt)]
            if qualification:
                command += ["--qualify"]
            if resume:
                command += ["--resume", str(resume)]
            wrapped = [python, str(ROOT / "scripts/prebind_shared_gpu.py"), "--gpu", str(desired_gpu),
                       "--record", str(out / (name + "-guard.json")), "--", *command]
            code = run_child(wrapped, name)
            if code == 75:
                runtime = BASE / "process" / name / "runtime.json"
                if runtime.exists():
                    result = read(runtime)
                    assert result["resource_pause"] and not result["completed_normally"]
                    marker = read(runtime.parent / "resource-pause.json")["resume"]
                    if marker is not None:
                        resume = ROOT / marker["path"]
                        assert resume.resolve().is_relative_to("/data/Users/zrh/data/MobileIE-Neo") and digest(resume) == marker["sha256"]
                time.sleep(60)
                continue
            if code != 0:
                raise RuntimeError(f"{name} exit {code}; non-resource failure is not retried")
            result = read(BASE / "process" / name / "runtime.json")
            limit = 2 if qualification else 8000
            assert result["completed_normally"] and result["counts"] == dict(common_steps=limit,
                   optimizer_updates=2 * limit, student_images=32 * limit, F0_CPU_anchor_images=16 if qualification else 0)
            if qualification:
                proof = read(BASE / "process" / name / "qualification.json")
                assert proof["passed"] and proof["all_updates_discarded"] and proof["no_qualification_weights_exported"]
                assert result["final_actual_device"]["matched"] and result["own_final_NVML_mib"] <= 4096
                assert result["maximum_allocated_bytes"] <= 2 * 1024**3
            return

    write(out / "config.json", dict(at=stamp(), process=identity(), source_freeze=frozen, model_seed=seed,
          GPU_group=[0, 1, 2, 3], preferred_seed_GPU={"0": 1, "1": 2}, maximum_parallel_training_workers=2,
          formal_optimizer_updates=32000 if seed is None else 16000,
          formal_student_images=512000 if seed is None else 256000,
          new_DEV=False, independent_final=False, wall_clock_cutoff=False, automatic_scientific_retry=False))
    try:
        if seed is None:
            training(0, True)
            children = {s: start_child([python, str(BASE / "run.py"), "--seed", str(s)], f"seed{s}-chain") for s in (0, 1)}
            codes = {s: wait_child(c, f"seed{s}-chain") for s, c in children.items()}
            assert codes == {0: 0, 1: 0}, f"Retained seed exits {codes}; no scientific retry"
            assert run_child([python, str(BASE / "summarize.py")], "summarize", cpu=True) == 0
        else:
            training(seed, False)
            assert run_child([python, str(BASE / "evaluate.py"), "--seed", str(seed)], f"held-seed{seed}", cpu=True) == 0
        verify_training()
        state.update(at=stamp(), stage="completed")
        write(out / "state.json", state)
        complete = True
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        write(out / "runtime.json", dict(at=stamp(), process=identity(), completed_normally=complete,
              seconds=time.perf_counter() - began, state=state))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1))
    main(parser.parse_args().seed)
