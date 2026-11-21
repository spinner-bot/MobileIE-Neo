"""Resume scheduling after outer143, reusing passed CUDA007 and unchanged seed drivers.

Launch this controller in its own process session with owned log files. This is
an execution recovery, not a qualification retry, scientific change or resume
from the discarded qualification weights.
"""
import os
from pathlib import Path
import subprocess
import time
import traceback
from training_io import BASE, ROOT, append, digest, identity, owned_cache, read, stamp, verify_training, write


def main():
    frozen = verify_training()
    execution = read(BASE / "process/execution-source-freeze001.json")
    assert all(digest(ROOT / p) == h for p, h in execution["source_sha256"].items())
    terminal = read(BASE / "process/chain001-terminal.json")
    assert terminal["outer_tool_exit_code"] == 143 and not terminal["formal_seed_drivers_or_workers_started"]
    for pid in (3843495, 3848340, 3848429):
        assert not Path(f"/proc/{pid}").exists(), "Original handle exists; investigate before recovering"
    qualification = BASE / "process/cuda-qualification007"
    proof, runtime = read(qualification / "qualification.json"), read(qualification / "runtime.json")
    assert proof["passed"] and proof["all_updates_discarded"] and proof["no_qualification_weights_exported"]
    assert runtime["completed_normally"] and runtime["final_actual_device"]["matched"]
    assert runtime["counts"] == dict(common_steps=2, optimizer_updates=4, student_images=64, F0_CPU_anchor_images=16)
    assert read(qualification / "config.json")["source_freeze"] == frozen
    assert proof["own_final_NVML_mib"] <= 4096 and proof["maximum_allocated_bytes"] <= 2*1024**3
    assert not list((BASE / "process").glob("train-seed*-attempt*"))
    assert not list((BASE / "process").glob("seed*-chain*"))
    out, logs = BASE / "process/chain002", owned_cache("chain002")
    out.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    python = str(ROOT / "scripts/project-python")
    began, complete = time.perf_counter(), False
    state = dict(at=stamp(), process=identity(), stage="initial", children={})
    write(out / "config.json", dict(at=stamp(), process=identity(), process_session=os.getsid(0),
          source_freeze=frozen, execution_source_freeze=execution, reused_passed_qualification="cuda-qualification007",
          qualification_marker_sha256=digest(qualification / "qualification.json"),
          qualification_rerun=False, qualification_weights_or_Adam_inherited=False,
          unchanged_original_seed_driver="run.py --seed", new_formal_optimizer_updates=32000,
          new_formal_student_images=512000, wall_clock_cutoff=False, new_DEV=False, independent_final=False))

    def start(command, name, cpu=False):
        env = os.environ.copy()
        if cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        with (logs / (name + ".log")).open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
        values = Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()
        state["children"][name] = dict(pid=child.pid, start_ticks=int(values[19]), state=values[0])
        state.update(at=stamp(), stage=name)
        write(out / "state.json", state)
        append(out / "events.jsonl", dict(at=stamp(), action="start", label=name,
               child=state["children"][name], command=command))
        return child

    def wait(child, name):
        code = child.wait()
        state["children"][name]["exit_code"] = code
        state.update(at=stamp())
        write(out / "state.json", state)
        append(out / "events.jsonl", dict(at=stamp(), action="exit", label=name, code=code))
        return code

    try:
        children = {s: start([python, str(BASE / "run.py"), "--seed", str(s)], f"seed{s}-chain") for s in (0, 1)}
        codes = {s: wait(c, f"seed{s}-chain") for s, c in children.items()}
        assert codes == {0: 0, 1: 0}, f"Retained seed exits {codes}; do not kill or restart a sibling"
        assert wait(start([python, str(BASE / "summarize.py")], "summarize", cpu=True), "summarize") == 0
        verify_training()
        state.update(at=stamp(), stage="completed")
        write(out / "state.json", state)
        complete = True
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        write(out / "runtime.json", dict(at=stamp(), process=identity(), completed_normally=complete,
              seconds=time.perf_counter()-began, state=state))


if __name__ == "__main__":
    main()
