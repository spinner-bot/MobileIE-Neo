"""Start a research command only after two idle-GPU samples; keep an audit."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def sample(index):
    query = subprocess.check_output(["nvidia-smi", f"--id={index}",
        "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip().split(",")
    gpu, uuid, memory, utilization = [s.strip() for s in query]
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                                         "--format=csv,noheader,nounits"], text=True)
    active = [line.strip() for line in processes.splitlines() if uuid in line]
    return {"gpu": int(gpu), "uuid": uuid, "memory_mib": int(memory), "utilization_percent": int(utilization),
            "compute_processes": active, "time": datetime.datetime.now().astimezone().isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("command required after --")
    record = Path(args.record)
    record.parent.mkdir(parents=True, exist_ok=True)
    if record.exists():
        raise FileExistsError(record)
    lock_path = Path(".cache") / f"gpu-{args.gpu}.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        samples = [sample(args.gpu)]
        time.sleep(1)
        samples.append(sample(args.gpu))
        allowed = all(not s["compute_processes"] and s["memory_mib"] <= 64 and s["utilization_percent"] <= 5 for s in samples)
        root = Path(__file__).resolve().parents[3]
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in Path(__file__).resolve().parent.glob("*.py")}
        audit = {"samples": samples, "allowed": allowed, "command": command,
                 "repository_revision_at_launch": revision, "script_sha256_at_launch": hashes,
                 "scope": "two preflight samples and project-local lock; not a reservation system shared with other users"}
        record.write_text(json.dumps(audit, indent=2) + "\n")
        if not allowed:
            raise SystemExit("GPU has insufficient idle headroom; command not launched")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        start = time.perf_counter()
        result = subprocess.run(command, env=env)
        audit.update(returncode=result.returncode, process_wall_seconds=time.perf_counter()-start,
                     ended=datetime.datetime.now().astimezone().isoformat())
        record.write_text(json.dumps(audit, indent=2) + "\n")
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
