"""Same-device eager FP32 architecture cost, not a mobile speed claim."""
import argparse
import json
import statistics
from pathlib import Path

import torch

from bounded_fst import apply_bounded
from explore_temporal_controls import ROOT, MobileIELLENetS, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = Path(args.out)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    limits = json.loads((ROOT/"research/4.temporal-consistency/results/exploration-r2-training-bounded/limits.json").read_text())
    models = {"ordinary": MobileIELLENetS(12).eval().cuda(),
              "bounded": apply_bounded(MobileIELLENetS(12).eval().cuda(), limits)}
    for model in models.values():
        model.load_state_dict(torch.load(ROOT/"MobileIE/pretrain/lolv1_best_slim.pkl", map_location="cuda", weights_only=True), strict=False)
    with torch.no_grad():
        models["bounded"].head.gamma.fill_(.2)
        models["bounded"].body.gamma.fill_(.2)
    value = torch.rand(1, 3, 400, 600, device="cuda")
    records = {name: {"latencies_ms": [], "peak_incremental_bytes": []} for name in models}
    with torch.inference_mode():
        for repeat in range(3):
            for name in (["ordinary", "bounded"] if repeat%2 == 0 else ["bounded", "ordinary"]):
                model = models[name]
                for _ in range(50):
                    model(value)
                torch.cuda.synchronize()
                initial = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                events = []
                for _ in range(200):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    model(value)
                    end.record()
                    events.append((begin, end))
                torch.cuda.synchronize()
                records[name]["latencies_ms"].extend(a.elapsed_time(b) for a,b in events)
                records[name]["peak_incremental_bytes"].append(torch.cuda.max_memory_allocated()-initial)
    for name, result in records.items():
        result["median_ms"] = statistics.median(result["latencies_ms"])
        result["parameters"] = sum(p.numel() for p in models[name].parameters())
    save_json(path, {"shape": [1, 3, 400, 600], "dtype": "float32", "backend": "PyTorch eager",
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(),
        "conv_tf32_allowed": torch.backends.cudnn.allow_tf32, "warmup_per_block": 50,
        "timed_per_block": 200, "blocks_per_model": 3, "models": records,
        "latency_ratio": records["bounded"]["median_ms"]/records["ordinary"]["median_ms"]})


if __name__ == "__main__":
    main()
