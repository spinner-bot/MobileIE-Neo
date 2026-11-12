#!/usr/bin/env python3
"""Compare six Android CPU outputs to the unchanged official Python pipeline."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
MODEL_SHA = "2ea09b840634469eceab5263bf7b549c33fcd5df2a25701ce5fefc585402ad44"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--android-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("demo_official_wrapper", REPO / "demo/org/scripts/run_official_tflite.py")
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    wrapper.ensure_tflite_runtime()
    official = wrapper.load_official(REPO / "MobileIE/test_TFLite_RGB.py")
    model = REPO / "MobileIE/pretrain/lolv1.tflite"
    assert digest(model) == MODEL_SHA
    interpreter = official.load_model(str(model))
    inputs = args.output / "official-inputs"
    outputs = args.output / "official-outputs"
    inputs.mkdir(exist_ok=True)
    cases = sorted(args.android_dir.glob("case*-input.png"))
    assert len(cases) == 6, f"Expected six cases, got {len(cases)}"
    reference = {}
    for path in cases:
        assert Image.open(path).size == (600, 400)
        shutil.copyfile(path, inputs / path.name)
        value = official.preprocess_image(str(path))
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], value)
        interpreter.invoke()
        raw = interpreter.get_tensor(interpreter.get_output_details()[0]["index"]).copy()
        raw.astype("<f4").tofile(args.output / path.name.replace("-input.png", "-reference.f32"))
        reference[path.name] = raw
    # Exercise the official image postprocessing and saving code too (six more calls).
    official.inference_and_save_results(interpreter, str(inputs), str(outputs))
    records = []
    for path in cases:
        android = np.fromfile(args.android_dir / path.name.replace("-input.png", "-android.f32"), dtype="<f4").reshape(1, 400, 600, 3)
        raw = reference[path.name]
        finite = bool(np.isfinite(android).all() and np.isfinite(raw).all())
        error = np.abs(android.astype(np.float64) - raw.astype(np.float64))
        phone = np.asarray(Image.open(args.android_dir / path.name.replace("-input.png", "-android.png")).convert("RGB"), dtype=np.int16)
        saved = np.asarray(Image.open(outputs / path.name).convert("RGB"), dtype=np.int16)
        pixels = np.abs(phone - saved)
        maximum = float(error.max())
        byte_maximum = int(pixels.max())
        records.append({
            "case": path.stem.replace("-input", ""), "input_file_sha256": digest(path),
            "input_rgb_sha256": hashlib.sha256(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).tobytes()).hexdigest(),
            "finite": finite, "raw_max_abs": maximum, "raw_mean_abs": float(error.mean()),
            "uint8_max_abs": byte_maximum, "uint8_differing_channels": int(np.count_nonzero(pixels)),
            "pass": finite and maximum <= 1e-4 and byte_maximum <= 1,
        })
    report = {
        "model_sha256": MODEL_SHA,
        "official_script_sha256": digest(REPO / "MobileIE/test_TFLite_RGB.py"),
        "android_runtime": "LiteRT 1.4.2, CPU builtin kernels, four threads, x86_64 emulator",
        "reference_runtime": "ai-edge-litert 2.3.0, BUILTIN_WITHOUT_DEFAULT_DELEGATES",
        "reference_inferences": 12, "android_inferences": 6,
        "registered_gates": {"raw_max_abs": 1e-4, "uint8_max_abs": 1, "all_finite": True},
        "cases": records, "all_pass": all(r["pass"] for r in records),
        "scope": "Runtime conformance on fixed inputs, not image-quality superiority or real-device performance",
    }
    target = args.output / "numerical-report.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_pass"]:
        raise SystemExit("Numerical gate failed; do not relax thresholds")


if __name__ == "__main__":
    main()
