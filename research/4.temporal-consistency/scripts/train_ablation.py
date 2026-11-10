"""Train one MobileIE temporal-consistency ablation with a reproducible setup.

This runner is intentionally separate from the upstream ``MobileIE/main.py``:
it reuses the upstream model and loss, but avoids importing the upstream data
package (which pulls in torchvision) and does not require OpenCV. It does not
change the upstream training defaults.

Example (run from the repository root):
    python research/4.temporal-consistency/scripts/train_ablation.py \
      --variant full --seed 1 --train-low /path/our485/low \
      --train-high /path/our485/high --valid-low /path/eval15/low \
      --valid-high /path/eval15/high --output-root /path/experiments/ablation
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
MOBILEIE = ROOT / "MobileIE"
sys.path.insert(0, str(MOBILEIE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loss import import_loss  # noqa: E402
from model_variants import build_variant  # noqa: E402


VARIANTS = ("full", "no_global_att", "no_local_att", "no_att", "no_fst", "no_mbr")


class PairedImages(Dataset):
    """Sorted paired RGB folders with deterministic per-epoch random crops."""

    def __init__(self, low_dir, high_dir, patch_size=0, seed=0, training=False):
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)
        self.patch_size = int(patch_size)
        self.seed = int(seed)
        self.training = bool(training)
        if not self.low_dir.is_dir() or not self.high_dir.is_dir():
            raise FileNotFoundError(f"paired folders not found: {self.low_dir}, {self.high_dir}")
        low = {p.name for p in self.low_dir.iterdir() if p.is_file()}
        high = {p.name for p in self.high_dir.iterdir() if p.is_file()}
        if low != high:
            raise ValueError(
                f"input/target filenames differ; missing targets={sorted(low-high)[:5]}, "
                f"missing inputs={sorted(high-low)[:5]}"
            )
        self.names = sorted(low)
        if not self.names:
            raise ValueError(f"no paired images in {self.low_dir} and {self.high_dir}")
        self.epoch = 0

    def __len__(self):
        return len(self.names)

    @staticmethod
    def _tensor(path):
        image = Image.open(path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array.transpose(2, 0, 1).copy())

    def __getitem__(self, index):
        name = self.names[index]
        low = self._tensor(self.low_dir / name)
        high = self._tensor(self.high_dir / name)
        if low.shape != high.shape:
            raise ValueError(f"paired image dimensions differ for {name}: {low.shape} vs {high.shape}")

        size = self.patch_size
        if self.training and size > 0:
            _, height, width = low.shape
            if height < size or width < size:
                raise ValueError(f"{name} is {width}x{height}, smaller than patch size {size}")
            # Crop coordinates depend only on (seed, epoch, sorted sample index),
            # so every variant sees the same patch for a given epoch/sample.
            rng = random.Random(self.seed + self.epoch * len(self.names) + index)
            top = rng.randint(0, height - size)
            left = rng.randint(0, width - size)
            low = low[:, top:top + size, left:left + size]
            high = high[:, top:top + size, left:left + size]
        return low, high, name


def set_seed(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def mean_psnr(net, loader, device):
    net.eval()
    scores = []
    with torch.inference_mode():
        for low, high, _ in loader:
            low, high = low.to(device), high.to(device)
            output = net(low)
            mse = ((output - high) ** 2).mean((2, 3))
            scores.extend((10 * torch.log10(1.0 / mse)).mean(dim=1).cpu().tolist())
    return float(np.mean(scores))


def save_state(path, model):
    torch.save(model.state_dict(), path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--train-low", required=True)
    parser.add_argument("--train-high", required=True)
    parser.add_argument("--valid-low", required=True)
    parser.add_argument("--valid-high", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256,
                        help="random paired training crop; set 0 to use full images")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-learning-rate", type=float, default=1e-6)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--rep-scale", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="V6 protocol control: start main training without warm-up")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.warmup_epochs < 0 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive; warmup-epochs cannot be negative")
    if args.patch_size < 0 or args.num_workers < 0 or args.save_every < 1:
        parser.error("patch-size and num-workers cannot be negative; save-every must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but no CUDA device is available")

    run_dir = Path(args.output_root) / args.variant / f"seed-{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    model_dir = run_dir / "models"
    model_dir.mkdir()

    set_seed(args.seed, args.deterministic)
    device = torch.device(args.device)
    train_data = PairedImages(args.train_low, args.train_high, args.patch_size, args.seed, True)
    valid_data = PairedImages(args.valid_low, args.valid_high, 0, args.seed, False)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, generator=generator,
    )
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=0)
    if len(train_loader) == 0:
        raise ValueError("training set has fewer pairs than batch-size")

    settings = vars(args).copy()
    settings.update({"device": str(device), "run_dir": str(run_dir), "train_pairs": len(train_data),
                     "valid_pairs": len(valid_data), "warmup_enabled": not args.skip_warmup})
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    net = build_variant(args.variant, training=True, channels=args.channels).to(device)
    train_loss = import_loss("lle")
    warmup_loss = import_loss("warmup")
    best_psnr = -float("inf")
    history_path = run_dir / "history.jsonl"

    def record(row):
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if not args.skip_warmup:
        optimizer = torch.optim.Adam(net.parameters(), args.warmup_learning_rate, weight_decay=0)
        for epoch in range(args.warmup_epochs):
            train_data.epoch = epoch
            net.train()
            losses = []
            for low, high, _ in train_loader:
                low, high = low.to(device), high.to(device)
                pred1, pred2 = net.forward_warm(low)
                loss = warmup_loss(low, high, pred1, pred2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            save_state(model_dir / "model_pre.pkl", net)
            record({"phase": "warmup", "epoch": epoch + 1, "loss": float(np.mean(losses))})

    optimizer = torch.optim.Adam(net.parameters(), args.learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 50, 2, 1e-7
    )
    for epoch in range(args.epochs):
        train_data.epoch = epoch + args.warmup_epochs
        net.train()
        losses = []
        for low, high, _ in train_loader:
            low, high = low.to(device), high.to(device)
            output = net(low)
            loss = train_loss(output, high)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        validation_psnr = mean_psnr(net, valid_loader, device)
        if (epoch + 1) % args.save_every == 0:
            save_state(model_dir / f"model_{epoch + 1}.pkl", net)
        improved = validation_psnr > best_psnr
        if improved:
            best_psnr = validation_psnr
            save_state(model_dir / "model_best.pkl", net)
            slim = net.slim().to(device)
            save_state(model_dir / "model_best_slim.pkl", slim)
        record({"phase": "train", "epoch": epoch + 1, "loss": float(np.mean(losses)),
                "validation_psnr": validation_psnr, "best_psnr": best_psnr,
                "best_checkpoint_updated": improved})


if __name__ == "__main__":
    main()
