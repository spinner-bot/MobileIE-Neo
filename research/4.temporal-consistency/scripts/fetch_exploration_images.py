"""Cache Kodak 01-12, validate images, and save a reproducible split manifest."""
import argparse
import datetime
import hashlib
import json
import urllib.request
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=".cache/data/kodak")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(1, 13):
        name = f"kodim{index:02d}.png"
        url = "https://r0k.us/graphics/kodak/kodak/" + name
        target = cache / name
        if not target.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "MobileIE-Neo-research/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            temporary = target.with_suffix(".download")
            temporary.write_bytes(payload)
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(target)
        with Image.open(target) as image:
            size = list(image.size)
        rows.append({"name": name[:-4], "path": str(target), "url": url, "size": size,
                     "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                     "split": "train" if index <= 6 else "exploratory_validation"})
        print(name, size, flush=True)
    destination = Path(args.manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps({"retrieved": datetime.datetime.now().astimezone().isoformat(),
        "source": "https://r0k.us/graphics/kodak/", "credits": "https://r0k.us/graphics/kodak/PhotoCD_credits.txt",
        "purpose": "exploration only; original pretraining overlap not audited; source images kept in ignored cache",
        "images": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
