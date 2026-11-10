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
        mirror = "https://raw.githubusercontent.com/cszn/FFDNet/master/testsets/Kodak24/" + name
        target = cache / name
        provenance = target.with_suffix(".source.json")
        source = url
        if not target.exists():
            errors = []
            for candidate in [url, mirror]:
                try:
                    request = urllib.request.Request(candidate, headers={"User-Agent": "MobileIE-Neo-research/1.0"})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        payload = response.read()
                    source = candidate
                    break
                except Exception as error:
                    errors.append(f"{candidate}: {error}")
                    print(errors[-1], flush=True)
            else:
                raise RuntimeError("; ".join(errors))
            temporary = target.with_suffix(".download")
            temporary.write_bytes(payload)
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(target)
            provenance.write_text(json.dumps({"download_url": source, "errors": errors}))
        if provenance.exists():
            source = json.loads(provenance.read_text())["download_url"]
        with Image.open(target) as image:
            size = list(image.size)
        rows.append({"name": name[:-4], "path": str(target), "url": url, "download_url": source, "size": size,
                     "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                     "source_version": "native dimensions" if size in [[768, 512], [512, 768]] else "mirror 500x500 version",
                     "split": "train" if index <= 6 else "exploratory_validation"})
        print(name, size, flush=True)
    destination = Path(args.manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps({"retrieved": datetime.datetime.now().astimezone().isoformat(),
        "source": "https://r0k.us/graphics/kodak/", "credits": "https://r0k.us/graphics/kodak/PhotoCD_credits.txt",
        "purpose": "exploration only; original pretraining overlap not audited; source images kept in ignored cache",
        "images": rows,
        "transform": "aspect-preserving cover and center crop to 768x512; not a standard Kodak benchmark"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
