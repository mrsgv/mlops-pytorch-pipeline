#!/usr/bin/env python3
"""Write ``test_image.png`` for the /predict demo.

Prefers a real image from the CIFAR-10 test split (so the prediction can be
checked against a known label). Falls back to a synthetic image if the dataset
has not been downloaded yet.

    python scripts/make_test_image.py                  # ./test_image.png
    python scripts/make_test_image.py --index 7 --out /tmp/cat.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from PIL import Image  # noqa: E402

from model import CIFAR10_CLASSES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--index", type=int, default=0, help="index in the test split")
    parser.add_argument("--out", default=str(REPO_ROOT / "test_image.png"))
    parser.add_argument("--upscale", type=int, default=1, help="nearest-neighbour zoom")
    args = parser.parse_args()

    out_path = Path(args.out)
    label = None

    try:
        from torchvision import datasets

        # download=False on purpose: this script should never pull 170 MB by
        # surprise. Run training once first, or pass --data-dir.
        dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=False)
        image, target = dataset[args.index]
        label = CIFAR10_CLASSES[target]
    except Exception as exc:  # noqa: BLE001 - any failure means "no dataset"
        print(f"CIFAR-10 not available ({type(exc).__name__}); writing a synthetic image")
        image = Image.new("RGB", (32, 32), color=(64, 128, 192))

    if args.upscale > 1:
        size = (image.width * args.upscale, image.height * args.upscale)
        image = image.resize(size, Image.NEAREST)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"wrote {out_path} ({image.width}x{image.height})", end="")
    print(f", ground truth: {label}" if label else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
