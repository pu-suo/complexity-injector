#!/usr/bin/env python3
"""Resize a screenshot to the Chrome Web Store's required dimensions.

The store accepts 1280x800 or 640x400 and rejects anything else. This scales an
image to fit and pads it on a neutral background rather than cropping, so the
page content stays intact.

    python scripts/make_screenshot.py shot.png
    python scripts/make_screenshot.py shot.png --background 1e1e1e
"""
from __future__ import annotations
import argparse
from pathlib import Path

from PIL import Image

W, H = 1280, 800


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--background", default="ffffff")
    ap.add_argument("--out-dir", default="store-assets")
    args = ap.parse_args()

    bg = tuple(int(args.background[i:i + 2], 16) for i in (0, 2, 4))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    for src in args.images:
        im = Image.open(src).convert("RGB")
        scale = min(W / im.width, H / im.height)
        # Never upscale: a stretched screenshot looks worse than a padded one.
        scale = min(scale, 1.0)
        resized = im.resize((round(im.width * scale), round(im.height * scale)),
                            Image.LANCZOS)
        canvas = Image.new("RGB", (W, H), bg)
        canvas.paste(resized, ((W - resized.width) // 2, (H - resized.height) // 2))
        out = out_dir / f"{Path(src).stem}-1280x800.png"
        canvas.save(out)
        print(f"[shot] {src} ({im.width}x{im.height}) -> {out}")


if __name__ == "__main__":
    main()
