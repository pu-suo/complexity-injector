#!/usr/bin/env python3
"""Turn a macOS screenshot into a Chrome Web Store screenshot.

The store accepts 1280x800 or 640x400 and rejects anything else, including the
alpha channel that macOS window captures always carry.

A window capture also arrives sitting in a transparent margin with a drop
shadow, and at a 1.644 aspect ratio rather than the required 1.600. This trims
the margin, crops to the exact aspect from the top (page content matters more
than the bottom of the window), then scales.

    python scripts/make_screenshot.py ~/Desktop/Screenshot*.png
    python scripts/make_screenshot.py shot.png --crop-top 120   # cut a toolbar
"""
from __future__ import annotations
import argparse
from pathlib import Path

from PIL import Image, ImageChops

W, H = 1280, 800


def trim_margin(im: Image.Image, backdrop=(255, 255, 255)) -> Image.Image:
    """Find the window inside a macOS capture and drop everything around it.

    The alpha channel is no help for locating it: the drop shadow is
    semi-transparent and reaches the edges of the frame, so an alpha bounding
    box returns almost the whole image. Compositing onto black and taking the
    bounding box of what is not black finds the window.

    The result is then composited onto `backdrop` rather than black, so the
    window's rounded corners and shadow blend into the page instead of leaving
    dark notches at the corners.
    """
    if im.mode != "RGBA":
        return im.convert("RGB")
    alpha = im.getchannel("A")
    on_black = Image.new("RGB", im.size, (0, 0, 0))
    on_black.paste(im, mask=alpha)
    box = ImageChops.difference(on_black, Image.new("RGB", im.size, (0, 0, 0))).getbbox()
    on_backdrop = Image.new("RGB", im.size, backdrop)
    on_backdrop.paste(im, mask=alpha)
    return on_backdrop.crop(box) if box else on_backdrop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out-dir", default="store-assets")
    ap.add_argument("--crop-top", type=int, default=0,
                    help="pixels to remove from the top, before scaling")
    ap.add_argument("--size", default="1280x800", choices=["1280x800", "640x400"])
    ap.add_argument("--fit", default="pad", choices=["pad", "crop"],
                    help="pad keeps the whole window; crop trims to fill")
    args = ap.parse_args()

    target_w, target_h = (int(x) for x in args.size.split("x"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    for src in args.images:
        im = trim_margin(Image.open(src))
        if args.crop_top:
            im = im.crop((0, args.crop_top, im.width, im.height))

        want = target_w / target_h
        if args.fit == "crop":
            if im.width / im.height > want:               # too wide, trim sides
                new_w = round(im.height * want)
                off = (im.width - new_w) // 2
                im = im.crop((off, 0, off + new_w, im.height))
            else:                                        # too tall, trim bottom
                new_h = round(im.width / want)
                im = im.crop((0, 0, im.width, new_h))
            im = im.convert("RGB").resize((target_w, target_h), Image.LANCZOS)
        else:
            # Scale to fit and pad. The fill is sampled from the window's own
            # top-left corner, so the bars read as part of the chrome rather
            # than as a black letterbox.
            im = im.convert("RGB")
            # Sample the page, not the corner: (2, 2) lands on the window's
            # rounded corner, which is backdrop rather than chrome.
            fill = im.getpixel((2, im.height // 2))
            scale = min(target_w / im.width, target_h / im.height)
            small = im.resize((round(im.width * scale), round(im.height * scale)),
                              Image.LANCZOS)
            canvas = Image.new("RGB", (target_w, target_h), fill)
            canvas.paste(small, ((target_w - small.width) // 2,
                                 (target_h - small.height) // 2))
            im = canvas
        out = out_dir / f"{Path(src).stem.replace(' ', '-')}-{target_w}x{target_h}.png"
        im.save(out)
        print(f"  {Path(src).name}  ->  {out.name}  "
              f"{im.size[0]}x{im.size[1]} {im.mode} (no alpha)")


if __name__ == "__main__":
    main()
