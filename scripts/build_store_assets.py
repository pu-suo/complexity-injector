#!/usr/bin/env python3
"""Build Chrome Web Store graphic assets.

Two constraints from the store's image guidelines drive the shapes here:
the 128x128 store icon must sit in a 96x96 safe area with transparent padding,
and promo tiles must be 24-bit PNG with no alpha channel.

    python scripts/build_store_assets.py
"""
from __future__ import annotations
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "store-assets"
SLATE = (58, 64, 78)
ACCENT = (37, 99, 214)
PAPER = (247, 248, 250)


def mark(size: int) -> Image.Image:
    """The caret, drawn at an arbitrary size on a transparent canvas."""
    S = 1024
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.line([(60, 632), (512, 184), (964, 632)], fill=SLATE + (255,),
           width=208, joint="curve")
    d.line([(60, 860), (964, 860)], fill=ACCENT + (255,), width=168)
    return im.resize((size, size), Image.LANCZOS)


def font(px: int):
    for f in ("/System/Library/Fonts/Supplemental/Futura.ttc",
              "/System/Library/Fonts/HelveticaNeue.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(f, px)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # Store icon: 96x96 of artwork inside a 128x128 canvas.
    icon = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    icon.paste(mark(96), (16, 16), mark(96))
    icon.save(OUT / "store-icon-128.png")

    # Small promo tile: 440x280, flattened onto a solid background.
    tile = Image.new("RGB", (440, 280), PAPER)
    m = mark(104)
    tile.paste(m, (40, 58), m)
    d = ImageDraw.Draw(tile)
    d.text((40, 178), "Complexity", font=font(34), fill=SLATE)
    d.text((40, 212), "Injector", font=font(34), fill=SLATE)
    d.text((186, 96), "talkative", font=font(21), fill=(150, 155, 165))
    d.line([(184, 112), (294, 112)], fill=(176, 182, 194), width=3)
    d.text((186, 122), "garrulous", font=font(25), fill=ACCENT)
    d.line([(186, 156), (300, 156)], fill=ACCENT, width=3)
    tile.save(OUT / "promo-small-440x280.png")

    for p in (OUT / "store-icon-128.png", OUT / "promo-small-440x280.png"):
        im = Image.open(p)
        print(f"  {p.name:<28} {im.size[0]}x{im.size[1]}  {im.mode}"
              f"{'  (alpha)' if im.mode == 'RGBA' else '  (no alpha)'}")


if __name__ == "__main__":
    main()
