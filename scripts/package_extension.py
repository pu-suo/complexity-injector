#!/usr/bin/env python3
"""Build the Chrome Web Store upload.

Ships only what the browser needs: no node_modules, no source maps, no build
scripts. The store rejects packages containing obvious development leftovers.

    python scripts/package_extension.py
"""
from __future__ import annotations
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "extension"
SKIP_DIRS = {"node_modules", "icons/options"}
SKIP_NAMES = {"package.json", "package-lock.json", ".gitignore", "preview.png",
              "content.js"}          # content.js ships bundled, not as source


def included() -> list[Path]:
    out = []
    for p in sorted(EXT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(EXT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if rel.name in SKIP_NAMES or rel.suffix == ".map":
            continue
        out.append(p)
    return out


def main() -> None:
    manifest = json.loads((EXT / "manifest.json").read_text())
    version = manifest["version"]
    files = included()

    required = {"manifest.json", "content.bundle.js", "lib/judge.onnx",
                "lib/ort/ort.mjs", "lib/config.json", "lib/vocab.txt",
                "lib/tokenizer.js", "lib/judge.js", "lib/serialize.js",
                "offscreen.html", "offscreen.js", "background.js"}
    have = {str(p.relative_to(EXT)) for p in files}
    missing = required - have
    if missing:
        raise SystemExit("[package] missing, run the build steps first: "
                         + ", ".join(sorted(missing)))

    out = ROOT / f"complexity-injector-{version}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            z.write(p, p.relative_to(EXT))

    raw = sum(p.stat().st_size for p in files)
    print(f"[package] {len(files)} files, {raw/1e6:.0f} MB raw")
    print(f"[package] -> {out.name}  {out.stat().st_size/1e6:.0f} MB compressed")
    print(f"[package] store limit is 2 GB; "
          f"{out.stat().st_size/2e9*100:.0f}% of it used")


if __name__ == "__main__":
    main()
