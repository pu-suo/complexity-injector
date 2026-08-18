#!/usr/bin/env python3
"""Stage onnxruntime-web into the extension.

The runtime is vendored rather than committed: it is 39 MB of build output that
npm can reproduce exactly. A strict extension CSP forbids loading it from a CDN,
so it has to sit inside the package.

    python scripts/vendor_ort.py
"""
from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "extension"
DIST = EXT / "node_modules" / "onnxruntime-web" / "dist"
OUT = EXT / "lib" / "ort"
FILES = {
    "ort.all.bundle.min.mjs": "ort.mjs",
    "ort-wasm-simd-threaded.jsep.wasm": None,   # WebGPU-capable build
    "ort-wasm-simd-threaded.jsep.mjs": None,
    "ort-wasm-simd-threaded.wasm": None,        # CPU fallback
    "ort-wasm-simd-threaded.mjs": None,
}


def main() -> None:
    if not DIST.exists():
        print("[ort] installing onnxruntime-web ...")
        subprocess.run(["npm", "install", "--no-audit", "--no-fund"],
                       cwd=EXT, check=True)
    if not DIST.exists():
        sys.exit(f"[ort] {DIST} still missing after npm install")

    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for src, dst in FILES.items():
        s = DIST / src
        if not s.exists():
            sys.exit(f"[ort] missing {s}; check the onnxruntime-web version")
        d = OUT / (dst or src)
        shutil.copy(s, d)
        total += d.stat().st_size
    print(f"[ort] staged {len(FILES)} files -> {OUT.relative_to(ROOT)} "
          f"({total / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
