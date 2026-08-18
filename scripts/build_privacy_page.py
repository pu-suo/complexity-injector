#!/usr/bin/env python3
"""Render PRIVACY.md to a standalone privacy.html.

GitHub Pages is case-sensitive, so a policy that lives only at /PRIVACY is a
404 for anyone who types /privacy -- a bad property for a URL that goes on a
store submission and into a published policy field. A lowercase static page
also does not depend on Jekyll processing the markdown at all.

    python scripts/build_privacy_page.py
"""
from __future__ import annotations
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "PRIVACY.md"
OUT = ROOT / "privacy.html"

TEMPLATE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Privacy Policy — Complexity Injector</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         max-width: 42rem; margin: 0 auto; padding: 3rem 1.25rem 5rem; }}
  h1 {{ font-size: 1.6rem; margin: 0 0 .25rem; }}
  h2 {{ font-size: 1.1rem; margin: 2rem 0 .5rem; }}
  ul {{ padding-left: 1.2rem; }}
  li {{ margin: .35rem 0; }}
  code {{ font-size: .9em; background: rgba(128,128,128,.15); padding: .1em .35em;
         border-radius: 3px; }}
  a {{ color: #2563d6; }}
  footer {{ margin-top: 3rem; padding-top: 1rem; font-size: .85rem; opacity: .6;
           border-top: 1px solid rgba(128,128,128,.3); }}
</style>
{body}
<footer>Complexity Injector — <a href="./">project home</a></footer>
</html>
"""


def main() -> None:
    html = markdown.markdown(SRC.read_text(), extensions=["extra", "sane_lists"])
    OUT.write_text(TEMPLATE.format(body=html))
    print(f"[privacy] {SRC.name} -> {OUT.name} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
