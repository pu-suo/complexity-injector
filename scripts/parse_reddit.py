#!/usr/bin/env python3
"""Strip Reddit UI chrome from a pasted thread, leaving comment bodies.

A pasted thread is mostly furniture -- vote counts, "Reply", "Award", "Share",
usernames, ad copy, sidebar rules. None of it is prose the extension would ever
touch, so leaving it in would understate the delivery rate per word of real text.
"""
from __future__ import annotations
import re, sys
from pathlib import Path

CHROME = {
    "upvote", "downvote", "reply", "award", "share", "repost", "vote",
    "upvotevote", "join", "learn more", "shop now", "order now", "ad",
    "go to comments", "sort by:", "best", "search comments",
    "expand comment search", "comments section", "join the conversation",
    "collapse video player", "view more comments", "community info section",
    "cake icon", "public", "wiki", "secret", "full rules", "reddiquette",
    "related subreddits", "moderators", "message mods", "view all moderators",
    "installed apps", "reddit rules", "privacy policy", "user agreement",
    "your privacy choices", "accessibility", "collapse navigation",
    "community guide", "community bookmarks", "resources", "comment mop",
    "edited", "deleted", "[deleted]", "ask reddit...", "e36", "donkey_right",
}
NUM = re.compile(r"^[\d.,]+[km]?$", re.I)
AGE = re.compile(r"^\d+[hdmy]o?\s*ago$|^edited\s|^â¢$|^•$", re.I)
MORE = re.compile(r"^\d+\s+more repl(y|ies)$|^\d+\s+more$", re.I)
USER = re.compile(r"^u/[\w-]+|^[\w-]{3,20}$")
URLISH = re.compile(r"^https?://|\.com$|\.org$|\.tech$|\.io$")
RULE = re.compile(r"^rule \d+ -", re.I)


def comments(text: str) -> list[str]:
    out, buf = [], []
    for raw in text.splitlines():
        ln = raw.strip()
        low = ln.lower()
        drop = (not ln or low in CHROME or NUM.match(ln) or AGE.match(ln)
                or MORE.match(ln) or URLISH.match(low) or RULE.match(ln)
                or low.startswith("clickable image") or low.startswith("thumbnail image")
                or low.startswith("created jan") or low.startswith("r/")
                or low.startswith("0:00") or "avatar" in low)
        if drop:
            if buf:
                out.append(" ".join(buf)); buf = []
            continue
        # A bare short token on its own line is almost always a username.
        if USER.match(ln) and " " not in ln and len(ln) < 21:
            if buf:
                out.append(" ".join(buf)); buf = []
            continue
        buf.append(ln)
    if buf:
        out.append(" ".join(buf))
    # Mojibake from the paste: â¢ / â / ð and friends.
    fix = lambda s: (s.replace("â", "'").replace("â", "'").replace("â", '"')
                      .replace("â", '"').replace("â¦", "...").replace("â", "-")
                      .replace("â", "-").replace("â¢", "").strip())
    return [c for c in map(fix, out) if len(c.split()) >= 2]


if __name__ == "__main__":
    for p in sys.argv[1:]:
        cs = comments(Path(p).read_text(errors="replace"))
        w = sum(len(c.split()) for c in cs)
        print(f"{Path(p).name:<24} {len(cs):>4} comments  {w:>6} words  "
              f"median {sorted(len(c.split()) for c in cs)[len(cs)//2]:>3} words")
