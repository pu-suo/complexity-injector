#!/usr/bin/env python3
"""Unattended model-selection run.

Goal: settle BOTH shipping models by morning.
  * a large judge for machines with WebGPU
  * a small judge for CPU-only fallback

Every step is wrapped so one failure cannot end the night, and every step logs
what it decided and why. Nothing here calls a paid API.
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "data" / "models"
LOG = ROOT / "data" / "models" / "overnight_log.txt"


def log(m: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def run(cmd: list[str], why: str) -> bool:
    log(f"START {why}")
    log("  $ " + " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        if any(k in ln for k in ("test:", "dev:", "params", "int8:", "->")):
            log("  " + ln.strip())
    if r.returncode != 0:
        log(f"  FAILED rc={r.returncode}: {r.stderr.strip().splitlines()[-1:]}")
        return False
    log(f"  done in {(time.time()-t0)/60:.1f} min")
    return True


def auc_of(tag: str) -> float:
    f = MODELS / f"result_{tag}.json"
    if not f.exists():
        return -1.0
    return json.load(f.open()).get("test", {}).get("auc", -1.0)


def main() -> None:
    log("=" * 70)
    log("OVERNIGHT RUN START")

    # ---- Phase 1: wait for the in-flight large-model runs -----------------
    while subprocess.run(["pgrep", "-f", "train_judge.py --model"],
                         capture_output=True).returncode == 0:
        time.sleep(60)
    log("phase 1 complete: in-flight large-model training finished")

    # ---- Phase 2: rank every trained model, pick the large winner --------
    run([sys.executable, "scripts/rank_models.py"], "rank all trained models")

    ranking = {}
    f = MODELS / "ranking.json"
    if f.exists():
        ranking = json.load(f.open())
    large = ranking.get("best_large")
    small_pool = ranking.get("small_candidates", ["mini", "deep12"])
    if not large:
        log("no large winner found; defaulting to distil")
        large = "distil"
    log(f"LARGE WINNER: {large}   small candidates: {small_pool}")

    # ---- Phase 3: distil the winner into each CPU candidate --------------
    # kd-focus/kd-temp added because the first KD run gained AUC but lost
    # recall at the low-FP operating point the extension actually uses.
    for student, hf in (("mini", "google/bert_uncased_L-4_H-256_A-4"),
                        ("deep12", "google/bert_uncased_L-12_H-256_A-4")):
        if student not in small_pool:
            continue
        run([sys.executable, "scripts/distill.py", "--teacher", large,
             "--student", hf, "--tag", f"{student}_kd2",
             "--epochs", "6", "--kd-focus", "2.0", "--kd-temp", "0.7"],
            f"distil {large} -> {student}")

    # ---- Phase 4: export + benchmark every finalist ----------------------
    finalists = [large] + [f"{s}_kd2" for s in small_pool]
    for tag in finalists:
        if (MODELS / f"judge_{tag}.pt").exists():
            run([sys.executable, "scripts/export_and_benchmark.py", "--tag", tag],
                f"export+benchmark {tag}")

    # ---- Phase 5: thresholds on dev, final numbers on gold ---------------
    run([sys.executable, "scripts/pick_thresholds.py",
         "--tags", ",".join(finalists)], "choose operating thresholds")

    log("OVERNIGHT RUN COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
