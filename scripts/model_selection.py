#!/usr/bin/env python3
"""Model selection: rank every trained judge, distil the winner, benchmark.

Runs unattended and logs every decision, so a sweep can be left to finish and
audited afterwards. Each step is isolated: one failure does not end the run.

  1. wait for any in-flight training
  2. rank all checkpoints on the dev split
  3. distil the winner into the CPU-affordable candidates
  4. export each finalist to ONNX and measure latency
  5. calibrate thresholds on dev, report on the gold set

    python scripts/model_selection.py
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "data" / "models"
LOG = ROOT / "data" / "models" / "model_selection_log.txt"


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
    log("MODEL SELECTION START")

    # ---- 1. wait for any training already running -------------------------
    while subprocess.run(["pgrep", "-f", "train_judge.py --model"],
                         capture_output=True).returncode == 0:
        time.sleep(60)
    log("in-flight training finished")

    # ---- 2. rank every checkpoint, pick the large winner ------------------
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

    # ---- 3. distil the winner into each CPU candidate ---------------------
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

    # ---- 4. export and benchmark every finalist ---------------------------
    finalists = [large] + [f"{s}_kd2" for s in small_pool]
    for tag in finalists:
        if (MODELS / f"judge_{tag}.pt").exists():
            run([sys.executable, "scripts/export_and_benchmark.py", "--tag", tag],
                f"export+benchmark {tag}")

    # ---- 5. thresholds on dev, final numbers on gold ----------------------
    run([sys.executable, "scripts/pick_thresholds.py",
         "--tags", ",".join(finalists)], "choose operating thresholds")

    log("MODEL SELECTION COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
