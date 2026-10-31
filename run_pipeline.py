"""Runs the whole training pipeline in order, stopping at the first failure.

    python run_pipeline.py                # steps 1-7
    python run_pipeline.py --from-step 5  # resume (e.g. after step 3 already succeeded)
    python run_pipeline.py --list         # show the steps
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = "delhi_ncr_aqi_dataset.csv"
ENGINEERED = "engineered_aqi_features.pkl"
PREDICTIONS = "predictions_2025.pkl"

# (script, description, input file it needs, step that creates that input)
STEPS = [
    ("step1_verify.py", "Verify the raw dataset", RAW, None),
    ("step2_target.py", "Check the forecast target (AQI at t+1)", RAW, None),
    ("step3_features.py", "Engineer lag/rolling/cyclical features -> " + ENGINEERED, RAW, None),
    ("step4_split_and_baseline.py", "Chronological split + persistence baseline", ENGINEERED, 3),
    ("step5_train.py", "Train benchmark model -> " + PREDICTIONS, ENGINEERED, 3),
    ("step6_diagnostics.py", "Diagnostics + plots", PREDICTIONS, 5),
    ("export_model.py", "Train + evaluate + export aqi_forecast_model.pkl", ENGINEERED, 3),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-step", type=int, default=1, metavar="N", help="start at step N (1-7)")
    parser.add_argument("--list", action="store_true", help="list the steps and exit")
    args = parser.parse_args()

    if args.list:
        for i, (script, desc, _, _) in enumerate(STEPS, 1):
            print(f"{i}. {script:<30} {desc}")
        return 0
    if not 1 <= args.from_step <= len(STEPS):
        parser.error(f"--from-step must be between 1 and {len(STEPS)}")

    for i, (script, desc, needs, made_by) in enumerate(STEPS, 1):
        if i < args.from_step:
            continue
        if not (ROOT / needs).exists():
            hint = (f"Put the dataset in {ROOT} (it is git-ignored, so it is not in the repo)."
                    if made_by is None else f"Run step {made_by} first (or start from step 1).")
            print(f"\nCannot run step {i} ({script}): '{needs}' not found.\n{hint}")
            return 1
        print(f"\n{'=' * 70}\nStep {i}/{len(STEPS)}: {desc}\n{'=' * 70}", flush=True)
        started = time.time()
        result = subprocess.run([sys.executable, script], cwd=ROOT)
        if result.returncode != 0:
            print(f"\nStep {i} ({script}) failed with exit code {result.returncode}. Stopping.")
            return result.returncode
        print(f"-- step {i} finished in {time.time() - started:.1f}s")

    print("\nPipeline complete. Start the API with:  uvicorn app:app --reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
