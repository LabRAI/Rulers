#!/usr/bin/env python
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rulers_demo.config import DEFAULT_MODEL, OUT_DIR_DEFAULT, RANDOM_SEED
from rulers_demo.pipeline import run_asap2_demo


def build_args():
    ap = argparse.ArgumentParser(
        description="RULERS ASAP2.0 type-aware evidence demo. This is a compact method demo, not a full paper reproduction script."
    )
    ap.add_argument("--data-dir", type=str, required=True, help="Path to ASAP2.0 dataset directory containing train/test CSVs or zips.")
    ap.add_argument("--repo-url", type=str, default="", help="Optional dataset repo URL to clone if --data-dir is missing.")
    ap.add_argument("--zip-password", type=str, default="asap2_test", help="Password for ASAP2.0 zip files if needed.")

    ap.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenRouter/OpenAI model name, e.g., openai/gpt-4o-mini.")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR_DEFAULT)
    ap.add_argument("--max-workers", type=int, default=4)

    ap.add_argument("--rubric-txt", type=str, default="", help="Optional path to override the built-in ASAP2.0 holistic rubric.")
    ap.add_argument("--rubric-bundle-path", type=str, default="", help="Load an existing locked rubric bundle JSON instead of compiling a new one.")
    ap.add_argument("--rubric-version", type=str, default="asap2_demo_v1")
    ap.add_argument("--force-regen-rubric", action="store_true")
    ap.add_argument("--rubric-train-n", type=int, default=240)
    ap.add_argument("--rubric-reps-per-score", type=int, default=3)
    ap.add_argument("--rubric-temperature", type=float, default=0.2)

    ap.add_argument("--calib-n", type=int, default=200, help="Number of labeled examples used for WGR calibration.")
    ap.add_argument("--test-max", type=int, default=0, help="Limit test examples for quick debugging. 0 means all labeled test rows.")
    ap.add_argument("--score-min", type=int, default=1)
    ap.add_argument("--score-max", type=int, default=6)
    ap.add_argument("--score-max-tokens", type=int, default=2400)
    ap.add_argument("--sent-max-chars", type=int, default=260)

    ap.add_argument("--prefer-checklist", action="store_true", help="Use checklist-derived trait scores instead of returned trait_scores.")
    ap.add_argument("--no-calibration", dest="enable_calibration", action="store_false", help="Disable WGR calibration and use raw score average.")
    ap.set_defaults(enable_calibration=True)

    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--zip-artifacts", action="store_true", default=True)
    ap.add_argument("--no-zip-artifacts", dest="zip_artifacts", action="store_false")
    ap.add_argument("--colab-download", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    run_asap2_demo(build_args())
