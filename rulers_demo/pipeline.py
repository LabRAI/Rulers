import concurrent.futures
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from .calibration import attach_final_scores
from .data_asap import load_asap2_official, stratified_sample
from .metrics import compute_metrics, summarize_cost
from .rubric import build_or_load_rubric_bundle
from .scoring import safe_score_essay
from .utils import append_jsonl, load_json, safe_mkdir, save_json, set_global_seed, ts, zip_dir, maybe_colab_download


def _score_dataframe(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    bundle: Dict[str, Any],
    model: str,
    out_jsonl: str,
    max_workers: int,
    prefer_checklist: bool,
    score_min: int,
    score_max: int,
    sent_max_chars: int,
    score_max_tokens: int,
) -> List[Dict[str, Any]]:
    id_col = meta["id_col"]
    text_col = meta["text_col"]
    score_col = meta["score_col"]
    rows = []

    def job(row_tuple):
        _, row = row_tuple
        essay_id = str(row[id_col])
        text = str(row[text_col])
        human_score = None
        if score_col in row and pd.notna(row[score_col]):
            human_score = int(row[score_col])
        return safe_score_essay(
            model=model,
            essay_id=essay_id,
            text=text,
            human_score=human_score,
            bundle=bundle,
            sent_max_chars=sent_max_chars,
            prefer_checklist=prefer_checklist,
            score_min=score_min,
            score_max=score_max,
            max_tokens=score_max_tokens,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(job, item) for item in df.iterrows()]
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Scoring"):
            row = fut.result()
            rows.append(row)
            append_jsonl(row, out_jsonl)
    return rows


def run_asap2_demo(args) -> Dict[str, Any]:
    set_global_seed(args.seed)
    run_dir = os.path.join(args.out_dir, f"asap2_typeaware_{ts()}")
    safe_mkdir(run_dir)

    train_df, test_df, meta = load_asap2_official(
        data_dir=args.data_dir,
        repo_url=args.repo_url,
        zip_password=args.zip_password,
    )
    print(f"[DATA] train={len(train_df)}, test={len(test_df)}")
    print(f"[DATA] schema={meta}")

    score_col = meta["score_col"]
    calib_df_src = stratified_sample(train_df, score_col, min(args.calib_n, len(train_df)), args.seed).copy()

    # This demo evaluates labeled test rows if the test CSV includes labels. If not,
    # it still writes predictions but cannot compute QWK.
    if score_col in test_df.columns:
        test_eval = test_df.dropna(subset=[score_col]).copy()
    else:
        test_eval = test_df.copy()
    if args.test_max and int(args.test_max) > 0:
        test_eval = test_eval.head(int(args.test_max)).copy()

    bundle, rubric_src = build_or_load_rubric_bundle(
        train_df=train_df,
        meta=meta,
        run_dir=run_dir,
        model=args.model,
        seed=args.seed,
        rubric_txt=args.rubric_txt,
        rubric_bundle_path=args.rubric_bundle_path,
        rubric_version=args.rubric_version,
        rubric_train_n=args.rubric_train_n,
        rubric_reps_per_score=args.rubric_reps_per_score,
        rubric_temperature=args.rubric_temperature,
        force_regen=args.force_regen_rubric,
        calibration_df=calib_df_src,
    )
    save_json(bundle, os.path.join(run_dir, "rubric_bundle_locked.json"))

    calib_rows = _score_dataframe(
        df=calib_df_src,
        meta=meta,
        bundle=bundle,
        model=args.model,
        out_jsonl=os.path.join(run_dir, "calib_scored.jsonl"),
        max_workers=args.max_workers,
        prefer_checklist=args.prefer_checklist,
        score_min=args.score_min,
        score_max=args.score_max,
        sent_max_chars=args.sent_max_chars,
        score_max_tokens=args.score_max_tokens,
    )
    test_rows = _score_dataframe(
        df=test_eval,
        meta=meta,
        bundle=bundle,
        model=args.model,
        out_jsonl=os.path.join(run_dir, "test_scored.jsonl"),
        max_workers=args.max_workers,
        prefer_checklist=args.prefer_checklist,
        score_min=args.score_min,
        score_max=args.score_max,
        sent_max_chars=args.sent_max_chars,
        score_max_tokens=args.score_max_tokens,
    )

    calib_table = pd.DataFrame(calib_rows)
    test_table = pd.DataFrame(test_rows)
    calib_table, test_table, _calib_model = attach_final_scores(
        calib_table,
        test_table,
        enable_calibration=args.enable_calibration,
        score_min=args.score_min,
        score_max=args.score_max,
    )

    calib_table.to_csv(os.path.join(run_dir, "calib_table.csv"), index=False)
    test_table.to_csv(os.path.join(run_dir, "test_table.csv"), index=False)

    metrics: Dict[str, Any] = {}
    labeled = test_table.dropna(subset=["human_score"]).copy() if "human_score" in test_table.columns else pd.DataFrame()
    if len(labeled) > 0:
        metrics = compute_metrics(
            labeled["human_score"].astype(int).values,
            labeled["final_score"].astype(int).values,
        )

    summary = {
        "run_dir": run_dir,
        "model": args.model,
        "data_source": meta.get("_source"),
        "data_paths": meta.get("_paths"),
        "rubric_source": rubric_src,
        "rubric_version": bundle.get("rubric_version"),
        "bundle_hash": bundle.get("bundle_hash"),
        "type_aware_checklist": True,
        "prefer_checklist": bool(args.prefer_checklist),
        "enable_calibration": bool(args.enable_calibration),
        "calib_n": int(len(calib_table)),
        "test_n": int(len(test_table)),
        "labeled_test_n": int(len(labeled)),
        "metrics": metrics,
        "cost": summarize_cost(calib_rows + test_rows),
        "final_score_column": "final_score",
        "final_score_cont_column": "final_score_cont",
    }
    save_json(summary, os.path.join(run_dir, "summary.json"))
    print("[DONE] Summary:", summary)

    if args.zip_artifacts:
        zip_path = zip_dir(run_dir)
        print(f"[DONE] Zipped artifacts: {zip_path}")
        if args.colab_download:
            maybe_colab_download(zip_path)
        summary["zip_path"] = zip_path
    return summary
