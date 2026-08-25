import os
import re
import zipfile
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def ensure_repo(data_dir: str, repo_url: str) -> None:
    if os.path.exists(data_dir):
        return
    if not repo_url:
        raise FileNotFoundError(
            f"{data_dir} does not exist. Provide --repo-url or download ASAP2.0 manually."
        )
    print(f"[DATA] {data_dir} not found; cloning {repo_url} ...")
    ret = os.system(f'git clone "{repo_url}" "{data_dir}"')
    if ret != 0:
        raise RuntimeError("git clone failed. Please check --repo-url or provide local --data-dir.")


def extract_all_zips(data_dir: str, password: str = "asap2_test") -> None:
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if not fn.lower().endswith(".zip") or fn.startswith("._"):
                continue
            zip_path = os.path.join(root, fn)
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    try:
                        zf.setpassword(password.encode("utf-8"))
                        zf.extractall(root)
                    except RuntimeError:
                        zf.extractall(root)
            except Exception as exc:
                print(f"[DATA] Warning: failed to unzip {zip_path}: {exc}")


def find_best_csv(root: str, split_pattern: str) -> Optional[str]:
    best = None
    for r, _, files in os.walk(root):
        for fn in files:
            if not fn.lower().endswith(".csv") or fn.startswith("._"):
                continue
            if not re.search(split_pattern, fn, flags=re.I):
                continue
            path = os.path.join(r, fn)
            base = os.path.basename(path).lower()
            if base in (
                "train.csv",
                "test.csv",
                "asap2_train.csv",
                "asap2_test.csv",
                "asap_2_train.csv",
                "asap_2_test.csv",
            ):
                return path
            if best is None:
                best = path
    return best


def read_csv_safely(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


def infer_schema(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Optional[str]]:
    id_col = None
    for c in ["essay_id", "id", "ID", "essayid", "uid"]:
        if c in train_df.columns and c in test_df.columns:
            id_col = c
            break
    if id_col is None:
        id_col = "__essay_id__"
        train_df[id_col] = np.arange(len(train_df)).astype(int)
        test_df[id_col] = np.arange(len(test_df)).astype(int)

    prompt_id_col = None
    for c in ["prompt_id", "essay_set", "set_id", "set", "prompt", "task_id", "question_id", "domain1_prompt_id"]:
        if c in train_df.columns and c in test_df.columns:
            prompt_id_col = c
            break

    score_col = None
    for c in ["score", "holistic_score", "domain1_score", "human_score", "label", "y", "overall_score"]:
        if c in train_df.columns:
            score_col = c
            break
    if score_col is None:
        for c in [c for c in train_df.columns if "score" in c.lower()]:
            if pd.api.types.is_numeric_dtype(train_df[c]):
                score_col = c
                break
    if score_col is None:
        raise KeyError(f"Could not infer score column. Train columns: {list(train_df.columns)[:80]}")

    text_col = None
    for c in ["full_text", "essay", "text", "essay_text", "response", "content"]:
        if c in train_df.columns and c in test_df.columns:
            text_col = c
            break
    if text_col is None:
        shared_obj_cols = [c for c in train_df.columns if c in test_df.columns and train_df[c].dtype == object]
        if not shared_obj_cols:
            raise KeyError("Could not infer essay text column.")
        lens = [(c, train_df[c].astype(str).str.len().mean()) for c in shared_obj_cols]
        lens.sort(key=lambda x: x[1], reverse=True)
        text_col = lens[0][0]

    return {
        "id_col": id_col,
        "text_col": text_col,
        "score_col": score_col,
        "prompt_id_col": prompt_id_col,
    }


def load_asap2_official(
    data_dir: str,
    repo_url: str = "",
    zip_password: str = "asap2_test",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if repo_url and not os.path.exists(data_dir):
        ensure_repo(data_dir, repo_url)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"ASAP2.0 data directory not found: {data_dir}")

    extract_all_zips(data_dir, password=zip_password)
    train_path = find_best_csv(data_dir, r"train")
    test_path = find_best_csv(data_dir, r"test")
    if not train_path or not test_path:
        raise FileNotFoundError(
            f"Could not find train/test CSVs under {data_dir}. Found train={train_path}, test={test_path}."
        )

    train_df = read_csv_safely(train_path)
    test_df = read_csv_safely(test_path)
    schema = infer_schema(train_df, test_df)

    score_col = schema["score_col"]
    train_df[score_col] = pd.to_numeric(train_df[score_col], errors="coerce")
    train_df = train_df.dropna(subset=[score_col]).copy()
    train_df[score_col] = train_df[score_col].astype(int)

    if score_col in test_df.columns:
        test_df[score_col] = pd.to_numeric(test_df[score_col], errors="coerce")

    meta: Dict[str, Any] = {
        **schema,
        "_source": "ASAP2.0 official/local CSV",
        "_paths": {"train_path": train_path, "test_path": test_path},
    }
    return train_df, test_df, meta


def stratified_sample(df: pd.DataFrame, score_col: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    scores = sorted(df[score_col].dropna().unique().tolist())
    groups = []
    per = max(1, n // max(1, len(scores)))
    for s in scores:
        sub = df[df[score_col] == s]
        if len(sub) == 0:
            continue
        take = min(per, len(sub))
        idx = rng.choice(sub.index.values, size=take, replace=False)
        groups.append(df.loc[idx])
    out = pd.concat(groups, axis=0) if groups else df.sample(n=min(n, len(df)), random_state=seed)
    if len(out) < n and len(df) > len(out):
        extra = df.drop(out.index)
        out2 = extra.sample(n=min(n - len(out), len(extra)), random_state=seed)
        out = pd.concat([out, out2], axis=0)
    return out.sample(frac=1.0, random_state=seed)
