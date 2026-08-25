import hashlib
import json
import os
import random
import shutil
import sys
import time
from typing import Any, Dict, List

import numpy as np


def ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def clamp_int(x: float, low: int, high: int) -> int:
    return int(max(low, min(high, round(float(x)))))


def json_default(o: Any) -> Any:
    if isinstance(o, set):
        return sorted(list(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(obj: Any, path: str) -> None:
    safe_mkdir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=json_default)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(row: Dict[str, Any], path: str) -> None:
    safe_mkdir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def read_text_if_exists(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return text or None


def zip_dir(run_dir: str, zip_path: str | None = None) -> str:
    if zip_path is None:
        zip_path = run_dir.rstrip(os.sep) + ".zip"
    base = zip_path[:-4] if zip_path.endswith(".zip") else zip_path
    return shutil.make_archive(base, "zip", root_dir=run_dir)


def maybe_colab_download(path: str) -> bool:
    if "google.colab" not in sys.modules:
        return False
    try:
        from google.colab import files  # type: ignore
        files.download(path)
        return True
    except Exception:
        return False
