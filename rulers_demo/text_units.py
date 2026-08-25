import re
from typing import Any, Dict, List

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_paragraphs(text: str, max_paras: int = 80) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n+", t) if p.strip()]
    if len(paras) <= 1:
        paras = [p.strip() for p in t.split("\n") if p.strip()]
    return paras[:max_paras]


def split_sentences(text: str, max_sents: int = 320) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    sents = [s.strip() for s in _SENT_SPLIT.split(t) if s.strip()]
    return sents[:max_sents]


def build_segment_index(text: str, sent_max_chars: int = 280) -> Dict[str, Any]:
    paras = split_paragraphs(text)
    out_paras = []
    out_sents = []
    sent_id = 0
    for i, p in enumerate(paras):
        out_paras.append({"para_id": i, "text": p[:1200]})
        for s in split_sentences(p):
            s2 = s.strip()
            if not s2:
                continue
            if len(s2) > sent_max_chars:
                s2 = s2[:sent_max_chars].rstrip() + "…"
            out_sents.append({"sent_id": sent_id, "para_id": i, "text": s2})
            sent_id += 1
    if not out_sents:
        for s in split_sentences(text):
            s2 = s[:sent_max_chars].rstrip()
            out_sents.append({"sent_id": len(out_sents), "para_id": 0, "text": s2})
        out_paras = [{"para_id": 0, "text": (text or "")[:1200]}]
    return {"paragraphs": out_paras, "sentences": out_sents}


def build_snippet(text: str, max_chars: int = 900) -> str:
    seg = build_segment_index(text, sent_max_chars=240)
    sents = seg["sentences"]
    if not sents:
        return (text or "")[:max_chars]
    picks = [sents[0]["text"]]
    if len(sents) >= 3:
        picks.append(sents[len(sents) // 2]["text"])
        picks.append(sents[-1]["text"])
    return " ".join([p for p in picks if p])[:max_chars]


def pick_representatives_per_score(df, score_col: str, text_col: str, reps_per_score: int, seed: int):
    import numpy as np

    out = {}
    scores = sorted(df[score_col].dropna().unique().tolist())
    rng = np.random.RandomState(seed)
    for s in scores:
        sub = df[df[score_col] == s].copy()
        if len(sub) == 0:
            continue
        if len(sub) > 200:
            sub = sub.sample(n=200, random_state=seed)
        idx = rng.choice(sub.index.values, size=min(reps_per_score, len(sub)), replace=False)
        reps = []
        for j in idx.tolist():
            row = sub.loc[j]
            reps.append({
                "essay_id": str(row.get("essay_id", row.name)),
                "score": int(row[score_col]),
                "snippet": build_snippet(str(row[text_col])[:1400]),
            })
        out[int(s)] = reps
    return out
