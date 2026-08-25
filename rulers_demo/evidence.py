from typing import Any, Dict, List, Tuple

from .config import ASAP2_TRAITS


def validate_evidence_quotes(
    sent_bank: Dict[int, str],
    evidence: Dict[str, List[Dict[str, Any]]],
    max_quote_words: int = 25,
) -> Tuple[Dict[str, List[Dict[str, Any]]], int]:
    """Validate that returned evidence quotes are verbatim substrings.

    Type-aware metadata is preserved, but quote validity is intentionally simple:
    a valid quote must be a verbatim substring of its sentence, or at least appear
    somewhere in the sentence bank if the model selected a slightly wrong sent_id.
    """
    invalid = 0
    cleaned: Dict[str, List[Dict[str, Any]]] = {}
    all_sents = "\n".join(sent_bank.values())
    for trait in ASAP2_TRAITS:
        clean_list = []
        for item in evidence.get(trait, []) or []:
            sid = int(item.get("sent_id", -1))
            quote = str(item.get("quote", "")).strip()
            if not quote:
                invalid += 1
                continue
            if len(quote.split()) > max_quote_words:
                quote = " ".join(quote.split()[:max_quote_words]).strip()
            sent = sent_bank.get(sid, "")
            ok = bool(sent and quote in sent)
            if not ok and quote in all_sents:
                ok = True
            if ok:
                rec = {"sent_id": sid, "quote": quote}
                for k in ["operator", "para_id", "diagnostic", "human_review_recommended"]:
                    if k in item:
                        rec[k] = item[k]
                clean_list.append(rec)
            else:
                invalid += 1
        cleaned[trait] = clean_list
    return cleaned, invalid


def evidence_missing_count(evidence: Dict[str, List[Dict[str, Any]]], min_per_trait: int = 2) -> int:
    missing = 0
    for trait in ASAP2_TRAITS:
        missing += max(0, min_per_trait - len(evidence.get(trait, []) or []))
    return int(missing)
