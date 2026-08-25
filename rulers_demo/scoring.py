import json
import time
from typing import Any, Dict, List, Tuple

from .config import ASAP2_TRAITS, EVIDENCE_OPERATORS, TYPE_AWARE_RULE_TABLE
from .evidence import evidence_missing_count, validate_evidence_quotes
from .llm_client import openai_structured
from .text_units import build_segment_index
from .utils import clamp_int, json_default


def score_output_schema(score_min: int = 1, score_max: int = 6, type_aware: bool = True) -> Dict[str, Any]:
    ev_item_props: Dict[str, Any] = {
        "sent_id": {"type": "integer", "minimum": 0},
        "quote": {"type": "string"},
    }
    ev_item_required = ["sent_id", "quote"]
    if type_aware:
        ev_item_props.update({
            "operator": {"type": "string", "enum": EVIDENCE_OPERATORS},
            "para_id": {"type": "integer", "minimum": 0},
            "diagnostic": {"type": "string"},
            "human_review_recommended": {"type": "boolean"},
        })
        ev_item_required += ["operator", "para_id", "diagnostic", "human_review_recommended"]

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "essay_id": {"type": "string"},
            "paragraph_outline": {
                "type": "array",
                "minItems": 1,
                "maxItems": 80,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "para_id": {"type": "integer", "minimum": 0},
                        "topic_sent_id": {"type": "integer", "minimum": 0},
                        "summary": {"type": "string"},
                    },
                    "required": ["para_id", "topic_sent_id", "summary"],
                },
            },
            "trait_scores": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    t: {"type": "integer", "minimum": score_min, "maximum": score_max}
                    for t in ASAP2_TRAITS
                },
                "required": ASAP2_TRAITS,
            },
            "boundary_checks": {
                "type": "object",
                "additionalProperties": False,
                "properties": {t: {"type": "string"} for t in ASAP2_TRAITS},
                "required": ASAP2_TRAITS,
            },
            "checklist_ratings": {
                "type": "array",
                "minItems": 20,
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "decision": {"type": "integer", "minimum": 0, "maximum": 2},
                    },
                    "required": ["id", "decision"],
                },
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "evidence": {
                "type": "object",
                "additionalProperties": False,
                "properties": {t: {"$ref": "#/$defs/ev2"} for t in ASAP2_TRAITS},
                "required": ASAP2_TRAITS,
            },
        },
        "required": [
            "essay_id",
            "paragraph_outline",
            "trait_scores",
            "boundary_checks",
            "checklist_ratings",
            "confidence",
            "evidence",
        ],
        "$defs": {
            "evItem": {
                "type": "object",
                "additionalProperties": False,
                "properties": ev_item_props,
                "required": ev_item_required,
            },
            "ev2": {"type": "array", "items": {"$ref": "#/$defs/evItem"}, "minItems": 2, "maxItems": 2},
        },
    }


def checklist_to_trait_scores(
    bundle: Dict[str, Any],
    checklist_ratings: List[Dict[str, Any]],
    score_min: int = 1,
    score_max: int = 6,
) -> Dict[str, int]:
    dim_by_id = {str(item["id"]): item["dimension"] for item in bundle["checklist"]}
    values: Dict[str, List[int]] = {t: [] for t in ASAP2_TRAITS}
    for rating in checklist_ratings:
        cid = str(rating.get("id", ""))
        if cid not in dim_by_id:
            continue
        values[dim_by_id[cid]].append(int(rating.get("decision", 0)))
    scores: Dict[str, int] = {}
    for trait in ASAP2_TRAITS:
        vals = values[trait]
        if not vals:
            scores[trait] = score_min
            continue
        mu = sum(vals) / (2.0 * len(vals))
        scores[trait] = clamp_int(score_min + (score_max - score_min) * mu, score_min, score_max)
    return scores


def score_one_essay(
    model: str,
    essay_id: str,
    bundle: Dict[str, Any],
    segment_index: Dict[str, Any],
    score_min: int = 1,
    score_max: int = 6,
    temperature: float = 0.0,
    max_tokens: int = 2400,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    system = (
        "You are a strict evidence-based essay rater. "
        "Return JSON ONLY. Follow the schema exactly. Keep outputs compact. "
        "No long chain-of-thought. Use short, auditable justifications."
    )

    checklist_compact = []
    for it in bundle["checklist"]:
        checklist_compact.append({
            "id": it["id"],
            "dimension": it["dimension"],
            "criterion": it["criterion"],
            "operational_definition": it["operational_definition"][:240],
            "positive_cues": [x[:70] for x in (it.get("positive_cues", []) or [])[:4]],
            "negative_cues": [x[:70] for x in (it.get("negative_cues", []) or [])[:4]],
            "item_type": it.get("item_type", "local_quote"),
            "evidence_operator": it.get("evidence_operator", "sentence_quote"),
            "quote_instruction": it.get("quote_instruction", ""),
        })

    evidence_instruction = f"""
3) Provide evidence for EACH trait:
   - EXACTLY 2 evidence objects.
   - Each evidence object must include operator, para_id, sent_id, quote, diagnostic, and human_review_recommended.
   - For local_quote / sentence_quote: quote must be a verbatim substring of the specified sentence text.
   - For span_level / paragraph_span: para_id should identify the paragraph span; quote should be a representative verbatim supporting sentence from that paragraph.
   - For global_diagnostic: diagnostic should summarize the document-level pattern; quote should be a representative supporting sentence span.
   - For weakly_groundable: use lower confidence and set human_review_recommended=true if the item is not directly observable; still provide a short supporting quote when possible.
   - The quote field is checked as a verbatim sentence substring for auditability and downstream variables.
Type-aware evidence rules:
{json.dumps(bundle.get('evidence_rules', {}).get('item_type_rules', TYPE_AWARE_RULE_TABLE), ensure_ascii=False, indent=2)}
"""

    user = f"""
Score the essay with 4 fixed traits ({score_min}–{score_max}):
- ClaimPosition
- EvidenceElaboration
- OrganizationCoherence
- LanguageConventions

You MUST:
1) Build a paragraph outline:
   - For each para_id, pick one topic_sent_id from the sentence list and a short summary.
2) Fill ALL 20 checklist items exactly once, decision in {{0,1,2}}.
{evidence_instruction}
4) Provide a boundary_check string for EACH trait:
   - Explain why not one level higher AND why not one level lower in one compact sentence.
5) Anti-halo constraint:
   - Do NOT output identical scores across ALL 4 traits unless boundary_checks justify equality.

Trait anchors (locked; do not change):
{json.dumps(bundle['traits'], ensure_ascii=False, indent=2, default=json_default)}

Checklist (20 items, 0/1/2 each):
{json.dumps(checklist_compact, ensure_ascii=False, default=json_default)}

Essay paragraphs:
{json.dumps(segment_index['paragraphs'], ensure_ascii=False, default=json_default)}

Essay sentences (use sent_id for evidence):
{json.dumps(segment_index['sentences'], ensure_ascii=False, default=json_default)}

Evidence rules:
{json.dumps(bundle['evidence_rules'], ensure_ascii=False, default=json_default)}

Return the structured output now for essay_id={essay_id}.
"""
    obj, usage, _ = openai_structured(
        model=model,
        system=system,
        user=user,
        schema_name="essay_score_locked",
        schema=score_output_schema(score_min=score_min, score_max=score_max, type_aware=True),
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
    )
    return obj, usage


def postprocess_score_output(
    obj: Dict[str, Any],
    usage: Dict[str, Any],
    essay_id: str,
    human_score: int | None,
    bundle: Dict[str, Any],
    segment_index: Dict[str, Any],
    prefer_checklist: bool = False,
    score_min: int = 1,
    score_max: int = 6,
    latency_sec: float = 0.0,
) -> Dict[str, Any]:
    sent_bank = {int(x["sent_id"]): str(x["text"]) for x in segment_index["sentences"]}
    max_quote_words = int(bundle.get("evidence_rules", {}).get("max_quote_words", 25))
    evidence, invalid = validate_evidence_quotes(sent_bank, obj.get("evidence", {}), max_quote_words=max_quote_words)
    ev_miss = evidence_missing_count(evidence, min_per_trait=int(bundle.get("evidence_rules", {}).get("min_evidence_per_trait", 2)))

    trait_scores = {t: clamp_int(obj.get("trait_scores", {}).get(t, score_min), score_min, score_max) for t in ASAP2_TRAITS}
    checklist_scores = checklist_to_trait_scores(bundle, obj.get("checklist_ratings", []), score_min, score_max)
    used_scores = checklist_scores if prefer_checklist else trait_scores
    raw_score = float(sum(used_scores.values()) / len(ASAP2_TRAITS))
    raw_pred = clamp_int(raw_score, score_min, score_max)
    vals = list(used_scores.values())

    row: Dict[str, Any] = {
        "essay_id": str(essay_id),
        "human_score": human_score,
        "raw_score_cont": raw_score,
        "raw_score": raw_pred,
        "confidence": float(obj.get("confidence", 0.0)),
        "invalid_evidence": int(invalid),
        "ev_miss": int(ev_miss),
        "raw_quote_total": int(sum(len(obj.get("evidence", {}).get(t, []) or []) for t in ASAP2_TRAITS)),
        "valid_quote_total": int(sum(len(evidence.get(t, []) or []) for t in ASAP2_TRAITS)),
        "tie": int(len(set(vals)) == 1),
        "trait_range": int(max(vals) - min(vals)) if vals else 0,
        "total_calls": int(usage.get("n_calls", 1)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "latency_sec": float(latency_sec),
        "structured_json": json.dumps(obj, ensure_ascii=False, default=json_default),
        "verified_evidence_json": json.dumps(evidence, ensure_ascii=False, default=json_default),
    }
    for trait in ASAP2_TRAITS:
        row[f"trait_{trait}"] = int(trait_scores[trait])
        row[f"checklist_{trait}"] = int(checklist_scores[trait])
        row[f"used_{trait}"] = int(used_scores[trait])
        row[f"evidence_n_{trait}"] = int(len(evidence.get(trait, []) or []))
        row[f"boundary_{trait}"] = str(obj.get("boundary_checks", {}).get(trait, ""))
    return row


def safe_score_essay(
    model: str,
    essay_id: str,
    text: str,
    human_score: int | None,
    bundle: Dict[str, Any],
    sent_max_chars: int = 260,
    prefer_checklist: bool = False,
    score_min: int = 1,
    score_max: int = 6,
    max_tokens: int = 2400,
) -> Dict[str, Any]:
    segment_index = build_segment_index(text, sent_max_chars=sent_max_chars)
    t0 = time.time()
    try:
        obj, usage = score_one_essay(
            model=model,
            essay_id=str(essay_id),
            bundle=bundle,
            segment_index=segment_index,
            score_min=score_min,
            score_max=score_max,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return postprocess_score_output(
            obj=obj,
            usage=usage,
            essay_id=str(essay_id),
            human_score=human_score,
            bundle=bundle,
            segment_index=segment_index,
            prefer_checklist=prefer_checklist,
            score_min=score_min,
            score_max=score_max,
            latency_sec=time.time() - t0,
        )
    except Exception as exc:
        return {
            "essay_id": str(essay_id),
            "human_score": human_score,
            "raw_score_cont": float((score_min + score_max) / 2.0),
            "raw_score": clamp_int((score_min + score_max) / 2.0, score_min, score_max),
            "confidence": 0.0,
            "invalid_evidence": 0,
            "ev_miss": len(ASAP2_TRAITS) * 2,
            "raw_quote_total": 0,
            "valid_quote_total": 0,
            "tie": 1,
            "trait_range": 0,
            "total_calls": 0,
            "total_tokens": 0,
            "latency_sec": time.time() - t0,
            "parse_failed": 1,
            "error": str(exc)[:500],
        }
