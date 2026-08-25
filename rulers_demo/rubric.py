import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd

from .config import (
    ASAP2_TRAITS,
    BUILTIN_ASAP2_HOLISTIC_RUBRIC,
    CHECKLIST_ITEM_TYPES,
    EVIDENCE_OPERATORS,
    TYPE_AWARE_RULE_TABLE,
)
from .data_asap import stratified_sample
from .llm_client import openai_structured
from .text_units import pick_representatives_per_score
from .utils import load_json, read_text_if_exists, save_json, sha1, ts, json_default


def load_rubric_text(rubric_txt: str = "") -> Tuple[str, str]:
    custom = read_text_if_exists(rubric_txt)
    if custom:
        return custom, f"TXT:{rubric_txt}"
    return BUILTIN_ASAP2_HOLISTIC_RUBRIC, "BUILTIN_ASAP2_HOLISTIC_RUBRIC"


def rubric_bundle_schema(type_aware: bool = True) -> Dict[str, Any]:
    check_item_props: Dict[str, Any] = {
        "id": {"type": "string"},
        "dimension": {"type": "string", "enum": ASAP2_TRAITS},
        "criterion": {"type": "string"},
        "operational_definition": {"type": "string"},
        "positive_cues": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 8},
        "negative_cues": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 8},
    }
    check_item_required = [
        "id",
        "dimension",
        "criterion",
        "operational_definition",
        "positive_cues",
        "negative_cues",
    ]

    evidence_props: Dict[str, Any] = {
        "quote_must_be_verbatim": {"type": "boolean"},
        "max_quote_words": {"type": "integer"},
        "min_evidence_per_trait": {"type": "integer"},
    }
    evidence_required = ["quote_must_be_verbatim", "max_quote_words", "min_evidence_per_trait"]

    if type_aware:
        check_item_props.update({
            "item_type": {"type": "string", "enum": CHECKLIST_ITEM_TYPES},
            "evidence_operator": {"type": "string", "enum": EVIDENCE_OPERATORS},
            "quote_instruction": {"type": "string"},
        })
        check_item_required += ["item_type", "evidence_operator", "quote_instruction"]
        evidence_props["item_type_rules"] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                k: {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "applies_to": {"type": "string"},
                        "evidence_operator": {"type": "string"},
                        "anchor": {"type": "string"},
                        "instruction": {"type": "string"},
                    },
                    "required": ["applies_to", "evidence_operator", "anchor", "instruction"],
                }
                for k in CHECKLIST_ITEM_TYPES
            },
            "required": CHECKLIST_ITEM_TYPES,
        }
        evidence_required.append("item_type_rules")

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rubric_version": {"type": "string"},
            "bundle_hash": {"type": "string"},
            "traits": {
                "type": "object",
                "additionalProperties": False,
                "properties": {t: {"$ref": "#/$defs/traitAnchors"} for t in ASAP2_TRAITS},
                "required": ASAP2_TRAITS,
            },
            "checklist": {
                "type": "array",
                "items": {"$ref": "#/$defs/checkItem"},
                "minItems": 20,
                "maxItems": 20,
            },
            "evidence_rules": {
                "type": "object",
                "additionalProperties": False,
                "properties": evidence_props,
                "required": evidence_required,
            },
        },
        "required": ["rubric_version", "bundle_hash", "traits", "checklist", "evidence_rules"],
        "$defs": {
            "traitAnchors": {
                "type": "object",
                "additionalProperties": False,
                "properties": {str(i): {"type": "string"} for i in range(1, 7)},
                "required": [str(i) for i in range(1, 7)],
            },
            "checkItem": {
                "type": "object",
                "additionalProperties": False,
                "properties": check_item_props,
                "required": check_item_required,
            },
        },
    }


def validate_rubric_bundle(bundle: Dict[str, Any], type_aware: bool = True) -> Tuple[bool, str]:
    try:
        if not str(bundle.get("rubric_version", "")).strip():
            return False, "missing rubric_version"
        if not str(bundle.get("bundle_hash", "")).strip():
            return False, "missing bundle_hash"
        for trait in ASAP2_TRAITS:
            anchors = bundle["traits"][trait]
            for i in range(1, 7):
                if str(i) not in anchors or not str(anchors[str(i)]).strip():
                    return False, f"missing {trait}[{i}]"
        checklist = bundle["checklist"]
        if len(checklist) != 20:
            return False, f"checklist size {len(checklist)} != 20"
        seen = set()
        for item in checklist:
            cid = str(item["id"])
            if cid in seen:
                return False, f"duplicate checklist id {cid}"
            seen.add(cid)
            if item["dimension"] not in ASAP2_TRAITS:
                return False, f"bad dimension for {cid}"
            if type_aware:
                if item.get("item_type") not in CHECKLIST_ITEM_TYPES:
                    return False, f"bad item_type for {cid}"
                if item.get("evidence_operator") not in EVIDENCE_OPERATORS:
                    return False, f"bad evidence_operator for {cid}"
        rules = bundle["evidence_rules"]
        if int(rules["max_quote_words"]) <= 0:
            return False, "bad max_quote_words"
        if int(rules["min_evidence_per_trait"]) != 2:
            return False, "min_evidence_per_trait must be 2"
        if type_aware and "item_type_rules" not in rules:
            return False, "missing item_type_rules"
        return True, "ok"
    except Exception as exc:
        return False, f"exception: {exc}"


def compile_rubric_bundle(
    model: str,
    holistic_text: str,
    reps_by_score: Dict[int, List[Dict[str, Any]]],
    rubric_version: str,
    seed: int,
    rubric_temperature: float = 0.2,
    type_aware: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    examples = []
    for score in sorted(reps_by_score.keys()):
        examples.append({
            "score": int(score),
            "examples": [
                {"essay_id": r["essay_id"], "snippet": r["snippet"]}
                for r in reps_by_score[score]
            ],
        })

    system = (
        "You are an expert writing-assessment designer. "
        "You are a RUBRIC COMPILER, not a rubric author. "
        "Output STRICT JSON matching schema only."
    )
    type_block = f"""
TYPE-AWARE CHECKLIST REQUIREMENT:
Each checklist item MUST include:
- item_type in {CHECKLIST_ITEM_TYPES}
- evidence_operator in {EVIDENCE_OPERATORS}
- quote_instruction: a short instruction for how evidence should be cited for this item.

Use the following mapping:
{json.dumps(TYPE_AWARE_RULE_TABLE, ensure_ascii=False, indent=2)}

Design principle:
- local_quote: concrete, directly observable sentence-level textual cues.
- span_level: paragraph/discourse development criteria.
- global_diagnostic: holistic document-level qualities, with supporting spans.
- weakly_groundable: subjective preference-like criteria; use lower confidence / human review.
""" if type_aware else "Checklist items must be auditable from verbatim text evidence."

    user = f"""
We are scoring essays on a 1–6 scale based on the OFFICIAL holistic rubric below.

Your job is to compile this rubric into a FIXED analytic rubric for this dataset:
- EXACTLY 4 traits (fixed names; DO NOT invent new traits):
  1) ClaimPosition
  2) EvidenceElaboration
  3) OrganizationCoherence
  4) LanguageConventions
- EXACTLY 20 checklist checkpoints total (decisionable 0/1/2 each):
  0=not present, 1=partially present, 2=clearly present.
- Anchors per trait: for scores 1..6, 1–2 sentences each, decisionable.

Hard constraints:
- Must COVER the official rubric key concepts (position/critical thinking, evidence, organization/coherence, language/grammar).
- DO NOT reinvent a new scoring system; keep faithful to the official rubric and examples.
- evidence_rules.min_evidence_per_trait MUST be 2.

Output IDs:
- checklist ids must be "C01".."C20".

{type_block}

Rubric version (must copy verbatim): {rubric_version}
Seed: {seed}

OFFICIAL HOLISTIC RUBRIC TEXT:
{holistic_text}

TRAIN EXAMPLES (score-labeled snippets; use them only as calibration anchors, not as new rubric invention):
{json.dumps(examples, ensure_ascii=False, indent=2, default=json_default)}
"""
    obj, usage, _ = openai_structured(
        model=model,
        system=system,
        user=user,
        schema_name="rubric_bundle_locked",
        schema=rubric_bundle_schema(type_aware=type_aware),
        temperature=float(rubric_temperature),
        max_tokens=3600 if type_aware else 3000,
        max_retries=3,
    )
    obj["rubric_version"] = str(rubric_version)
    obj["evidence_rules"]["min_evidence_per_trait"] = 2
    if type_aware:
        obj["evidence_rules"]["item_type_rules"] = TYPE_AWARE_RULE_TABLE
    obj["checklist"] = sorted(list(obj.get("checklist", [])), key=lambda x: str(x.get("id", "")))
    tmp = dict(obj)
    tmp["bundle_hash"] = ""
    obj["bundle_hash"] = sha1(json.dumps(tmp, ensure_ascii=False, sort_keys=True))
    ok, reason = validate_rubric_bundle(obj, type_aware=type_aware)
    if not ok:
        raise RuntimeError(f"Rubric bundle validation failed: {reason}")
    return obj, usage


def build_or_load_rubric_bundle(
    train_df: pd.DataFrame,
    meta: Dict[str, Any],
    run_dir: str,
    model: str,
    seed: int,
    rubric_txt: str = "",
    rubric_bundle_path: str = "",
    rubric_version: str = "asap2_demo_v1",
    rubric_train_n: int = 240,
    rubric_reps_per_score: int = 3,
    rubric_temperature: float = 0.2,
    force_regen: bool = False,
    calibration_df: pd.DataFrame | None = None,
) -> Tuple[Dict[str, Any], str]:
    holistic_text, holistic_src = load_rubric_text(rubric_txt)
    if rubric_bundle_path and os.path.exists(rubric_bundle_path):
        bundle = load_json(rubric_bundle_path)
        ok, reason = validate_rubric_bundle(bundle, type_aware=True)
        if not ok:
            raise RuntimeError(f"Invalid rubric bundle: {reason}")
        return bundle, holistic_src

    cache_key = sha1(holistic_text + f"|{rubric_version}|{seed}|{rubric_train_n}|{rubric_reps_per_score}|typeaware")
    bundle_path = os.path.join(run_dir, f"rubric_bundle_{rubric_version}_{cache_key}.json")
    if not force_regen and os.path.exists(bundle_path):
        bundle = load_json(bundle_path)
        ok, reason = validate_rubric_bundle(bundle, type_aware=True)
        if ok:
            return bundle, holistic_src

    text_col = meta["text_col"]
    score_col = meta["score_col"]
    if calibration_df is not None:
        train_sample = calibration_df.copy()
    else:
        train_sample = stratified_sample(train_df, score_col, min(rubric_train_n, len(train_df)), seed).copy()
    reps = pick_representatives_per_score(
        train_sample,
        score_col=score_col,
        text_col=text_col,
        reps_per_score=rubric_reps_per_score,
        seed=seed,
    )
    bundle, usage = compile_rubric_bundle(
        model=model,
        holistic_text=holistic_text,
        reps_by_score=reps,
        rubric_version=rubric_version,
        seed=seed,
        rubric_temperature=rubric_temperature,
        type_aware=True,
    )
    bundle["_meta"] = {
        "created_at": ts(),
        "model": model,
        "usage": usage,
        "holistic_src": holistic_src,
        "note": "Locked task-level type-aware rubric bundle for the ASAP2.0 demo.",
    }
    save_json(bundle, bundle_path)
    return bundle, holistic_src
