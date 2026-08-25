from dataclasses import dataclass
from typing import List

DEFAULT_MODEL = "openai/gpt-4o-mini"
RANDOM_SEED = 42
OUT_DIR_DEFAULT = "./rulers_demo_outputs"

ASAP2_TRAITS: List[str] = [
    "ClaimPosition",
    "EvidenceElaboration",
    "OrganizationCoherence",
    "LanguageConventions",
]

CHECKLIST_ITEM_TYPES = [
    "local_quote",
    "span_level",
    "global_diagnostic",
    "weakly_groundable",
]

EVIDENCE_OPERATORS = [
    "sentence_quote",
    "paragraph_span",
    "document_diagnostic_with_supporting_spans",
    "lower_confidence_human_review",
]

TYPE_AWARE_RULE_TABLE = {
    "local_quote": {
        "applies_to": "claim, concrete example, source quote, grammar/mechanics error, explicit textual cue",
        "evidence_operator": "sentence_quote",
        "anchor": "sentence_id",
        "instruction": "cite a short verbatim sentence-level quote",
    },
    "span_level": {
        "applies_to": "coherence, organization, argument development, progression of ideas",
        "evidence_operator": "paragraph_span",
        "anchor": "paragraph_id plus supporting sentence quote",
        "instruction": "identify a paragraph-level span and cite a representative supporting sentence quote",
    },
    "global_diagnostic": {
        "applies_to": "tone, creativity, overall flow, holistic quality pattern",
        "evidence_operator": "document_diagnostic_with_supporting_spans",
        "anchor": "document-level diagnostic plus supporting sentence quotes",
        "instruction": "write a compact document-level diagnostic and cite supporting sentence-level spans",
    },
    "weakly_groundable": {
        "applies_to": "subjective preference or weakly grounded judgment",
        "evidence_operator": "lower_confidence_human_review",
        "anchor": "optional supporting sentence quote plus human-review flag",
        "instruction": "lower confidence, mark human_review_recommended=true when evidence is not directly observable",
    },
}

BUILTIN_ASAP2_HOLISTIC_RUBRIC = """
Holistic Rating Form

After reading each essay and completing the analytical rating form, assign a holistic score based on the rubric below.
For the following evaluations you will need to use a grading scale between 1 (minimum) and 6 (maximum).
As with the analytical rating form, the distance between each grade (e.g., 1-2, 3-4, 4-5) should be considered equal.

SCORE OF 6: An essay in this category demonstrates clear and consistent mastery, although it may have a few minor errors.
A typical essay effectively and insightfully develops a point of view on the issue and demonstrates outstanding critical thinking;
the essay uses clearly appropriate examples, reasons, and other evidence taken from the source text(s) to support its position;
the essay is well organized and clearly focused, demonstrating clear coherence and smooth progression of ideas;
the essay exhibits skillful use of language, using a varied, accurate, and apt vocabulary and demonstrates meaningful variety in sentence structure;
the essay is free of most errors in grammar, usage, and mechanics.

SCORE OF 5: An essay in this category demonstrates reasonably consistent mastery, although it will have occasional errors or lapses in quality.
A typical essay effectively develops a point of view on the issue and demonstrates strong critical thinking;
the essay generally using appropriate examples, reasons, and other evidence taken from the source text(s) to support its position;
the essay is well organized and focused, demonstrating coherence and progression of ideas;
the essay exhibits facility in the use of language, using appropriate vocabulary demonstrates variety in sentence structure;
the essay is generally free of most errors in grammar, usage, and mechanics.

SCORE OF 4: An essay in this category demonstrates adequate mastery, although it will have lapses in quality.
A typical essay develops a point of view on the issue and demonstrates competent critical thinking;
the essay using adequate examples, reasons, and other evidence taken from the source text(s) to support its position;
the essay is generally organized and focused, demonstrating some coherence and progression of ideas;
the essay may demonstrate inconsistent facility in the use of language, using generally appropriate vocabulary demonstrates some variety in sentence structure;
the essay may have some errors in grammar, usage, and mechanics.

SCORE OF 3: An essay in this category demonstrates developing mastery, and is marked by ONE OR MORE of the following weaknesses:
develops a point of view on the issue, demonstrating some critical thinking, but may do so inconsistently or use inadequate examples, reasons,
or other evidence taken from the source texts to support its position;
the essay is limited in its organization or focus, or may demonstrate some lapses in coherence or progression of ideas;
the essay may demonstrate facility in the use of language, but sometimes uses weak vocabulary or inappropriate word choice and/or lacks variety
or demonstrates problems in sentence structure;
the essay may contain an accumulation of errors in grammar, usage, and mechanics.

SCORE OF 2: An essay in this category demonstrates little mastery, and is flawed by ONE OR MORE of the following weaknesses:
develops a point of view on the issue that is vague or seriously limited, and demonstrates weak critical thinking;
the essay provides inappropriate or insufficient examples, reasons, or other evidence taken from the source text to support its position;
the essay is poorly organized and/or focused, or demonstrates serious problems with coherence or progression of ideas;
the essay displays very little facility in use of language, using very limited vocabulary or incorrect word choice and/or demonstrates frequent problems
in sentence structure;
the essay contains errors in grammar, usage, and mechanics so serious that meaning is somewhat obscured.

SCORE OF 1: An essay in this category demonstrates very little or no mastery, and is severely flawed by ONE OR MORE of the following weaknesses:
develops no viable point of view on the issue, or provides little or no evidence to support its position;
the essay is disorganized or unfocused, resulting in a disjointed or incoherent essay;
the essay displays fundamental errors in vocabulary and/or demonstrates severe flaws in sentence structure;
the essay contains pervasive errors in grammar, usage, or mechanics that persistently interfere with meaning.
""".strip()


@dataclass
class DemoConfig:
    model: str = DEFAULT_MODEL
    out_dir: str = OUT_DIR_DEFAULT
    score_min: int = 1
    score_max: int = 6
    seed: int = RANDOM_SEED
    max_workers: int = 4
    calib_n: int = 200
    test_max: int = 0
    rubric_version: str = "asap2_demo_v1"
    rubric_temperature: float = 0.2
    score_max_tokens: int = 2400
    prefer_checklist: bool = False
    enable_calibration: bool = True
