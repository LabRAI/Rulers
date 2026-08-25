# RULERS ASAP2.0 Demo

This repository is a **compact code demo** for the paper's RULERS method. It is designed to show how the method can be implemented on **ASAP2.0** with a locked rubric bundle, type-aware evidence grounding, and WGR calibration.

## What this demo implements

The demo follows the main RULERS pipeline:

1. **Rubric specification and locking**
   - Convert the official ASAP2.0 holistic rubric into a locked task-level bundle.
   - The bundle contains four fixed traits, twenty checklist items, score anchors, evidence rules, and a bundle hash.
   - In this demo, checklist items are **type-aware**: each item has an `item_type` and an `evidence_operator`.

2. **Evidence-grounded scoring**
   - Segment each essay into paragraph and sentence banks.
   - Ask the frozen LLM judge to execute the locked bundle.
   - Return structured JSON with trait scores, checklist decisions, boundary checks, and two evidence objects per trait.
   - Verify whether evidence quotes are verbatim substrings of the sentence bank.

3. **WGR calibration**
   - Fit a lightweight calibration model on labeled calibration examples only.
   - The demo uses second-order polynomial features, ridge regression, and monotone quantile mapping to align model signals with the human score scale.
   - The fitted mapping is then applied to test examples.

## Repository structure

```text
rulers_asap2_demo/
├── README.md
├── requirements.txt
├── scripts/
│   └── run_asap2_demo.py
├── rulers_demo/
│   ├── __init__.py
│   ├── calibration.py       # WGR calibration: ridge + monotone quantile mapping
│   ├── config.py            # ASAP2.0 rubric, traits, evidence-type table, defaults
│   ├── data_asap.py         # ASAP2.0 train/test loading and schema inference
│   ├── evidence.py          # evidence validation helpers
│   ├── llm_client.py        # OpenAI/OpenRouter-compatible structured output calls
│   ├── metrics.py           # QWK and basic metrics
│   ├── pipeline.py          # end-to-end ASAP2.0 demo pipeline
│   ├── rubric.py            # Phase I rubric-bundle compilation and validation
│   ├── scoring.py           # Phase II scoring schema, prompt, postprocessing
│   └── text_units.py        # paragraph/sentence segmentation
└── examples/
    └── asap2_rubric.txt     # placeholder showing how to override the built-in rubric
```

## Installation

```bash
git clone <your-repo-url>
cd rulers_asap2_demo
pip install -r requirements.txt
```

The demo uses the OpenAI Python SDK with an OpenAI-compatible endpoint.

### Option A: OpenRouter

```bash
export OPENROUTER_API_KEY="sk-or-..."
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

Then use an OpenRouter model slug, for example:

```bash
--model openai/gpt-4o-mini
```

### Option B: OpenAI API

```bash
export OPENAI_API_KEY="sk-..."
```

Then use an OpenAI model name, for example:

```bash
--model gpt-4o-mini
```

## Data

This repository does **not** redistribute ASAP2.0. Prepare the dataset locally and pass its directory with `--data-dir`.

The loader searches under `--data-dir` for train/test CSV files. It also tries to unzip `.zip` files under the directory. If the zip files require the ASAP2.0 password, pass it with `--zip-password`.

Example:

```bash
python scripts/run_asap2_demo.py \
  --data-dir ./asap_data \
  --model openai/gpt-4o-mini \
  --calib-n 200 \
  --test-max 50 \
  --max-workers 4
```

`--test-max 50` is useful for a quick smoke test. Remove it or set `--test-max 0` for the full labeled test set.

## Basic usage

### 1. Run the full demo with a newly induced type-aware bundle

```bash
python scripts/run_asap2_demo.py \
  --data-dir ./asap_data \
  --model openai/gpt-4o-mini \
  --calib-n 200 \
  --test-max 100 \
  --rubric-version asap2_demo_v1 \
  --max-workers 4
```

The first run will call the LLM once to compile the locked rubric bundle, then call the LLM for calibration and test essays.

### 2. Reuse an existing locked rubric bundle

After the first run, the output folder contains:

```text
rubric_bundle_locked.json
```

To avoid recompiling the rubric, reuse it:

```bash
python scripts/run_asap2_demo.py \
  --data-dir ./asap_data \
  --model openai/gpt-4o-mini \
  --rubric-bundle-path ./rulers_demo_outputs/<run_name>/rubric_bundle_locked.json \
  --calib-n 200 \
  --test-max 100
```

This is closer to the intended RULERS usage: the rubric bundle is constructed once, locked, and reused unchanged for all examples in the same task.

### 3. Disable WGR calibration

```bash
python scripts/run_asap2_demo.py \
  --data-dir ./asap_data \
  --model openai/gpt-4o-mini \
  --no-calibration \
  --test-max 100
```

This uses the raw average of the four selected trait scores as the final score.

## Outputs

Each run creates a timestamped folder under `rulers_demo_outputs/`, for example:

```text
rulers_demo_outputs/asap2_typeaware_YYYYMMDD_HHMMSS/
├── calib_scored.jsonl
├── test_scored.jsonl
├── calib_table.csv
├── test_table.csv
├── rubric_bundle_locked.json
├── summary.json
└── asap2_typeaware_YYYYMMDD_HHMMSS.zip
```

Important files:

- `rubric_bundle_locked.json`: the locked type-aware rubric bundle.
- `calib_table.csv`: calibration examples with raw model signals and final calibrated scores.
- `test_table.csv`: test examples with raw model signals and final calibrated scores.
- `summary.json`: run settings, bundle hash, metrics, and cost summary.
- `*_scored.jsonl`: per-example structured scoring records.

Important columns:

- `human_score`: gold human score if available.
- `raw_score`: uncalibrated rounded score.
- `final_score`: final score after WGR calibration.
- `final_score_cont`: continuous calibrated score before rounding.
- `trait_*`: LLM-returned trait scores.
- `checklist_*`: checklist-derived trait scores.
- `used_*`: the trait signal actually used downstream.
- `invalid_evidence`: number of evidence quotes that failed verbatim validation.
- `ev_miss`: missing verified evidence count.
- `verified_evidence_json`: evidence retained after validation.

## What to change for another task

To adapt this demo to a new rubric-based scoring task, you need to provide or modify the following components.

### 1. Data loader

Create a new loader similar to `rulers_demo/data_asap.py`. It should return:

```python
train_df, test_df, meta
```

where `meta` must include:

```python
{
  "id_col": "...",        # unique example id
  "text_col": "...",      # text to be evaluated
  "score_col": "...",     # human score column for calibration/evaluation
  "prompt_id_col": "..."   # optional; can be None
}
```

The calibration split must have human scores. The test split needs human scores only if you want to compute QWK or other evaluation metrics.

### 2. Official rubric text

Provide the task's human-authored rubric as a `.txt` file and pass:

```bash
--rubric-txt path/to/new_rubric.txt
```

For a substantially different task, you should not rely on the built-in ASAP2.0 rubric in `config.py`.

### 3. Score scale

Set the task score range:

```bash
--score-min 1 --score-max 6
```

For example, if the task uses a 0--5 scale:

```bash
--score-min 0 --score-max 5
```

If the task constructs a total score from multiple traits, compute that target score in the data loader before calibration.

### 4. Trait list and rubric-bundle schema

This ASAP2.0 demo uses four writing traits:

```python
ClaimPosition
EvidenceElaboration
OrganizationCoherence
LanguageConventions
```

For another task, update:

- `ASAP2_TRAITS` in `rulers_demo/config.py`
- the trait schema in `rulers_demo/rubric.py`
- the score-output schema in `rulers_demo/scoring.py`
- feature construction in `rulers_demo/calibration.py`

If the new task still has four comparable traits, you can rename them. If it has a different number of traits, update schema constraints and prompts accordingly.

### 5. Checklist size

This demo uses exactly 20 checklist items because the ASAP2.0 demo is designed that way. For another task, decide whether to keep 20 or choose a task-specific number. If you change it, update:

- `minItems` / `maxItems` in the rubric schema
- `minItems` / `maxItems` in the scoring schema
- prompt text that says “ALL 20 checklist items”

### 6. Evidence types and operators

This demo uses four evidence types:

```python
local_quote
span_level
global_diagnostic
weakly_groundable
```

and four operators:

```python
sentence_quote
paragraph_span
document_diagnostic_with_supporting_spans
lower_confidence_human_review
```

For another task, revise `TYPE_AWARE_RULE_TABLE` in `config.py`. For example:

- summarization may need factual-consistency spans and source-document support;
- structured-input generation may need RDF-triple coverage evidence;
- EFL writing may need grammar/mechanics evidence and discourse-level evidence.

If you add new operators, also update the evidence schema and validation logic in `scoring.py` and `evidence.py`.

### 7. Calibration features

The current WGR features include trait scores, confidence, evidence validity, evidence missingness, and quote counts. For a new task, inspect whether these are still meaningful. Update `rulers_demo/calibration.py` if the new task requires additional signals, such as:

- source coverage rate,
- hallucination indicators,
- factual consistency flags,
- trait-specific confidence,
- length or completeness features.

### 8. Metrics

This demo reports QWK when human test labels are available. If your task uses a different metric, modify `rulers_demo/metrics.py` and `pipeline.py`.

## Notes and limitations

- The demo sends evaluated text to the configured LLM provider. Do not use private or sensitive text unless your data-governance policy allows it.
- The first run can be expensive because it scores both calibration and test examples through an LLM.
- If your provider/model does not support strict JSON schema output, use a compatible model or precompile the bundle with a compatible model and then reuse it.
- The locked bundle should be treated as a task-level artifact. Do not regenerate it for every example.
- Automated scores should be used with human oversight in high-stakes settings.
