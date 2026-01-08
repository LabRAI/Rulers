# Rulers: Locked Rubrics and Evidence-Anchored Scoring for Robust LLM Evaluation

This repository contains the official implementation of **Rulers** (**R**ubric **U**nification, **L**ocking, and **E**vidence-anchored **R**obust **S**coring), a framework designed to align frozen LLM judges with human grading standards.

The framework reframes evaluation not as an open-ended generation task, but as a **Compiler-Executor** protocol. By locking rubrics into executable checklists and enforcing verbatim evidence extraction, Rulers significantly improves stability and human-agreement across diverse tasks.

## 📖 Methodology

The Rulers pipeline consists of three distinct phases designed to eliminate stochasticity and enforce auditability:

1.  **Phase I: Rubric Unification (The Compiler)**
    * Compiles natural language guidelines into a **Locked Rubric Bundle**.
    * Freezes criteria definitions into a fixed taxonomy and deterministic checklist to eliminate interpretation drift.
    * *Output*: A hashed, immutable JSON specification.

2.  **Phase II: Evidence-Anchored Scoring (The Executor)**
    * The model executes the checklist with a strict constraint: every high score must be supported by `MIN_EV` verbatim quotes extracted from the input.
    * Includes deterministic verification to prevent hallucinated justifications and anti-halo boundary checks.
    * *Output*: Structured evidence logs and raw checklist decisions.

3.  **Phase III: Robust Scoring Alignment (WGR)**
    * Applies **Wasserstein Generative Regression (WGR)** to map the model's internal score distribution to the specific granularity of human labels (e.g., 0.5-step intervals) without fine-tuning parameters.
    * *Output*: Final calibrated scores aligned with human distributions.

---

## 📂 Repository Structure

We provide three standalone, standardized scripts. Each script is self-contained and tailored to a specific dataset's characteristics while sharing the core Rulers architecture.

| Script Name | Target Dataset | Domain | Key Traits |
| :--- | :--- | :--- | :--- |
| `rulers_asap_standardized.py` | **ASAP 2.0** | Argumentative Writing | Content, Evidence, Organization, Language |
| `rulers_summ_standardized.py` | **SummHF** | Text Summarization | Coherence, Accuracy, Coverage |
| `rulers_dress_standardized.py` | **DREsS** | EFL Student Writing | Content, Organization, Language (0.5-step precision) |

---

## 📊 Datasets

This code is benchmarked on the three datasets discussed in the paper:

1.  **ASAP 2.0 (Argumentative Essay Scoring)**
    * **Type**: Student Argumentative Writing.
    * **Focus**: Structural and linguistic quality evaluation.
    * **Setting**: Standard 1-6 holistic or multi-trait scoring.

2.  **SummHF (Summarization Quality)**
    * **Type**: Summaries derived from human feedback (OpenAI).
    * **Focus**: Hallucination detection and factual consistency in high-compression texts.
    * **Setting**: 1-7 Likert scale.

3.  **DREsS (Deep Rubric for EFL Student Scoring)**
    * **Type**: EFL (English as a Foreign Language) Student Essays.
    * **Focus**: Large-scale rubric-based assessment for classroom essays scored by experts.
    * **Setting**: High-precision multi-trait scoring (Content, Organization, Language) with **0.5-point increments** (Scale 1.0–5.0).

---

## 🚀 Usage

### Installation

```bash
pip install openai pandas numpy scikit-learn scipy
```

### Execution

The scripts are designed to be run independently. You must provide your OpenAI API key and the path to your data.

> **Note on Hyperparameters**: The default values for `checklist_n` (checklist size) and `wgr_alpha` (calibration strength) in the scripts are generic placeholders. As noted in the paper, these should be tuned based on your specific validation set to achieve optimal alignment.

#### 1. Argumentative Writing (ASAP 2.0)
*Targeting structural argumentation with strict evidence requirements.*

```bash
python rulers_asap_standardized.py \
  --data_path ./data/asap_test.csv \
  --api_key "YOUR_OPENAI_API_KEY" \
  --model gpt-4o \
  --checklist_n 15 \
  --min_ev 2 \
  --wgr_alpha 1.0
```

#### 2. Summarization (SummHF)
*Targeting factual consistency with a lower evidence threshold (due to short text length).*

```bash
python rulers_summ_standardized.py \
  --data_path ./data/summhf_test.jsonl \
  --api_key "YOUR_OPENAI_API_KEY" \
  --model gpt-4o \
  --checklist_n 12 \
  --min_ev 1 \
  --wgr_alpha 2.5
```

#### 3. EFL Student Writing (DREsS)
*Targeting high-precision (0.5-step) scoring for language learners.*

```bash
python rulers_dress_standardized.py \
  --data_path ./data/dress_full.tsv \
  --api_key "YOUR_OPENAI_API_KEY" \
  --model gpt-4o \
  --checklist_n 18 \
  --min_ev 2 \
  --wgr_alpha 0.5
```

---

## 📂 Output Artifacts

Upon execution, the scripts will generate the following outputs, corresponding to the three phases of the Rulers framework:

1.  **The Locked Rubric Bundle (Phase I)**
    * *Console Output*: `[*] Bundle Locked. Hash: a1b2c3...`
    * *Content*: A JSON object containing the compiled taxonomy and the deterministic checklist (e.g., `C01` to `C15`).
    * *Significance*: This hash guarantees **reproducibility**. Any change in the prompt or model interpretation would alter this hash.

2.  **Evidence-Anchored Logs (Phase II)**
    * *Console Output*: `[*] Phase II Scoring started...`
    * *Content*: A structured log for every input sample, containing:
        * `checklist_decisions`: Binary/Ternary vectors for each checklist item.
        * `evidence_quotes`: Verbatim substrings extracted from the text to support scores (enforcing the *Evidence Support* constraint).
        * `boundary_justification`: A compact rationale for why the score is not higher or lower.

3.  **Calibrated Scores (Phase III)**
    * *Console Output*: `[*] Calibration complete. Final scores generated.`
    * *Content*: A final set of scores processed by the **Wasserstein Generative Regression (WGR)** engine.
    * *Significance*: These scores are distributionally aligned with human grading standards (e.g., matching the specific 0.5-step granularity of the DREsS dataset) without fine-tuning the model parameters.

---

## ⚙️ Key Arguments Explained

* `--checklist_n`: The number of granular checklist items generated in Phase I.
    * *Tip*: Complex rubrics (like DREsS) often benefit from a higher N (e.g., 18-20), while simpler tasks may require fewer.
* `--min_ev`: The minimum number of verbatim quotes required to support a high checklist decision.
    * *Default*: `2` for essays, `1` for short summaries.
* `--wgr_alpha`: The regularization strength for the WGR calibration layer.
    * *Tip*: Adjust this to prevent overfitting when fitting the calibrator on small subsets.


