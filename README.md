# Probabilistic PAPILLON

Code and data for **Beyond Direct Identifiers: Probabilistic Privacy Risk Estimation for Privacy-Conscious LLM Query Delegation** (Li Siyan, Zhou Yu, Julia Hirschberg; HAIPS @ COLM 2026).

In the paper, we extend [PAPILLON](https://github.com/siyan-sylvia-li/PAPILLON) with a second privacy objective. The original objective is to avoid leaking explicit PII to the remote model. The new one is to keep the rewritten prompt *k-anonymous*: a combination of quasi-identifiers such as "nurse", "Austin, TX" and "pregnant" should not narrow the user down to a handful of people. We estimate k-anonymity with an LLM-driven, BRANCH-style estimator (Zheng et al., "Probabilistic Reasoning with LLMs for Privacy Risk Estimation", NeurIPS 2025). We also release **PUPA-SD**, 166 real user queries from WildChat and LMSYS-Chat-1M that contain self-disclosures.

## Repository layout

```
prob_papillon/            library code
  disclosures.py          self-disclosure extraction + BRANCH ordering prompts (App. A)
  k_anonymity.py          k-anonymity estimators (Sec. 4.2)
  papillon.py             the PAPILLON pipeline
  llm_judge.py            quality / PII leakage / prompt / k-anonymity judges (Sec. 4.3)
scripts/
  optimize_disclosure_extractor.py   Table 1 + MIPROv2 prompt selection (Sec. 3.1)
  extract_pupa_sd.py                 mine PUPA-SD candidates (Sec. 3.2)
  count_pii_gliner.py                PII distribution of PUPA-SD (Figure 2)
  sanity_check_k_anon.py             Spearman sanity check on PUPA-TNB (Sec. 5.1)
  optimize_papillon.py               SIMBA optimization of one pipeline (Sec. 5.2)
  evaluate_papillon.py               evaluation of one pipeline on PUPA-TNB
  run_model_series.py                optimize + evaluate all local models (Table 2)
  summarize_results.py               print Table 2 from evaluation outputs
data/
  PUPA_SD.csv                        PUPA-SD (optimization data)
  PUPA_TNB.csv                       PUPA-TNB (held-out evaluation data, from PAPILLON)
  disclosure_extraction_eval.json    annotated texts from Zheng et al. (2025) for extractor evaluation
artifacts/
  disclosure_processor_gpt-5-mini.json   selected self-disclosure extractor + ordering prompts
  extractor_candidates_gpt-5-mini.json   MIPROv2-proposed extractor instructions
  extractor_baseline_*.json              Table 1 scores
  k_anon_query_cache.json                cached answers to population / percentage queries
  optimized_prompts/<model>.json         SIMBA-optimized PAPILLON pipelines from the paper
results/                               per-query evaluation outputs behind Table 2
configs/models.csv                     local models used in the paper
```

## Setup

```bash
conda create -n prob_papillon python=3.10 && conda activate prob_papillon
pip install -e ".[data]"          # add ",serve" to install vLLM for hosting local models
cp .env.example .env              # then fill in OPENAI_API_KEY
```

Every LLM call goes through DSPy 3.1.2. Scripts can be run from any directory.

## Data

**PUPA-SD** (`data/PUPA_SD.csv`) has the same columns as PUPA:

| column | description |
|---|---|
| `user_query` | first user turn of the conversation |
| `target_response` | the original assistant response |
| `pii_units` | extracted self-disclosures, formatted `category: span; category: span` |
| `conversation` | the first exchange as `user: …\nassistant: …` |
| `index` | row id |

To regenerate candidates from the raw corpora (the first 20,000 conversations of WildChat-1M and of LMSYS-Chat-1M):

```bash
huggingface-cli login     # LMSYS-Chat-1M is gated
python scripts/extract_pupa_sd.py --output data/PUPA_SD_candidates.csv
```

The released PUPA-SD was produced from such candidates by *manually* removing non-English and sexual content and excluding any PUPA-TNB queries. The script does the PUPA-TNB exclusion. By default it also drops conversations whose `language` field is not English; pass `--keep_non_english` to disable this. It does not filter sexual content.

## Reproducing the paper

**Self-disclosure extractor (Table 1, Sec. 3.1)**

```bash
for m in gpt-4.1-mini gpt-5-mini gpt-4o-mini; do
  python scripts/optimize_disclosure_extractor.py --model_name $m --baseline_only
done
python scripts/optimize_disclosure_extractor.py --model_name gpt-5-mini   # selects the best of 10 proposed instructions
```

The proposed instructions ship in `artifacts/extractor_candidates_gpt-5-mini.json`, so the last command re-scores them instead of proposing new ones. Delete that file to re-run the proposal step. The run overwrites `artifacts/disclosure_processor_gpt-5-mini.json` with the best candidate (F-1 = 72.5 in the paper).

**k-anonymity sanity check (Sec. 5.1)**

```bash
python scripts/sanity_check_k_anon.py      # paper: ρ = −0.4045
```

**PAPILLON optimization and evaluation (Table 2, Figure 3)**

This launches a vLLM server per model, optimizes on PUPA-SD, and evaluates the zero-shot and optimized pipelines on PUPA-TNB:

```bash
python scripts/run_model_series.py --config configs/models.csv --output_dir runs/
python scripts/summarize_results.py runs/
```

For a single model with an already running server:

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct --port 10002
python scripts/optimize_papillon.py --model_name meta-llama/Llama-3.2-3B-Instruct --port 10002 \
    --prompt_output runs/meta-llama_Llama-3.2-3B-Instruct.json
python scripts/evaluate_papillon.py --model_name meta-llama/Llama-3.2-3B-Instruct --port 10002 \
    --prompt_file runs/meta-llama_Llama-3.2-3B-Instruct.json --output_file runs/meta-llama_Llama-3.2-3B-Instruct_after.json
```

The optimized pipelines reported in the paper are in `artifacts/optimized_prompts/`, and their per-query evaluations are in `results/`. To reprint Table 2:

```bash
python scripts/summarize_results.py results/
```

### Models and settings

| role | model | flag |
|---|---|---|
| untrusted remote model | GPT-4o-mini | `--openai_model` |
| judge during optimization | GPT-5-mini | `optimize_papillon.py --judge_model` |
| judge during evaluation | GPT-4o-mini | `evaluate_papillon.py --judge_model` |
| disclosure extraction / k-anonymity | the judge model, except BRANCH query answering (GPT-5-mini) | |
| k-anonymity estimator | `direct` (single call) or `branch` (Sec. 4.2) | `--k_anon_estimator` |

The optimization metric is `(quality + k_anon + prompt well-formedness) / 3`. PII leakage is only measured at evaluation time. SIMBA uses `max_demos=1, bsize=16, num_candidates=6, max_steps=4` on the first 50 PUPA-SD examples.

The `k_anon` score is `log2(k) / log2(400M)`.

## Citation

```bibtex
@inproceedings{siyan2026probabilistic,
  title     = {Beyond Direct Identifiers: Probabilistic Privacy Risk Estimation for Privacy-Conscious LLM Query Delegation},
  author    = {Siyan, Li and Yu, Zhou and Hirschberg, Julia},
  booktitle = {HAIPS Workshop at COLM},
  year      = {2026}
}
```

PUPA-SD is derived from [WildChat](https://huggingface.co/datasets/allenai/WildChat-1M) and [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m); please also follow their licenses. `data/disclosure_extraction_eval.json` comes from Zheng et al. (2025).
