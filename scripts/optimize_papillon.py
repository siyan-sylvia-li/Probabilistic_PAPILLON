"""Optimize a PAPILLON pipeline for quality + k-anonymity with SIMBA (paper Sec. 5.2).

The local model must be served behind an OpenAI-compatible endpoint (e.g. vLLM)
at http://0.0.0.0:<port>/v1.

Metric per example: (quality + k_anon + prompt well-formedness) / 3. PII leakage is
not part of the objective.

Example:
    vllm serve meta-llama/Llama-3.2-3B-Instruct --port 10002
    python scripts/optimize_papillon.py --model_name meta-llama/Llama-3.2-3B-Instruct --port 10002 \
        --prompt_output artifacts/optimized_prompts/my_run.json
"""
import json
from argparse import ArgumentParser

import dotenv
import dspy
import pandas
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt import SIMBA

from prob_papillon import DATA_DIR, REPO_ROOT
from prob_papillon.llm_judge import K_ANON_ESTIMATORS, LLMJudge
from prob_papillon.papillon import PAPILLON

dotenv.load_dotenv(REPO_ROOT / ".env")


def load_splits(data_file):
    """Rows 0-149 → train, 150-299 → val, rest → test; rows without disclosures are dropped."""
    df = pandas.read_csv(data_file, index_col=False)
    train, val, test = [], [], []
    for i, row in df.iterrows():
        if pandas.isna(row["pii_units"]) or not isinstance(row["pii_units"], str) or len(row["pii_units"]) == 0:
            continue
        example = dspy.Example({
            "target_response": row["target_response"],
            "user_query": row["user_query"],
        }).with_inputs("user_query")
        if i < 150:
            train.append(example)
        elif i < 300:
            val.append(example)
        else:
            test.append(example)
    return train, val, test


def make_metric(llm_judge: LLMJudge, judge_lm: dspy.LM):
    def metric(gold, pred, trace=None):
        if len(pred.prompt) == 0:
            return 0
        with dspy.context(lm=judge_lm):
            scores = llm_judge(user_query=gold.user_query, og_resp=gold.target_response,
                               new_resp=pred.output, updated_query=pred.prompt)
        if scores.k_anon == -1:
            return 0
        total = (scores.quality + scores.k_anon + scores.prompt) / 3
        if trace is not None:
            return total >= 1
        return total
    return metric


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, required=True, help="Port of the OpenAI-compatible server hosting the local model")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face identifier of the local model")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini", help="Untrusted remote model")
    parser.add_argument("--judge_model", type=str, default="gpt-5-mini")
    parser.add_argument("--k_anon_estimator", choices=K_ANON_ESTIMATORS, default="direct")
    parser.add_argument("--prompt_output", type=str, required=True, help="JSON path for the optimized pipeline")
    parser.add_argument("--data_file", type=str, default=str(DATA_DIR / "PUPA_SD.csv"))
    args = parser.parse_args()
    assert args.prompt_output.endswith(".json")

    local_lm = dspy.LM("openai/" + args.model_name, api_base=f"http://0.0.0.0:{args.port}/v1", api_key="", max_tokens=4000)
    dspy.configure(lm=local_lm)
    remote_lm = dspy.LM(model="openai/" + args.openai_model, max_tokens=4000)
    judge_lm = dspy.LM(model="openai/" + args.judge_model)

    metric = make_metric(LLMJudge(k_anon_estimator=args.k_anon_estimator), judge_lm)
    train, val, _ = load_splits(args.data_file)
    zeroshot = PAPILLON(remote_lm)
    evaluate = Evaluate(metric=metric, devset=val, num_threads=4, display_progress=True,
                        display_table=5, max_errors=10, provide_traceback=True)

    eval_scores = {}
    try:
        eval_scores["before_optimization"] = evaluate(zeroshot).score
    except Exception as e:
        print(f"Zero-shot evaluation failed: {e}")
        eval_scores["before_optimization"] = None
    print(eval_scores)

    try:
        teleprompter = SIMBA(metric=metric, max_demos=1, num_threads=8, bsize=16, num_candidates=6, max_steps=4)
        optimized = teleprompter.compile(zeroshot, trainset=train[:50])
        # Held-out portion of the optimization data (the remaining train rows plus val).
        eval_scores["after_optimization"] = evaluate(optimized, devset=train[50:] + val).score
        print(eval_scores)
        optimized.save(args.prompt_output)
    except ValueError as e:
        print(e)
        local_lm.inspect_history()

    json.dump(eval_scores, open(args.prompt_output.replace(".json", "_eval_scores.json"), "w"))
