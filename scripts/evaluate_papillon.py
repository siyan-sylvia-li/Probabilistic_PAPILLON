"""Evaluate a (zero-shot or optimized) PAPILLON pipeline on PUPA-format data (paper Table 2).

Reports average response quality, PII leakage, and normalized k-anonymity of the
prompts sent to the remote model. Omit --prompt_file to evaluate the zero-shot
pipeline ("before optimization").

Example:
    python scripts/evaluate_papillon.py --model_name meta-llama/Llama-3.2-3B-Instruct --port 10002 \
        --prompt_file artifacts/optimized_prompts/meta-llama_Llama-3.2-3B-Instruct.json \
        --output_file results/my_eval.json
"""
import json
from argparse import ArgumentParser

import dotenv
import dspy
import litellm
import pandas
import tqdm

from prob_papillon import DATA_DIR, REPO_ROOT
from prob_papillon.llm_judge import K_ANON_ESTIMATORS, LLMJudge
from prob_papillon.papillon import PAPILLON

dotenv.load_dotenv(REPO_ROOT / ".env")


def mean(xs):
    return sum(xs) / len(xs) if xs else None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, required=True, help="Port of the OpenAI-compatible server hosting the local model")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face identifier of the local model")
    parser.add_argument("--data_file", type=str, default=str(DATA_DIR / "PUPA_TNB.csv"))
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini", help="Untrusted remote model")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--k_anon_estimator", choices=K_ANON_ESTIMATORS, default="direct")
    parser.add_argument("--prompt_file", type=str, default=None, help="Optimized pipeline JSON; omit for zero-shot")
    parser.add_argument("--output_file", type=str, default="evaluation.json")
    args = parser.parse_args()

    local_lm = dspy.LM(f"openai/{args.model_name}", api_base=f"http://0.0.0.0:{args.port}/v1", api_key="", max_tokens=4000)
    dspy.configure(lm=local_lm)
    remote_lm = dspy.LM(model="openai/" + args.openai_model, max_tokens=4000)
    judge_lm = dspy.LM(model="openai/" + args.judge_model, max_tokens=4000)

    pipeline = PAPILLON(remote_lm)
    if args.prompt_file:
        pipeline.load(args.prompt_file)
    llm_judge = LLMJudge(k_anon_estimator=args.k_anon_estimator)

    data = pandas.read_csv(args.data_file)
    records = []
    for _, row in tqdm.tqdm(data.iterrows(), total=len(data)):
        if not isinstance(row["target_response"], str) or not isinstance(row["pii_units"], str):
            continue
        try:
            pred = pipeline(row["user_query"])
        except litellm.exceptions.BadRequestError:
            continue
        with dspy.context(lm=judge_lm):
            # Trailing "||" makes single-unit PII strings parse as a PII list.
            scores = llm_judge(user_query=row["user_query"], og_resp=row["target_response"], new_resp=pred.output,
                               updated_query=pred.prompt, pii_str=row["pii_units"] + "||")
        # Failed pipeline runs (empty prompt or output) are excluded.
        if scores.quality == -1 or scores.k_anon == -1:
            continue
        records.append({
            "quality": scores.quality,
            "k_anon": scores.k_anon,
            "pii_leakage": scores.pii_leakage,
            "user_query": row["user_query"],
            "target_response": row["target_response"],
            "papillon_completion": pred.output,
            "papillon_prompt": pred.prompt,
            "pii_units": row["pii_units"],
        })
        json.dump(records, open(args.output_file, "w"), indent=1)

    print("N EVALUATED           ", len(records))
    print("AVERAGE QUALITY SCORE ", mean([r["quality"] for r in records]))
    print("AVERAGE PII LEAKAGE   ", mean([r["pii_leakage"] for r in records if r["pii_leakage"] is not None]))
    print("AVERAGE K-ANONYMITY   ", mean([r["k_anon"] for r in records]))
