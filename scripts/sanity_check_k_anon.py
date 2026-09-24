"""Spearman correlation between #PII units and estimated log2(k) on PUPA-TNB (paper Sec. 5.1).

Usage:
    python scripts/sanity_check_k_anon.py
"""
from argparse import ArgumentParser

import dotenv
import dspy
import pandas
from scipy.stats import spearmanr

from prob_papillon import DATA_DIR, REPO_ROOT
from prob_papillon.llm_judge import build_k_anon_estimator

dotenv.load_dotenv(REPO_ROOT / ".env")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_file", type=str, default=str(DATA_DIR / "PUPA_TNB.csv"))
    parser.add_argument("--k_anon_estimator", choices=["direct", "branch"], default="branch")
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    args = parser.parse_args()

    dspy.configure(lm=dspy.LM("openai/" + args.model))
    estimator = build_k_anon_estimator(args.k_anon_estimator)

    data = pandas.read_csv(args.data_file)
    num_piis, log2_ks = [], []
    for _, row in data.iterrows():
        log2_ks.append(estimator(text=row["user_query"]).log2_k)
        num_piis.append(len(row["pii_units"].split("||")) if isinstance(row["pii_units"], str) else 0)

    rho, p_value = spearmanr(num_piis, log2_ks)
    print(f"Spearman correlation between #PII units and log2(k): {rho:.4f} (p = {p_value:.4e})")
