"""Count explicit PII units per PUPA-SD query with GLiNER-PII (paper Sec. 3.2, Figure 2).

Usage:
    python scripts/count_pii_gliner.py
"""
from argparse import ArgumentParser

import pandas
from gliner import GLiNER

from prob_papillon import DATA_DIR

LABELS = ["email", "phone_number", "user_name", "first_name", "last_name", "company_name", "url", "country", "city", "county"]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_file", type=str, default=str(DATA_DIR / "PUPA_SD.csv"))
    args = parser.parse_args()

    model = GLiNER.from_pretrained("nvidia/gliner-pii")

    def num_piis(text: str) -> int:
        entities = model.predict_entities(text, LABELS, threshold=0.85)
        return sum(1 for e in entities if e["score"] > 0.9)

    counts = pandas.read_csv(args.data_file)["user_query"].map(num_piis)
    bins = counts.map(lambda n: "No PIIs" if n == 0 else f"N = {n}" if n <= 2 else "N > 2")
    print(bins.value_counts(normalize=True).mul(100).round(1).to_string())
