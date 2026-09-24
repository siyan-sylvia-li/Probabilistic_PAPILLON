"""Mine self-disclosure-containing queries from WildChat and LMSYS-Chat-1M (paper Sec. 3.2).

For the first N conversations of each corpus, takes the first user turn and the
first assistant response, runs the optimized self-disclosure extractor, and keeps
turns with at least one disclosure. Queries that appear in PUPA-TNB are removed.

The output is a *candidate* file in PUPA-SD format. The released PUPA-SD
(data/PUPA_SD.csv) was produced from these candidates by manually removing
non-English and sexual content, which this script does not do.

LMSYS-Chat-1M is gated on Hugging Face; run `huggingface-cli login` first.

Example:
    python scripts/extract_pupa_sd.py --output data/PUPA_SD_candidates.csv
"""
import concurrent.futures
import itertools
from argparse import ArgumentParser

import dotenv
import dspy
import pandas as pd
import tqdm
from datasets import load_dataset

from prob_papillon import DATA_DIR, REPO_ROOT
from prob_papillon.disclosures import deduplicate_disclosures, load_optimized_processor, parse_disclosures

dotenv.load_dotenv(REPO_ROOT / ".env")

SOURCES = {
    "wildchat": "allenai/WildChat-1M",
    "lmsys": "lmsys/lmsys-chat-1m",
}


def first_turn(conversation: list[dict]):
    """Return (user_query, assistant_response) for the first exchange, or None."""
    if len(conversation) < 2 or conversation[0]["role"] != "user" or conversation[1]["role"] != "assistant":
        return None
    return conversation[0]["content"], conversation[1]["content"]


def load_first_turns(source: str, limit: int, english_only: bool) -> list[dict]:
    ds = load_dataset(SOURCES[source], split="train", streaming=True)
    rows = []
    for conv in itertools.islice(ds, limit):
        if english_only and conv.get("language") != "English":
            continue
        turn = first_turn(conv["conversation"])
        if turn is None:
            continue
        query, response = turn
        rows.append({
            "source": source,
            "user_query": query,
            "target_response": response,
            "conversation": f"user: {query}\nassistant: {response}",
        })
    return rows


def format_disclosures(answer: str) -> str:
    """'<answer>nurse</answer><type>occupation</type>...' → 'occupation: nurse; ...' (PUPA-SD pii_units format)."""
    return "; ".join(f"{cat.strip()}: {span.strip()}" for span, cat in parse_disclosures(answer))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "PUPA_SD_candidates.csv"))
    parser.add_argument("--limit", type=int, default=20_000, help="Conversations to read from each corpus.")
    parser.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--exclude", type=str, default=str(DATA_DIR / "PUPA_TNB.csv"),
                        help="PUPA-format CSV whose queries must not appear in the output (evaluation data).")
    parser.add_argument("--keep_non_english", action="store_true",
                        help="Do not pre-filter on the corpus 'language' field.")
    parser.add_argument("--num_threads", type=int, default=16)
    args = parser.parse_args()

    dspy.configure(lm=dspy.LM("openai/" + args.model))
    processor = load_optimized_processor()

    rows = []
    for source in args.sources:
        rows.extend(load_first_turns(source, args.limit, english_only=not args.keep_non_english))
    print(f"Loaded {len(rows)} first turns from {args.sources}")

    excluded = set(pd.read_csv(args.exclude)["user_query"].str.strip())
    rows = [r for r in rows if r["user_query"].strip() not in excluded]

    def extract(row):
        try:
            disclosures = deduplicate_disclosures(processor.extractor(text=row["user_query"]).answer)
        except Exception as e:
            print(f"[extract] {e}")
            return ""
        return format_disclosures(disclosures)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        disclosures = list(tqdm.tqdm(executor.map(extract, rows), total=len(rows)))

    candidates = pd.DataFrame([{**r, "pii_units": d} for r, d in zip(rows, disclosures) if d])
    candidates = candidates.drop_duplicates(subset="user_query").reset_index(drop=True)
    candidates.insert(0, "index", candidates.index)
    candidates = candidates[["index", "source", "pii_units", "conversation", "user_query", "target_response"]]
    candidates.to_csv(args.output, index=False)
    print(f"Saved {len(candidates)} candidates to {args.output}. "
          "Manually remove non-English and sexual content before use.")
