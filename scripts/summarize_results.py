"""Print a Table-2-style summary (0-100 scale) from evaluate_papillon.py outputs.

Expects files named <model>_before.json / <model>_after.json.

Usage:
    python scripts/summarize_results.py results/
"""
import json
import sys
from pathlib import Path


def summarize(path: Path) -> dict:
    rows = json.load(open(path))
    pii = [r["pii_leakage"] for r in rows if r["pii_leakage"] is not None]
    return {
        "n": len(rows),
        "quality": 100 * sum(r["quality"] for r in rows) / len(rows),
        "pii_leakage": 100 * sum(pii) / len(pii),
        "k_anon": 100 * sum(r["k_anon"] for r in rows) / len(rows),
    }


if __name__ == "__main__":
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    models = sorted({p.name.rsplit("_", 1)[0] for p in results_dir.glob("*_before.json")}
                    | {p.name.rsplit("_", 1)[0] for p in results_dir.glob("*_after.json")})
    header = f"{'model':40s} | {'stage':6s} | {'n':>4s} | {'QUAL↑':>6s} | {'LEAK↓':>6s} | {'k-ANON↑':>7s}"
    print(header)
    print("-" * len(header))
    for model in models:
        for stage in ("before", "after"):
            path = results_dir / f"{model}_{stage}.json"
            if not path.exists():
                continue
            s = summarize(path)
            print(f"{model:40s} | {stage:6s} | {s['n']:4d} | {s['quality']:6.2f} | {s['pii_leakage']:6.2f} | {s['k_anon']:7.2f}")
