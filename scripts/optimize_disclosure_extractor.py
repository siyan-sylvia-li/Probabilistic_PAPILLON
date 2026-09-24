"""Evaluate and optimize the self-disclosure extractor (paper Sec. 3.1, Table 1).

1. Scores the seed extractor prompt with the chosen model on the annotated texts
   from Zheng et al. (2025), using an LLM judge to count correct disclosures.
2. Unless --baseline_only is set, proposes 10 alternative extractor instructions
   with MIPROv2 (GPT-5 as the proposer), scores each, and saves the best one.

Examples:
    python scripts/optimize_disclosure_extractor.py --model_name gpt-4.1-mini --baseline_only
    python scripts/optimize_disclosure_extractor.py --model_name gpt-5-mini
"""
import json
import threading
from argparse import ArgumentParser

import dotenv
import dspy

from prob_papillon import ARTIFACTS_DIR, DATA_DIR, REPO_ROOT
from prob_papillon.disclosures import DisclosureExtractorSignature, DisclosureProcessor, parse_disclosures

dotenv.load_dotenv(REPO_ROOT / ".env")


class DisclosureJudgeMatchNoCategory(dspy.Signature):
    """Compare predicted self-disclosures against a ground-truth list and count true positives.

    A predicted disclosure is a true positive if it is semantically equivalent to a ground-truth
    disclosure — exact wording is not required, but the meaning must match.
    Paraphrases and minor lexical differences are acceptable. Use the original text to resolve ambiguities about what the author actually disclosed. Count each ground-truth disclosure at most once."""
    original_text: str = dspy.InputField(description="The original post text that the disclosures were extracted from. Use this to resolve ambiguous cases.")
    ground_truth: str = dspy.InputField(description="The ground truth list of self-disclosures.")
    predicted: str = dspy.InputField(description="The predicted list of self-disclosures to evaluate.")
    correct_count: int = dspy.OutputField(description="The number of predicted disclosures that match or are semantically equivalent to a disclosure in the ground truth list.")


class ExtractionMetric:
    """Disclosure-level F1 against gold spans; also records precision and recall per example."""

    def __init__(self, judge_lm: dspy.LM):
        self.judge = dspy.Predict(DisclosureJudgeMatchNoCategory)
        self.judge.set_lm(judge_lm)
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self.f1s, self.precisions, self.recalls = [], [], []

    def summary(self) -> dict:
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        return {"f1": mean(self.f1s), "precision": mean(self.precisions), "recall": mean(self.recalls)}

    def __call__(self, gold, pred, trace=None):
        gold_disclosures = list({x[0] for x in gold.disclosures})

        if not hasattr(pred, "disclosures"):
            return 0.0

        predicted_disclosures = list({x[0] for x in parse_disclosures(pred.disclosures)})

        # If ground truth is empty, score 1.0 only when prediction is also empty
        if not gold_disclosures:
            return 1.0 if not predicted_disclosures else 0.0

        match_result = self.judge(original_text=gold.text, ground_truth=str(gold_disclosures), predicted=str(predicted_disclosures))
        try:
            correct = int(match_result.correct_count)
        except (ValueError, AttributeError):
            correct = 0

        precision = correct / (len(predicted_disclosures) or 1)
        recall = correct / len(gold_disclosures)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        with self._lock:
            self.f1s.append(f1)
            self.precisions.append(precision)
            self.recalls.append(recall)

        if trace is not None:
            return f1 >= 0.7
        return f1


def score(processor, evaluator, metric) -> dict:
    metric.reset()
    evaluator(processor)
    return metric.summary()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-5-mini", help="OpenAI model used as the extractor.")
    parser.add_argument("--judge_model", type=str, default="gpt-5-mini")
    parser.add_argument("--proposer_model", type=str, default="gpt-5")
    parser.add_argument("--data_file", type=str, default=str(DATA_DIR / "disclosure_extraction_eval.json"))
    parser.add_argument("--output_dir", type=str, default=str(ARTIFACTS_DIR))
    parser.add_argument("--baseline_only", action="store_true")
    args = parser.parse_args()

    if args.model_name == "gpt-5-mini":
        lm = dspy.LM("openai/gpt-5-mini")
    else:
        lm = dspy.LM("openai/" + args.model_name, temperature=0)
    dspy.configure(lm=lm)

    raw = json.load(open(args.data_file))
    dataset = [dspy.Example(x).with_inputs("text") for x in raw]

    metric = ExtractionMetric(dspy.LM("openai/" + args.judge_model))
    evaluator = dspy.Evaluate(devset=dataset, metric=metric, num_threads=8)

    baseline = score(DisclosureProcessor(), evaluator, metric)
    print("Baseline:", baseline)
    baseline_path = f"{args.output_dir}/extractor_baseline_{args.model_name}.json"
    json.dump({"model_name": args.model_name, "baseline": baseline}, open(baseline_path, "w"), indent=2)
    print(f"Baseline results saved to {baseline_path}")

    if args.baseline_only:
        raise SystemExit(0)

    candidates_path = f"{args.output_dir}/extractor_candidates_{args.model_name}.json"
    try:
        candidates = json.load(open(candidates_path))
        print(f"Loaded proposed candidates from {candidates_path}")
    except FileNotFoundError:
        teleprompter = dspy.MIPROv2(
            metric=metric,
            auto="medium",
            num_threads=8,
            prompt_model=dspy.LM("openai/" + args.proposer_model),
        )
        # Instruction proposal only (no full MIPRO search); candidates are scored below.
        candidates = teleprompter._propose_instructions(
            DisclosureProcessor(),
            trainset=dataset[:10],
            demo_candidates=None,
            view_data_batch_size=10,
            program_aware_proposer=True,
            data_aware_proposer=True,
            tip_aware_proposer=True,
            fewshot_aware_proposer=False,
            num_instruct_candidates=10,
        )
        json.dump(candidates, open(candidates_path, "w"), indent=2)
        print(f"Proposed candidates saved to {candidates_path}")

    # Key "0" holds the candidates for the extractor predictor.
    results = []
    for instruction in candidates["0"]:
        DisclosureExtractorSignature.instructions = instruction
        results.append({"extractor_instruction": instruction, **score(DisclosureProcessor(), evaluator, metric)})
        print({k: v for k, v in results[-1].items() if k != "extractor_instruction"})

    json.dump(results, open(f"{args.output_dir}/extractor_candidate_scores_{args.model_name}.json", "w"), indent=2)

    best = max(results, key=lambda r: r["f1"])
    print(f"\nBest candidate: f1={best['f1']:.4f} precision={best['precision']:.4f} recall={best['recall']:.4f}")

    DisclosureExtractorSignature.instructions = best["extractor_instruction"]
    processor_path = f"{args.output_dir}/disclosure_processor_{args.model_name}.json"
    DisclosureProcessor().save(processor_path)
    print(f"Optimized processor saved to {processor_path}")
