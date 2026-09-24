"""LLM-based probabilistic k-anonymity estimation (paper Sec. 4.2).

Two estimators are provided:

- ``BranchKAnonymityEstimator``: the BRANCH-style pipeline described in the paper.
  Disclosures are extracted, arranged into cumulative conditioning groups, turned
  into population / conditional-percentage queries, answered by an LLM, and
  multiplied together. Answers are cached and reused for any new query whose
  embedding has cosine similarity >= 0.95 with a cached one.
- ``DirectKAnonymityEstimator``: a cheaper proxy that extracts disclosures and asks
  an LLM for log2(k) in a single call.

Both return a ``dspy.Prediction`` with ``k``, ``log2_k`` and ``risk_level``.
"""
import concurrent.futures
import json
import math
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import dspy
import numpy as np

from prob_papillon import ARTIFACTS_DIR

BASE_POPULATION = 400_000_000   # ~400 M English speakers; fallback when no population is estimable
BASE_LOG2 = math.log2(BASE_POPULATION)
CACHE_SIMILARITY_THRESHOLD = 0.95
DEFAULT_QUERY_CACHE = ARTIFACTS_DIR / "k_anon_query_cache.json"


def risk_level(k: int) -> str:
    return (
        "VERY HIGH" if k < 10     else
        "HIGH"      if k < 100    else
        "MEDIUM"    if k < 10_000 else
        "LOW"
    )


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class DisclosureStep:
    span: str
    category: str
    # Conditioning context: items that must be estimated before this one
    parents: list[tuple[str, str]] = field(default_factory=list)
    is_population: bool = False   # True → integer count query; False → % query
    query: str = ""
    search_result: str = ""
    result_number: float = None


# ── Parsing ────────────────────────────────────────────────────────────────────

def parse_ordering(ordering: str) -> list[list[tuple[str, str]]]:
    """Parse <group> tags into a list of cumulative groups.

    Each group is a list of (span, category) tuples. Groups are cumulative:
    each one repeats all items from the previous group and appends new ones.
    """
    groups: list[list[tuple[str, str]]] = []
    for group_content in re.findall(r"<group>(.*?)</group>", ordering, re.DOTALL):
        items = re.findall(
            r"<answer>(.*?)</answer>\s*<type>(.*?)</type>", group_content, re.DOTALL
        )
        groups.append([(a.strip(), t.strip()) for a, t in items])
    return groups


def get_chain_steps(groups: list[list[tuple[str, str]]]) -> list[DisclosureStep]:
    """Convert cumulative groups into a flat ordered list of DisclosureSteps.

    - Within the anchor group (Group 1): process items sequentially; each item's
      parents = all items placed before it within the same group. The very first
      item overall gets a population query (is_population=True).
    - For each subsequent group k: new items (those absent from group k-1) get
      percentage queries conditioned on the full set of items in group k-1.
    """
    if not groups:
        return []

    steps: list[DisclosureStep] = []
    placed: list[tuple[str, str]] = []   # ordered running list of all placed items

    for group_idx, group in enumerate(groups):
        if group_idx == 0:
            for span, category in group:
                is_root = len(steps) == 0
                steps.append(DisclosureStep(
                    span=span,
                    category=category,
                    parents=list(placed),
                    is_population=is_root,
                ))
                placed.append((span, category))
        else:
            prev_set = set(groups[group_idx - 1])
            new_items = [(s, c) for s, c in group if (s, c) not in prev_set]
            parents_for_new = list(groups[group_idx - 1])
            for span, category in new_items:
                steps.append(DisclosureStep(
                    span=span,
                    category=category,
                    parents=parents_for_new,
                    is_population=False,
                ))
                placed.append((span, category))

    return steps


# ── Query generation ───────────────────────────────────────────────────────────

class PopulationQuerySignature(dspy.Signature):
    """Generate a concise search query to find the TOTAL POPULATION matching a
    personal disclosure. The query must return an INTEGER count of people, not
    a percentage. For a location, ask for its population directly."""
    disclosure: str = dspy.InputField(
        description="The disclosure as 'span (category)'."
    )
    query: str = dspy.OutputField(
        description="A short, specific search query that returns a population count."
    )


class ConditionalQuerySignature(dspy.Signature):
    """Generate a concise search query that yields the PERCENTAGE of the parent
    population who also have the current personal disclosure.

    Possible query structure: '[parent population] that [current disclosure]'

    Ensure that the query is estimable with census or survey data, which can mean that you will not include all of the parent population descriptions. DO NOT BE OVERLY SPECIFIC."""
    parents: str = dspy.InputField(
        description="Comma-separated parent disclosures defining the conditioning population."
    )
    disclosure: str = dspy.InputField(
        description="The disclosure to estimate, as 'span (category)'."
    )
    query: str = dspy.OutputField(
        description="A short, specific search query that returns a conditional percentage."
    )


class QueryGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.pop_query  = dspy.Predict(PopulationQuerySignature)
        self.cond_query = dspy.Predict(ConditionalQuerySignature)

    def forward(self, step: DisclosureStep) -> str:
        disc_str = f"{step.span} ({step.category})"
        if step.is_population:
            return self.pop_query(disclosure=disc_str).query
        parents_str = (
            ", ".join(f"{s} ({c})" for s, c in step.parents)
            if step.parents else "general population"
        )
        return self.cond_query(parents=parents_str, disclosure=disc_str).query


# ── Answering queries ──────────────────────────────────────────────────────────

class ResultParserSignature(dspy.Signature):
    """Provide a single number as the answer given a search query and relevant search results."""
    search_query: str = dspy.InputField()
    search_results: str = dspy.InputField()
    output_number: float = dspy.OutputField(desc="The total population or the percentage queried in the search query. If the search query is about a percentage, return a float between 0 and 1.")


class ResultParser(dspy.Module):
    """Answers a statistics query from the LLM's parametric knowledge, then parses a number."""

    def __init__(self, answer_lm: str = "openai/gpt-5-mini", callbacks=None):
        super().__init__(callbacks)
        self.answer_lm = dspy.LM(model=answer_lm)
        self.results_parser = dspy.Predict(ResultParserSignature)

    def format_search_query(self, query: str) -> str:
        return f"I am looking for the following information: {query}. Use your knowledge of census data, survey data, and general world knowledge to find a concise answer. If the search results do not contain relevant information, provide your best estimate based on your knowledge and reasoning abilities."

    def create_search(self, search_query: str) -> str:
        msg = [{"role": "user", "content": self.format_search_query(search_query)}]
        return self.answer_lm(messages=msg)[0]

    def forward(self, search_query: str, search_results: str):
        return self.results_parser(search_query=search_query, search_results=search_results).output_number


# ── k-value computation ────────────────────────────────────────────────────────

def compute_k(steps: list[DisclosureStep]) -> tuple[int, str]:
    """Compute the BRANCH k-value via chain-rule multiplication.

    k = N × p1 × p2 × … × pj

    N is the starting population (from the first is_population step, or
    BASE_POPULATION if none exists). Each p_i is a conditional probability;
    values > 1 are treated as percentages in the 0-100 range and divided by 100.

    Returns (k, human-readable chain-rule equation).
    """
    if not steps:
        return BASE_POPULATION, str(BASE_POPULATION)

    k = 1.0
    parts: list[str] = []

    for step in steps:
        val = step.result_number
        if val is None:
            continue
        if step.is_population:
            val = max(val, 1.0)
            if val == 1.0:
                val = BASE_POPULATION
            parts.append(f"{val:,.0f} [{step.span[:35]}]")
        else:
            if val > 1.0:       # treat as 0-100 percentage, normalise to fraction
                val /= 100.0
            val = max(val, 1e-9)
            parts.append(f"{val:.4f} [{step.span[:35]}]")
        k *= val

    if not any(s.is_population and s.result_number is not None for s in steps):
        k *= BASE_POPULATION
        parts.insert(0, f"{BASE_POPULATION:,} [base population]")

    k = max(1, round(k))
    return k, " × ".join(parts) if parts else "N/A"


# ── BRANCH-style estimator (paper Sec. 4.2) ────────────────────────────────────

def _cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class BranchKAnonymityEstimator:
    """Disclosure extraction → ordering → query generation → LLM answers → k.

    ``processor`` must return a Prediction with ``disclosures`` and ``ordering``
    (see ``prob_papillon.disclosures``). The query cache maps each query string to
    ``[answer_text, number]`` and is written back to ``cache_path`` after each call.
    """

    def __init__(self, processor: dspy.Module, cache_path: Path | str | None = DEFAULT_QUERY_CACHE, verbose: bool = False):
        self.processor = processor
        self.query_gen = QueryGenerator()
        self.result_parser = ResultParser()
        self.cache_path = Path(cache_path) if cache_path else None
        self.verbose = verbose
        self._lock = threading.Lock()

        self.corpus = {}
        if self.cache_path and self.cache_path.exists():
            self.corpus = json.load(open(self.cache_path))
        self.retrieval_corpus = list(self.corpus.keys())
        self.embedder = dspy.Embedder("openai/text-embedding-3-small", dimensions=512)
        self.search = self._build_retriever()

    def _build_retriever(self):
        if not self.retrieval_corpus:
            return None
        return dspy.retrievers.Embeddings(embedder=self.embedder, corpus=list(self.retrieval_corpus), k=1)

    def _lookup(self, query: str):
        """Return a cached [answer_text, number] for ``query`` or a near-duplicate, else None."""
        if query in self.corpus:
            return self.corpus[query]
        if self.search:
            top_1 = self.search(query).passages[0]
            if _cosine(self.embedder(query), self.embedder(top_1)) > CACHE_SIMILARITY_THRESHOLD:
                return self.corpus[top_1]
        return None

    def __call__(self, text: str) -> dspy.Prediction:
        pred = self.processor(text=text)
        groups = parse_ordering(pred.ordering) if pred.ordering else []
        steps = get_chain_steps(groups)

        for i, step in enumerate(steps):
            step.query = self.query_gen(step)
            if i > 0:
                step.query = step.query + "\n\nMAKE SURE THAT YOU RETURN A PERCENTAGE VALUE."

        uncached = []
        with self._lock:
            for step in steps:
                hit = self._lookup(step.query)
                if hit is not None:
                    step.search_result, step.result_number = hit
                else:
                    uncached.append(step)

        def _fetch_and_parse(step):
            step.search_result = self.result_parser.create_search(step.query)
            try:
                step.result_number = self.result_parser(search_query=step.query, search_results=step.search_result)
                if not isinstance(step.result_number, (int, float)) or math.isnan(step.result_number) or step.result_number == 0:
                    step.result_number = None
            except Exception:
                step.result_number = None

        if uncached:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                list(executor.map(_fetch_and_parse, uncached))

            with self._lock:
                for step in uncached:
                    if step.result_number is not None:
                        self.retrieval_corpus.append(step.query)
                        self.corpus[step.query] = [step.search_result, step.result_number]
                self.search = self._build_retriever()
                if self.cache_path:
                    json.dump(self.corpus, open(self.cache_path, "w"))

        k, equation = compute_k(steps)
        log2_k = math.log2(k) if k > 0 else 0.0
        if self.verbose:
            print(f"k = {k:,}  log2(k) = {log2_k:.3f}  risk = {risk_level(k)}\n  {equation}")
        return dspy.Prediction(k=k, log2_k=log2_k, risk_level=risk_level(k), equation=equation)


# ── Direct k-anonymity estimator (single-call proxy) ───────────────────────────

class DirectKAnonymitySignature(dspy.Signature):
    """Estimate the k-anonymity of a person described by the given list of personal
    disclosures. k is the number of people in the general population who share all
    of these attributes simultaneously.

    Use your knowledge of census data, government surveys, and official statistics
    to reason about how rare this combination of attributes is. Apply the chain rule
    of probability: start from the base population (~400 million English speakers),
    then multiply by the conditional probability of each attribute given the others.
    Return log2(k) — a higher value means more people share the profile (lower risk);
    a lower value means fewer people share it (higher risk)."""
    disclosures: str = dspy.InputField(
        description="Comma-separated list of personal disclosures as 'span (category)' pairs."
    )
    log2_k: float = dspy.OutputField(
        description="log2 of the estimated k-anonymity value. E.g. 28.5 for ~400M people, 10 for ~1000 people."
    )


class DirectKAnonymityEstimator(dspy.Module):
    """Extracts disclosures, then estimates log2(k) with a single LLM call."""

    def __init__(self, processor: dspy.Module):
        super().__init__()
        self.processor = processor
        self.k_estimator = dspy.ChainOfThought(DirectKAnonymitySignature)

    def forward(self, text: str) -> dspy.Prediction:
        pred = self.processor(text=text)

        if not pred.disclosures or pred.disclosures.strip() == "<list></list>":
            return dspy.Prediction(k=BASE_POPULATION, log2_k=round(BASE_LOG2, 3), risk_level="LOW")

        disclosures_str = ", ".join(
            f"{span} ({cat})"
            for span, cat in re.findall(r"<answer>(.*?)</answer>\s*<type>(.*?)</type>", pred.disclosures, re.DOTALL)
        ) or pred.disclosures

        try:
            result = self.k_estimator(disclosures=disclosures_str)
            log2_k = float(result.log2_k)
            k = max(1, round(2 ** log2_k))
        except Exception:
            log2_k = BASE_LOG2
            k = BASE_POPULATION

        return dspy.Prediction(k=k, log2_k=round(log2_k, 3), risk_level=risk_level(k))
