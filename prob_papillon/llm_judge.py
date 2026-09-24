"""LLM judges for response quality, PII leakage, prompt well-formedness and k-anonymity."""
import dspy

from prob_papillon.disclosures import AnyDisclosureProcessor
from prob_papillon.k_anonymity import BASE_LOG2, BranchKAnonymityEstimator, DirectKAnonymityEstimator

K_ANON_ESTIMATORS = ("direct", "branch")


class JudgeQuality(dspy.Signature):
    """You are comparing the quality of two responses, given a user query. Is response A at least as good as response B? Respond with yes or no."""

    user_query = dspy.InputField(desc="The user's request to be fulfilled.")
    response_A = dspy.InputField()
    response_B = dspy.InputField()
    output = dspy.OutputField()


class JudgeLeakage(dspy.Signature):
    """Determine whether a piece of information is present in the prompt given. Respond with yes or no."""

    information_piece = dspy.InputField(desc="The piece of information that we wish to check. If this piece of information is a celebrity name, respond with no.")
    prompt = dspy.InputField()
    output = dspy.OutputField()


class JudgePromptQual(dspy.Signature):
    """Determine whether a piece of text is a valid prompt for a language model written from a first-person perspective. Respond with yes or no."""
    prompt_input = dspy.InputField()
    output = dspy.OutputField()


def build_k_anon_estimator(name: str):
    processor = AnyDisclosureProcessor()
    if name == "direct":
        return DirectKAnonymityEstimator(processor=processor)
    if name == "branch":
        return BranchKAnonymityEstimator(processor=processor)
    raise ValueError(f"Unknown k-anonymity estimator {name!r}; choose from {K_ANON_ESTIMATORS}")


def _yes(answer) -> int:
    return int(answer.output.lower().startswith("yes"))


class LLMJudge(dspy.Module):
    """Returns ``quality`` (0/1), ``k_anon`` (log2 k / log2 400M, in [0, 1]),
    ``pii_leakage`` (fraction of PII units present in the rewritten prompt) and
    ``prompt`` (0/1 well-formedness). A score is -1 when its input is missing;
    ``pii_leakage`` is None when ``pii_str`` is not a "||"-separated PII list."""

    def __init__(self, k_anon_estimator: str = "direct"):
        super().__init__()
        self.quality_judge = dspy.ChainOfThought(JudgeQuality)
        self.k_anon_estimator = build_k_anon_estimator(k_anon_estimator)
        self.leakage_checker = dspy.ChainOfThought(JudgeLeakage)
        self.prompt_qual = dspy.ChainOfThought(JudgePromptQual)

    def forward(self, user_query, og_resp, new_resp=None, updated_query=None, pii_str=None):
        if new_resp:
            # Judge in both orders to reduce position bias; ties count as a win.
            judgment_1 = _yes(self.quality_judge(user_query=user_query, response_A=new_resp, response_B=og_resp))
            judgment_2 = _yes(self.quality_judge(user_query=user_query, response_A=og_resp, response_B=new_resp))
            quality = 1 if judgment_1 == judgment_2 else judgment_1
        else:
            quality = -1

        if updated_query:
            k_anon = self.k_anon_estimator(text=updated_query).log2_k / BASE_LOG2
            if pii_str and "||" in pii_str:
                all_pii_pieces = {x for x in pii_str.split("||") if len(x)}
                pii_score = sum(
                    _yes(self.leakage_checker(information_piece=p, prompt=updated_query)) for p in all_pii_pieces
                ) / len(all_pii_pieces)
            else:
                pii_score = None
            prompt_score = _yes(self.prompt_qual(prompt_input=updated_query))
        else:
            pii_score = -1
            prompt_score = -1
            k_anon = -1

        return dspy.Prediction(quality=quality, k_anon=k_anon, pii_leakage=pii_score, prompt=prompt_score)
