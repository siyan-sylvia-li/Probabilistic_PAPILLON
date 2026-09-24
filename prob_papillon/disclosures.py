"""Self-disclosure extraction and BRANCH-style disclosure ordering (paper Sec. 3.1, 4.2, App. A)."""
import re

import dspy

from prob_papillon import ARTIFACTS_DIR

# Self-disclosure extractor + ordering prompts selected via MIPROv2 instruction proposal (Sec. 3.1).
OPTIMIZED_PROCESSOR_PATH = ARTIFACTS_DIR / "disclosure_processor_gpt-5-mini.json"


# ── Signatures (prompts reproduced verbatim from the paper, Appendix A) ────────

class AnyDisclosureExtractorSignature(dspy.Signature):
    """From the text below, extract only those personal disclosures about ANY person that can be estimated using official, population-level statistics (e.g., census, national surveys, government records). Return a compact, canonical list of disclosures with one best-fit category per item.

Key definitions
- Personal Disclosure: Information about ANY person revealed in the text — including the author/poster, a described subject the request is written on behalf of (e.g., "a nurse who lives in…", "a third-year student"), or any named or role-described individual. Includes static traits (e.g., age, gender, location) and time-bound events (e.g., health diagnosis, education credential, employment change). The person does NOT need to be the one writing the text.
- Estimable: Attributes/events that are commonly measured in official data sources (census, government surveys/registries, administrative records). If it’s not plausibly supported by reputable population statistics, exclude it.

Allowed categories (pick the single best fit for each item)
- location, age, relationship_status, gender, pet, appearance, race/nationality,
  sexual_orientation, health, family, occupation, mental_health, emotions,
  reproductive_health, finance, education, crime, events, PII, other

Inclusions (examples are illustrative, not exhaustive)
- Demographics: age ("25 year old"), gender ("female", "trans woman"), race/nationality ("Black", "Mexican American").
- Geography: residence or stable presence ("New York City", "California", "UK"). Prefer official place names; expand common unambiguous abbreviations (e.g., "NYC" → "New York City").
- Household/family: "married", "single", "divorced", "have 2 children", "live alone".
- Education: "high school graduate", "bachelor’s degree", "college student", "in graduate school".
- Occupation and employment status: job titles ("software engineer"), statuses ("unemployed", "retired"), industry ("construction worker"). Freelance/gig counts as occupation; layoffs count only if stated as an event ("laid off").
- Health (physical): diagnoses, chronic conditions, disability status, medication use, BMI/height/weight if stated, substance use if specific.
- Mental health: diagnosed or clinically recognizable conditions ("diagnosed with depression", "ADHD"). Suicidal ideation/attempts belong here if explicitly stated.
- Reproductive health: pregnancy, contraception use/duration, fertility treatment, miscarriages, abortions.
- Finance: income bracket, rent amount, debt/loans, benefits/assistance participation.
- Pets: "pet dog", "2 cats".
- Crime/justice: "arrested for DUI", "felony conviction".
- Events (only if plausibly measured in official stats and not better placed above): "moved states this year", "evicted", "naturalized this year".
- PII (only if commonly available in official aggregate data): first name, ZIP code, year of birth. Exclude unique identifiers (full address, phone numbers, SSN, email).

Exclusions
- Purely incidental third parties with no representational role (e.g., celebrities or historical figures mentioned in passing, a researcher being contacted).
- Vague emotions/opinions ("I feel sad", "I’m overwhelmed") unless clearly tied to a diagnosable/estimable construct. If present without diagnosis, omit.
- Non-English disclosures: if the disclosure is not in English, omit it.
- Duplicates or near-duplicates ("woman" vs "female"; keep one).
- Unverifiable speculation ("might be pregnant", "maybe have ADHD") — instead capture concrete, stated facts (e.g., "missed period" as health).

Normalization rules
- Keep spans short and canonical: "I’m 25" → "25 year old"; "live in NYC" → "New York City".
- Parse compact tokens: "18F" → two items: "18" (age), "female" (gender).
- Prefer specific standard geographies (city + state/country) when given; do not invent detail.
- Map "student" to education; job titles and employment status to occupation; income/benefits to finance; relationship labels ("engaged", "married", "single", "widowed") to relationship_status.
- If multiple categories could apply, choose the best single fit. Do not duplicate across categories.
- Include durations when they are commonly measured (e.g., "on birth control for 3 years" under reproductive_health or health depending on context).

Edge cases
- If no eligible disclosures exist, return an empty list: <list></list>
- If both current and past facts are present, include both if clearly stated and estimable (e.g., "graduated last year" + "currently unemployed").
- When a subject is described in the third person but the request is clearly written on their behalf (e.g., "write a bio for a 35-year-old nurse in Boston"), extract their disclosures as you would for a first-person statement.

Output format (strict)
- Return ONLY the disclosures in this XML-like structure:
  <list>
    <answer>disclosure phrase</answer><type>category</type>
    ...
  </list>
- Do not include any other text. One disclosure per <answer>…</answer><type>…</type> pair.

Mini-examples
- Text: "I’m a 30M living in LA, recently laid off, now looking for work."
  Answer: <list><answer>30</answer><type>age</type><answer>male</answer><type>gender</type><answer>Los Angeles</answer><type>location</type><answer>laid off</answer><type>events</type><answer>unemployed</answer><type>occupation</type></list>
- Text: "Write a professional email from a third-year clinical psychology student focusing on neuropsychology."
  Answer: <list><answer>third-year clinical psychology student</answer><type>education</type><answer>focusing on neuropsychology</answer><type>education</type></list>
- Text: "I want to understand the cost of living in Northampton for a nurse who lives in a small studio apartment."
  Answer: <list><answer>nurse</answer><type>occupation</type><answer>Northampton</answer><type>location</type></list>

Now analyze the text and return the list. Only include disclosures that are estimable, normalized, non-duplicative, English, and mapped to the best-fit category."""
    text: str = dspy.InputField(description="The text to analyze for personal disclosures.")
    answer: str = dspy.OutputField(description="The list of personal disclosures in the text, formatted as specified above. Each disclosure should be short and succinct.")


class DeduplicateSignature(dspy.Signature):
    """**Task**: Remove duplicate disclosures from the list below. Disclosures are considered duplicates if they convey the same information, even if the wording is different."""
    disclosures: str = dspy.InputField(description="The list of personal disclosures to deduplicate, formatted as specified above.")
    answer: str = dspy.OutputField(description="The deduplicated list of personal disclosures, formatted as specified above.")

class DisclosureExtractorSignature(dspy.Signature):
    """**Task**: Determine which personal disclosures in the text below can be estimated \
using publicly available statistics and data.

**Personal Disclosure**: The act of intentionally sharing personal information about \
oneself. THIS IS ONLY ABOUT THE USER, NOT HYPOTHETICAL THIRD PERSONS. Includes static attributes (name, age, gender, location) and events \
(health, education, employment). Only information about the POSTER counts — \
exclude information about other people.

**Estimable**: Only include disclosures that can be looked up via official surveys, \
census databases, or government records.  Exclude vague emotions and opinions \
unless they correspond to a diagnosable/estimable condition.

**Disclosure categories (pick the best fit)**:
  location, age, relationship_status, gender, pet, appearance, race/nationality,
  sexual_orientation, health, family, occupation, mental_health, emotions,
  reproductive_health, finance, education, crime, events, PII, other

**Instructions**:
1. Review the text below carefully.
2. List each estimable disclosure as a short phrase (exact words or close paraphrase).
3. Return results in <list></list>.  Each disclosure: \
<answer>disclosure phrase</answer><type>category</type>
4. Do NOT duplicate spans conveying the same information.
5. Make sure the disclosure spans are about the USER, not other people. The disclosure spans should be with respect to the first person.
6. ONLY RETURN ENGLISH DISCLOSURES. If the text is in another language, do not return any disclosures.

EXAMPLES:

**Text**: Please focus on preparing for the college entrance examination again after failing the exam, reflecting the hardships of the preparation process and one\'s own perseverance. According to the following requirements, write an essay with a word count of 450 or more：\nWe learned about NAME_1 and NAME_2, and also explored how to face setbacks in life. The famous writer NAME_3 once wrote in his works A Farewell to Arms "The world breaks everyone and afterward many are strong at the broken places." \nWhat is the biggest setback or challenge you have encountered in life so far? How did you solve it? What inspiration or impact did this experience have on you?
**Answer**: <list></list>

**Text**: I am a 25 year old woman who lives in New York City. I work as a software engineer and have a pet dog. I am looking for a dog daycare center in the city. How should I get started?
**Answer**: <list><answer>25 year old</answer><type>age</type><answer>New York City</answer><type>location</type><answer>software engineer</answer><type>occupation</type><answer>woman</answer><type>gender</type><answer>pet dog</answer><type>pet</type></list>

**Text**: TITLE: Just missed my period
I (18F) have been on the birth control pill for 3 years and ive never been late. But this month i'm a week late and i'm freaking out. I take the pill every day, no misses, and have been tracking my period on an app. I don't know if i'm pregnant or if the pill isn't working. I also have mood swings and my breast feels tender, but idk if that's PMS or something else. What should i do?
**Answer**: <list><answer>18</answer><type>age</type><answer>female</answer><type>gender</type><answer>have been on the birth control pill for 3 years</answer><type>health</type><answer>missed period</answer><type>health</type><answer>mood swings</answer><type>health</type><answer>tender breasts</answer><type>health</type></list>
"""
    text: str = dspy.InputField(description="The text to analyze for personal disclosures.")
    answer: str = dspy.OutputField(description="The list of personal disclosures in the text, formatted as specified above.")


class DisclosureOrderingSignature(dspy.Signature):
    """**Task**: Arrange disclosures into cumulative conditioning groups so that the joint \
probability P(d1, d2, ..., dn) can be factored as a product of conditional probabilities \
(chain rule / BRANCH framework).

**Group construction algorithm**:
1. Identify INDEPENDENT disclosures — those with no statistical dependencies \
(typically location, age, gender). These seed Group 1.
2. Identify FIRST-WAVE disclosures — those that depend ONLY on the independent disclosures \
from step 1 (e.g. age given location+gender). Add them to Group 1 as well.
   → Group 1 = independent disclosures + first-wave disclosures.
3. For each subsequent group: find all remaining disclosures whose EVERY dependency is \
already present in the previous group. Add them to form the next group.
   → Group k = ALL items from Group k-1 PLUS the newly resolved disclosures.
4. Repeat until all disclosures are placed.

**Critical format rule**: every group is CUMULATIVE — it repeats every disclosure from \
all prior groups and then appends the new ones. Do NOT create a new group for items that \
can be resolved within the current group's context.

**Example**:
Disclosures: New York City (location), woman (gender), 25 year old (age), software engineer (occupation)
- Independent: New York City, woman → seed Group 1
- First-wave: 25 year old (depends only on location + gender) → also in Group 1
- Group 1 = {New York City, woman, 25 year old}
- Resolved by Group 1: software engineer (depends on location, gender, age — all in Group 1)
- Group 2 = Group 1 + {software engineer} = {New York City, woman, 25 year old, software engineer}

Output:
<list>\
<group><answer>New York City</answer><type>location</type><answer>woman</answer><type>gender</type><answer>25 year old</answer><type>age</type></group>\
<group><answer>New York City</answer><type>location</type><answer>woman</answer><type>gender</type><answer>25 year old</answer><type>age</type><answer>software engineer</answer><type>occupation</type></group>\
</list>

**Instructions**:
Return the result as <list></list> of <group> elements. \
Each group contains ALL disclosures accumulated up to that step: \
<group><answer>span1</answer><type>cat1</type><answer>span2</answer><type>cat2</type>...</group>
Do NOT omit any disclosure from any group. Do NOT create single-item groups unless only one \
disclosure exists at that step."""
    disclosures: str = dspy.InputField(description="The list of personal disclosures to reorder.")
    answer: str = dspy.OutputField(description="The reordered list of personal disclosures, formatted as specified above.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_disclosures(answer: str) -> list[tuple[str, str]]:
    """Extract (span, category) pairs from an answer with <answer>/<type> tags."""
    return re.findall(r"<answer>(.*?)</answer>\s*<type>(.*?)</type>", answer, re.DOTALL)


def deduplicate_disclosures(disclosures: str) -> str:
    deduplicator = dspy.Predict(DeduplicateSignature)
    return deduplicator(disclosures=disclosures).answer


# ── Modules ────────────────────────────────────────────────────────────────────

class DisclosureProcessor(dspy.Module):
    """Extracts disclosures about the *user* only.

    Used to mine PUPA-SD from WildChat / LMSYS-Chat-1M, where conversations often
    contain fictional characters that must not count as self-disclosure.
    """

    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(DisclosureExtractorSignature)
        self.ordering = dspy.ChainOfThought(DisclosureOrderingSignature)

    def forward(self, text: str) -> dspy.Prediction:
        disclosures = self.extractor(text=text).answer
        disclosures = deduplicate_disclosures(disclosures)
        ordering_response = self.ordering(disclosures=disclosures)
        return dspy.Prediction(disclosures=disclosures, ordering=ordering_response.answer)


def load_optimized_processor() -> DisclosureProcessor:
    processor = DisclosureProcessor()
    processor.load(str(OPTIMIZED_PROCESSOR_PATH))
    return processor


class AnyDisclosureProcessor(dspy.Module):
    """Extracts disclosures about *any* person (Appendix A.1, second extractor).

    Used for k-anonymity estimation on prompts written by PAPILLON pipelines, which
    typically rephrase the user in the third person. Reuses the optimized ordering
    module from the self-disclosure processor.
    """

    def __init__(self, callbacks=None):
        super().__init__(callbacks)
        self.processor = load_optimized_processor()
        self.extractor = dspy.ChainOfThought(AnyDisclosureExtractorSignature)

    def forward(self, text: str) -> dspy.Prediction:
        disclosures = self.extractor(text=text).answer
        disclosures = deduplicate_disclosures(disclosures)
        ordering_response = self.processor.ordering(disclosures=disclosures)
        return dspy.Prediction(disclosures=disclosures, ordering=ordering_response.answer)
