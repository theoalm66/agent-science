# Stage: verdict

Give the final triage verdict, weighing all evidence below. The critic may be wrong too;
judge on the merits. Judge the original text, not only its restatement.

CLAIM (stage 1's restatement):
$claim

ORIGINAL TEXT (exactly as the author submitted it):
$original

LITERATURE: $literature

LIMITS: $limits

COMPUTATION: $computation

CRITIQUE: $critique

Verdict categories (choose exactly one):
- "wrong": contradicts a fundamental limit or contains an error, with no relaxed assumption to rescue it
- "known": correct, but already established; name the field
- "plausible_but_hard": allowed by known principles, not clearly established, with major open obstacles
- "promising": allowed, not clearly established, and a concrete, cheap next step could test it
- "underspecified": cannot be assessed until the author specifies more

Return JSON:
{
  "verdict": "wrong | known | plausible_but_hard | promising | underspecified",
  "confidence": <number from 0 to 1: probability that this verdict is correct>,
  "deciding_argument": "the one argument that decides the verdict",
  "relaxed_assumption": "which assumption of the relevant limit the idea relaxes, or 'none'",
  "salvage": "if wrong or known: the closest legitimate idea worth exploring instead",
  "next_step": "the cheapest concrete next step: a calculation, simulation, experiment or paper to read",
  "summary": "3-5 sentence summary for the author"
}
