# Stage: compute

Write a short Python script that answers this question numerically or symbolically:

QUESTION: $question

CONTEXT:
Claim: $claim
Relevant limits: $limits

Requirements:
- Python 3 standard library plus numpy, scipy, sympy only. No network, no files, no plots.
- Use realistic, explicitly stated parameter values; print them.
- Finish within a few seconds.
- The LAST line printed must be: RESULT_JSON: {...}
  with keys "quantity", "value", "unit", "comparison" (the claimed value vs the limit),
  and "interpretation" (one sentence).
$previous_error
Return JSON:
{
  "purpose": "what the script computes",
  "code": "the complete Python script as a string"
}
