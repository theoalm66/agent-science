# Stage: limits

Check the claim against fundamental limits. This is the core of the assessment.

CLAIM:
$claim

ASSUMPTIONS: $assumptions
PREDICTED QUANTITY: $predicted_quantity

KNOWN LIMITS IN THIS DOMAIN:
$knowledge

Procedure:
1. Identify every limit that applies to the predicted quantity.
2. For each, list the limit's assumptions and determine which ones the idea actually relaxes.
   An idea that relaxes none of a limit's assumptions cannot exceed that limit.
3. If an assumption is relaxed, name the established technique that already exploits it.
4. Check dimensional consistency of the claim.

Return JSON:
{
  "applicable_limits": [{"name": "...", "statement": "...", "implication": "what it means for this claim"}],
  "relaxed_assumptions": ["assumptions the idea genuinely relaxes; empty if none"],
  "violates_known_limit": <true or false>,
  "violation_explanation": "if true: which limit, and why no relaxed assumption rescues the claim",
  "dimensional_check": "result of the dimensional analysis",
  "quantitative_question": "the single most decisive number to compute, e.g. 'number of spatial modes for 10 cm apertures at 1 km and 1550 nm'"
}
