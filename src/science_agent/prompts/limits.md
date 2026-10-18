# Stage: limits

Check the claim against fundamental limits. This is the core of the assessment.

CLAIM:
$claim

ASSUMPTIONS: $assumptions
PREDICTED QUANTITY: $predicted_quantity

KEY CONTENT OF THE SOURCE (verbatim from the author):
$key_content

KNOWN LIMITS IN THIS DOMAIN:
$knowledge

Procedure:
1. Identify every limit that applies to the predicted quantity.
2. For each, list the limit's assumptions and determine which ones the idea actually relaxes.
   An idea that relaxes none of a limit's assumptions cannot exceed that limit.
3. If an assumption is relaxed, name the established technique that already exploits it.
4. Check dimensional consistency of the claim.
5. If the key content contains a derivation, check every step in order: is it mathematically valid,
   dimensionally consistent, and are its assumptions stated? Name the first step that fails.

Return JSON:
{
  "applicable_limits": [{"name": "...", "statement": "...", "implication": "what it means for this claim"}],
  "relaxed_assumptions": ["assumptions the idea genuinely relaxes; empty if none"],
  "violates_known_limit": <true or false>,
  "violation_explanation": "if true: which limit, and why no relaxed assumption rescues the claim",
  "dimensional_check": "result of the dimensional analysis",
  "derivation_check": "for derivations: the first invalid step and why, or 'all steps valid'; 'n/a' if there is no derivation",
  "quantitative_question": "the single most decisive number to compute, e.g. 'number of spatial modes for 10 cm apertures at 1 km and 1550 nm'"
}
