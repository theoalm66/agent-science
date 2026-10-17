# Stage: formalize

Restate the idea below as a precise, testable claim.

IDEA:
$idea

DOMAIN KNOWLEDGE BASE (limits that may apply):
$knowledge_index

Rules:
- If the idea has gaps or ambiguities, choose the most standard textbook interpretation, state it
  in "assumptions", and continue. List the open points in "clarifying_questions" anyway.
- Set "underspecified" to true ONLY when no testable claim can be formed under any reasonable
  interpretation (e.g. "what about higher dimensions?" with nothing asserted). A claim with concrete
  numbers is never underspecified: if the numbers look wrong or inconsistent, that is a finding for
  the later stages to check, not a reason to stop.
- Search queries name concepts, not parameter values: "AWGN channel capacity bound", not
  "capacity at 30 dB SNR for 1 MHz".

Return JSON:
{
  "claim": "one or two sentences: what exactly is asserted, in technical terms",
  "domain": "the scientific field this belongs to",
  "assumptions": ["physical/mathematical assumptions the claim relies on or implicitly changes"],
  "variables": ["symbol: meaning [unit]"],
  "predicted_quantity": "what number or relation the idea predicts, e.g. 'capacity C in bit/s exceeding B log2(1+SNR)'",
  "standard_terms": ["established technical names for this or closely related concepts"],
  "search_queries": ["3 to 5 short literature search queries using standard terminology"],
  "underspecified": <true or false>,
  "clarifying_questions": ["open points the author should clarify; fill this even when not underspecified"]
}
