# Stage: formalize

Restate the idea below as a precise, testable claim.

IDEA:
$idea

DOMAIN KNOWLEDGE BASE (limits that may apply):
$knowledge_index

Rules (research mode: open to new results, but always state the established view):
- Never resolve an ambiguity by silently assuming the textbook setup. The author's setup or
  measurement may deliberately differ from it, and that difference may be the whole point.
- If an ambiguity changes how the claim has to be tested, set "underspecified" to true and ask
  about it in "clarifying_questions". Minor gaps that do not change the test: ask, but continue.
- Always fill "established_view": what established theory predicts for this situation under its
  standard assumptions, with numbers where you can compute them. This is the reference point, not a
  judgement of the idea.
- Always fill "what_must_be_true": if the idea or measurement is correct, which standard
  assumptions must fail or which effect must exist. This is where something new would be found.
- The input may be a short idea, a hypothesis, a derivation, a measurement report or a whole
  publication. Copy its decisive content VERBATIM into "key_content": central equations, every
  step of a derivation in order, measured values with units and conditions, and the main
  conclusions. Do not paraphrase, simplify or correct equations or numbers here; later stages
  check exactly what the author wrote.
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
  "clarifying_questions": ["open points the author should clarify; fill this even when not underspecified"],
  "input_type": "idea | hypothesis | derivation | measurement | publication",
  "key_content": ["verbatim equations, derivation steps (in order), measured values with units, main conclusions"],
  "established_view": "what standard theory predicts here, with numbers, under its usual assumptions",
  "what_must_be_true": ["standard assumptions that must fail, or effects that must exist, if the idea is correct"]
}
