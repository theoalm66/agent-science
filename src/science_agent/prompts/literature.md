# Stage: literature

Judge how this claim relates to existing work.

CLAIM:
$claim

STANDARD TERMS: $standard_terms

SEARCH RESULTS (title, year, source, abstract excerpt):
$papers

Use the search results and your own knowledge. If the results are empty or off-topic, rely on
your knowledge but say so in "summary". Do not invent paper titles or URLs; cite only the results
above or well-known textbooks/papers you are sure exist.

Return JSON:
{
  "established_field": "the established research area this idea belongs to",
  "novelty": "known | partially_known | novel | unclear",
  "closest_prior_work": [{"title": "...", "url": "url from results or empty", "relevance": "how it relates"}],
  "summary": "3-5 sentences: what is established, what (if anything) remains open"
}
