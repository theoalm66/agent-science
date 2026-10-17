import httpx

from science_agent.tools.literature import LiteratureSearch

ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <published>2012-03-01T00:00:00Z</published>
    <title>Capacity limits of
      OAM multiplexing</title>
    <summary>We show OAM is a subset of spatial modes.</summary>
  </entry>
</feed>"""

S2_JSON = {
    "data": [
        {"title": "Capacity Limits of OAM Multiplexing", "url": "https://s2/x", "year": 2012,
         "abstract": "dup", "citationCount": 10},
        {"title": "Communicating with waves between volumes", "url": "https://s2/y", "year": 2000,
         "abstract": "modes", "citationCount": 500},
    ]
}


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_search_merges_sources_and_deduplicates():
    def handler(request: httpx.Request):
        if "arxiv" in request.url.host:
            return httpx.Response(200, text=ARXIV_XML)
        return httpx.Response(200, json=S2_JSON)

    papers = LiteratureSearch(client=_client(handler)).search(["oam capacity"])
    assert [p.title for p in papers] == [
        "Capacity limits of OAM multiplexing",
        "Communicating with waves between volumes",
    ]
    assert papers[0].year == 2012 and papers[0].source == "arxiv"
    assert papers[1].citations == 500


def test_network_errors_degrade_to_empty():
    def handler(request):
        raise httpx.ConnectError("offline")

    assert LiteratureSearch(client=_client(handler)).search(["x"]) == []


def test_s2_api_key_header_sent():
    seen = {}

    def handler(request):
        if "semanticscholar" in request.url.host:
            seen["key"] = request.headers.get("x-api-key")
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, text=ARXIV_XML)

    LiteratureSearch(s2_api_key="secret", client=_client(handler)).semantic_scholar("q")
    assert seen["key"] == "secret"


def test_arxiv_query_requires_all_terms():
    from science_agent.tools.literature import arxiv_query

    assert arxiv_query("Capacity of OAM multiplexing") == "all:capacity AND all:oam AND all:multiplexing"
    assert arxiv_query("the of") == "all:the of"


def test_arxiv_falls_back_to_loose_query_when_strict_finds_nothing():
    queries = []

    def handler(request):
        q = request.url.params["search_query"]
        queries.append(q)
        empty = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        return httpx.Response(200, text=empty if " AND " in q else ARXIV_XML)

    papers = LiteratureSearch(client=_client(handler)).arxiv("oam capacity")
    assert queries == ["all:oam AND all:capacity", "all:oam capacity"]
    assert len(papers) == 1


class FakeTime:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(round(seconds, 2))
        self.now += seconds

    def clock(self):
        return self.now


def _s2_search(handler, t):
    return LiteratureSearch(client=_client(handler), sleep=t.sleep, clock=t.clock)


def test_s2_retries_429_honouring_retry_after():
    t = FakeTime()
    replies = iter([httpx.Response(429, headers={"retry-after": "5"}), httpx.Response(200, json=S2_JSON)])
    papers = _s2_search(lambda r: next(replies), t).semantic_scholar("q")
    assert len(papers) == 2
    assert 5.0 in t.sleeps


def test_s2_persistent_429_stops_querying_for_rest_of_search():
    t = FakeTime()
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429)

    search = _s2_search(handler, t)
    assert search.semantic_scholar("first") == []
    n = len(calls)
    assert n == 4  # S2_MAX_ATTEMPTS
    assert search.semantic_scholar("second") == []
    assert len(calls) == n  # no further requests once rate limited


def test_s2_requests_are_spaced_out():
    t = FakeTime()
    search = _s2_search(lambda r: httpx.Response(200, json={"data": []}), t)
    search.semantic_scholar("a")
    search.semantic_scholar("b")
    assert t.sleeps == [1.1]
