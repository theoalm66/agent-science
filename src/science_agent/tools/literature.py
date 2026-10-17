"""Literature search over arXiv and Semantic Scholar. Failures degrade to empty results."""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass

import httpx

log = logging.getLogger(__name__)

ARXIV_URL = "https://export.arxiv.org/api/query"
S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_ATOM = {"a": "http://www.w3.org/2005/Atom"}


@dataclass
class Paper:
    title: str
    url: str
    year: int | None
    abstract: str
    source: str
    citations: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(s: str | None) -> str:
    return " ".join((s or "").split())


def parse_arxiv(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("a:entry", _ATOM):
        published = entry.findtext("a:published", default="", namespaces=_ATOM)
        papers.append(
            Paper(
                title=_clean(entry.findtext("a:title", namespaces=_ATOM)),
                url=_clean(entry.findtext("a:id", namespaces=_ATOM)),
                year=int(published[:4]) if published[:4].isdigit() else None,
                abstract=_clean(entry.findtext("a:summary", namespaces=_ATOM)),
                source="arxiv",
            )
        )
    return [p for p in papers if p.title]


def parse_semantic_scholar(data: dict) -> list[Paper]:
    papers = []
    for item in data.get("data") or []:
        papers.append(
            Paper(
                title=_clean(item.get("title")),
                url=item.get("url") or "",
                year=item.get("year"),
                abstract=_clean(item.get("abstract")),
                source="semantic_scholar",
                citations=item.get("citationCount"),
            )
        )
    return [p for p in papers if p.title]


_STOPWORDS = {"a", "an", "and", "the", "of", "for", "in", "on", "to", "with", "by", "or", "via", "using"}


def arxiv_query(query: str) -> str:
    """Turn free text into an arXiv query that requires every meaningful term."""
    terms = [t for t in re.findall(r"[\w\-]+", query.lower()) if t not in _STOPWORDS]
    if not terms:
        return f"all:{query}"
    return " AND ".join(f"all:{t}" for t in terms)


class LiteratureSearch:
    def __init__(
        self,
        max_results: int = 5,
        s2_api_key: str | None = None,
        client: httpx.Client | None = None,
    ):
        self.max_results = max_results
        self.s2_api_key = s2_api_key
        self.client = client or httpx.Client(timeout=30, follow_redirects=True)

    def arxiv(self, query: str) -> list[Paper]:
        # arXiv ORs bare words ("all:a b c" matches any of them), which floods results with
        # off-topic papers. Require every term; fall back to the loose query if that finds nothing.
        strict = arxiv_query(query)
        papers = self._arxiv_request(strict, query)
        if not papers and strict != f"all:{query}":
            papers = self._arxiv_request(f"all:{query}", query)
        return papers

    def _arxiv_request(self, search_query: str, query: str) -> list[Paper]:
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": self.max_results,
            "sortBy": "relevance",
        }
        try:
            resp = self.client.get(ARXIV_URL, params=params)
            resp.raise_for_status()
            return parse_arxiv(resp.text)
        except (httpx.HTTPError, ET.ParseError) as e:
            log.warning("arXiv search failed for %r: %s", query, e)
            return []

    def semantic_scholar(self, query: str) -> list[Paper]:
        params = {
            "query": query,
            "limit": self.max_results,
            "fields": "title,abstract,year,url,citationCount",
        }
        headers = {"x-api-key": self.s2_api_key} if self.s2_api_key else {}
        for attempt in range(3):
            try:
                resp = self.client.get(S2_URL, params=params, headers=headers)
                if resp.status_code == 429 and attempt < 2:  # unauthenticated limit is tight
                    time.sleep(2 ** (attempt + 1))
                    continue
                resp.raise_for_status()
                return parse_semantic_scholar(resp.json())
            except (httpx.HTTPError, ValueError) as e:
                log.warning("Semantic Scholar search failed for %r: %s", query, e)
                return []
        return []

    def search(self, queries: list[str]) -> list[Paper]:
        """Search both sources for every query; de-duplicate by normalized title."""
        seen: set[str] = set()
        results: list[Paper] = []
        for q in queries:
            for paper in self.arxiv(q) + self.semantic_scholar(q):
                key = "".join(ch for ch in paper.title.lower() if ch.isalnum())
                if key not in seen:
                    seen.add(key)
                    results.append(paper)
        return results
