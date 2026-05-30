"""Web + research knowledge sources (stdlib only).

Web search/fetch plus interpretability-specific knowledge bases (arXiv,
Neuronpedia) used by the agent's tools and the research planner. All over
``urllib`` so nothing extra needs installing.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

_UA = "mechferret/0.1 (interpretability research agent)"

# --- web ----------------------------------------------------------------------------

def web_fetch(url: str, max_chars: int = 6000, timeout: int = 20) -> str:
    """Fetch a URL and return readable text (HTML stripped)."""

    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(2_000_000).decode("utf-8", errors="ignore")
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:max_chars]


def web_search(query: str, max_results: int = 8, timeout: int = 20) -> list[dict]:
    """General web search via DuckDuckGo's HTML endpoint (no API key)."""

    data = urllib.parse.urlencode({"q": query}).encode()
    req = urllib.request.Request(
        "https://html.duckduckgo.com/html/", data=data, headers={"User-Agent": _UA}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    results: list[dict] = []
    for m in re.finditer(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S):
        href, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
        # DuckDuckGo wraps targets in a redirect; pull out uddg=
        target = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg", [href])[0]
        results.append({"title": title, "url": target})
        if len(results) >= max_results:
            break
    return results


# --- arXiv (verified spec) ----------------------------------------------------------

_ATOM = "http://www.w3.org/2005/Atom"
_OPENSEARCH = "http://a9.com/-/spec/opensearch/1.1/"
_NS = {"a": _ATOM, "os": _OPENSEARCH}


def search_arxiv(query: str, max_results: int = 10, sort_by: str = "relevance", timeout: int = 30) -> tuple[int, list[dict]]:
    """Search arXiv. Returns (total_results, results). sort_by in {relevance, submittedDate, lastUpdatedDate}."""

    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    feed = ET.fromstring(raw)
    total = int(feed.findtext("os:totalResults", default="0", namespaces=_NS))
    results: list[dict] = []
    for entry in feed.findall("a:entry", _NS):
        arxiv_id = entry.findtext("a:id", default="", namespaces=_NS) or ""
        if "/api/errors" in arxiv_id:
            continue
        title = entry.findtext("a:title", default="", namespaces=_NS) or ""
        summary = entry.findtext("a:summary", default="", namespaces=_NS) or ""
        authors = [
            (a.findtext("a:name", default="", namespaces=_NS) or "").strip()
            for a in entry.findall("a:author", _NS)
        ]
        url_abs, url_pdf = arxiv_id, None
        for link in entry.findall("a:link", _NS):
            if link.get("rel") == "alternate":
                url_abs = link.get("href")
            elif link.get("title") == "pdf":
                url_pdf = link.get("href")
        results.append({
            "title": " ".join(title.split()),
            "abstract": " ".join(summary.split()),
            "authors": authors,
            "published": entry.findtext("a:published", default="", namespaces=_NS),
            "url": url_abs,
            "pdf_url": url_pdf,
        })
    return total, results


# --- Neuronpedia (verified endpoints) -----------------------------------------------

_NP_BASE = "https://neuronpedia.org/api"


def _np_post(path: str, payload: dict, api_key: str | None, timeout: int = 30) -> dict:
    import os

    headers = {"Content-Type": "application/json", "User-Agent": _UA}
    key = api_key or os.getenv("NEURONPEDIA_API_KEY")
    if key:
        headers["X-Api-Key"] = key
    req = urllib.request.Request(_NP_BASE + path, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def neuronpedia_search_explanations(model_id: str, query: str, api_key: str | None = None) -> dict:
    """Semantic search over SAE-feature explanations within a model."""

    return _np_post("/explanation/search", {"modelId": model_id, "query": query}, api_key)


def neuronpedia_feature(model_id: str, source: str, index: int, timeout: int = 30) -> dict:
    """Fetch a single SAE feature (modelId, source e.g. '6-res-jb', index)."""

    url = f"{_NP_BASE}/feature/{model_id}/{source}/{index}"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))
