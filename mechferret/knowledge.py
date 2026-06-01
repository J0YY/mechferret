"""Web + research knowledge sources (stdlib only).

Web search/fetch plus interpretability-specific knowledge bases (arXiv,
Neuronpedia) used by the agent's tools and the research planner. All over
``urllib`` so nothing extra needs installing.
"""

from __future__ import annotations

import json
import re
import html as html_lib
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

_UA = "mechferret/0.1 (interpretability research agent)"


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return ""


def _positive_int(value: Any, default: int, *, upper: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed <= 0:
        parsed = default
    if upper is not None:
        parsed = min(parsed, upper)
    return parsed


def _json_object(raw: bytes) -> dict:
    try:
        payload = json.loads(raw.decode("utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}

# --- web ----------------------------------------------------------------------------

def web_fetch(url: str, max_chars: int = 6000, timeout: int = 20) -> str:
    """Fetch a URL and return readable text (HTML stripped)."""

    url = _text(url).strip()
    if not url:
        return ""
    max_chars = _positive_int(max_chars, 6000, upper=100_000)
    timeout = _positive_int(timeout, 20, upper=120)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(2_000_000).decode("utf-8", errors="ignore")
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:max_chars]


def web_search(query: str, max_results: int = 12, timeout: int = 20) -> list[dict]:
    """General web search via DuckDuckGo's HTML endpoint (no API key)."""

    query = _text(query).strip()
    if not query:
        return []
    max_results = _positive_int(max_results, 12, upper=25)
    timeout = _positive_int(timeout, 20, upper=120)
    data = urllib.parse.urlencode({"q": query}).encode()
    req = urllib.request.Request(
        "https://html.duckduckgo.com/html/", data=data, headers={"User-Agent": _UA}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    results: list[dict] = []
    for m in re.finditer(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S):
        href = html_lib.unescape(m.group(1))
        title = html_lib.unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip()
        # DuckDuckGo wraps targets in a redirect; pull out uddg=
        target = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg", [href])[0]
        target = _text(target).strip()
        if not title or not target:
            continue
        results.append({"title": title, "url": target})
        if len(results) >= max_results:
            break
    return results


# --- arXiv (verified spec) ----------------------------------------------------------

_ATOM = "http://www.w3.org/2005/Atom"
_OPENSEARCH = "http://a9.com/-/spec/opensearch/1.1/"
_NS = {"a": _ATOM, "os": _OPENSEARCH}


def search_arxiv(query: str, max_results: int = 20, sort_by: str = "relevance", timeout: int = 30) -> tuple[int, list[dict]]:
    """Search arXiv. Returns (total_results, results). sort_by in {relevance, submittedDate, lastUpdatedDate}."""

    query = _text(query).strip()
    if not query:
        return 0, []
    max_results = _positive_int(max_results, 20, upper=50)
    sort_by = _text(sort_by).strip()
    if sort_by not in {"relevance", "submittedDate", "lastUpdatedDate"}:
        sort_by = "relevance"
    timeout = _positive_int(timeout, 30, upper=120)
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
    try:
        feed = ET.fromstring(raw)
    except ET.ParseError:
        return 0, []
    try:
        total = int(feed.findtext("os:totalResults", default="0", namespaces=_NS) or 0)
    except (TypeError, ValueError):
        total = 0
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
                url_abs = link.get("href") or url_abs
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

    timeout = _positive_int(timeout, 30, upper=120)
    headers = {"Content-Type": "application/json", "User-Agent": _UA}
    key = api_key or os.getenv("NEURONPEDIA_API_KEY")
    if key:
        headers["X-Api-Key"] = key
    req = urllib.request.Request(_NP_BASE + path, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return _json_object(resp.read())


def neuronpedia_search_explanations(model_id: str, query: str, api_key: str | None = None) -> dict:
    """Semantic search over SAE-feature explanations within a model."""

    model_id = _text(model_id).strip()
    query = _text(query).strip()
    if not model_id or not query:
        return {}
    return _np_post("/explanation/search", {"modelId": model_id, "query": query}, api_key)


def neuronpedia_feature(model_id: str, source: str, index: int, timeout: int = 30) -> dict:
    """Fetch a single SAE feature (modelId, source e.g. '6-res-jb', index)."""

    model_id = urllib.parse.quote(_text(model_id).strip(), safe="")
    source = urllib.parse.quote(_text(source).strip(), safe="")
    index = _positive_int(index, 0)
    timeout = _positive_int(timeout, 30, upper=120)
    if not model_id or not source:
        return {}
    url = f"{_NP_BASE}/feature/{model_id}/{source}/{index}"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return _json_object(resp.read())
