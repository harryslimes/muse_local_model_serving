#!/usr/bin/env python3
"""Muse Tool Server — lightweight HTTP service for companion tools.

Exposes:
  POST /tools/crawl   — crawl a URL, return clean extracted text
  POST /tools/search  — search the web via SearXNG, return results
  GET  /tools         — list available tools
  GET  /health        — health check

Uses crawl4ai (headless Chromium) for fetching JS-rendered pages,
then trafilatura for robust article/content extraction.
"""

import os

import httpx
import trafilatura
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Muse Tool Server")

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")
CRAWL_MAX_CHARS = int(os.getenv("CRAWL_MAX_CHARS", "32000"))
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "5"))

_crawler = None


async def _get_crawler():
    global _crawler
    if _crawler is None:
        from crawl4ai import AsyncWebCrawler, BrowserConfig
        browser_config = BrowserConfig(
            headless=True,
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            headers={
                "Accept-Language": "en-GB,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        _crawler = AsyncWebCrawler(config=browser_config)
        await _crawler.__aenter__()
    return _crawler


def _extract_with_trafilatura(html: str, url: str) -> str | None:
    """Extract clean article text from HTML using trafilatura."""
    return trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        include_links=True,
        include_images=False,
        favor_recall=True,
        output_format="txt",
    )


# --- Models ---

class CrawlRequest(BaseModel):
    url: str
    offset: int = 0  # character offset for pagination

class CrawlResponse(BaseModel):
    success: bool
    content: str
    error: str | None = None

class SearchRequest(BaseModel):
    query: str

class SearchResult(BaseModel):
    title: str
    url: str
    content: str

class SearchResponse(BaseModel):
    success: bool
    results: list[SearchResult]
    error: str | None = None

class ToolInfo(BaseModel):
    name: str
    description: str


# --- Endpoints ---

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/tools")
async def list_tools():
    tools = [
        ToolInfo(name="crawlWeb", description="Fetch and read a web page"),
        ToolInfo(name="searchWeb", description="Search the web for information"),
    ]
    return [t.model_dump() for t in tools]


@app.post("/tools/crawl", response_model=CrawlResponse)
async def crawl(req: CrawlRequest):
    url = req.url.strip()
    if not url:
        return CrawlResponse(success=False, content="", error="No URL provided.")
    try:
        # Fetch rendered HTML via headless browser
        crawler = await _get_crawler()
        result = await crawler.arun(url=url)
        raw_html = result.html or ""

        if not raw_html.strip():
            return CrawlResponse(
                success=False, content="",
                error=f"Page at {url} returned no HTML content.",
            )

        # Extract clean article text with trafilatura
        text = _extract_with_trafilatura(raw_html, url)

        # Fall back to crawl4ai markdown for non-article pages
        if not text or len(text.strip()) < 100:
            text = result.markdown or ""

        if not text.strip():
            return CrawlResponse(
                success=False, content="",
                error=f"Page at {url} returned no readable content.",
            )

        # Pagination
        total_len = len(text)
        offset = req.offset
        if offset >= total_len:
            return CrawlResponse(
                success=False, content="",
                error=f"Offset {offset} is past end of content ({total_len} chars).",
            )
        page = text[offset:offset + CRAWL_MAX_CHARS]
        has_more = (offset + CRAWL_MAX_CHARS) < total_len
        if has_more:
            next_offset = offset + CRAWL_MAX_CHARS
            page += (
                f"\n\n[Content truncated — showing chars {offset}–{offset + len(page)} "
                f"of {total_len}. Request offset={next_offset} for next page]"
            )
        return CrawlResponse(success=True, content=page)
    except Exception as exc:
        return CrawlResponse(success=False, content="", error=f"Failed to crawl {url}: {exc}")


@app.post("/tools/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    query = req.query.strip()
    if not query:
        return SearchResponse(success=False, results=[], error="No query provided.")
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json"},
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        raw_results = data.get("results", [])[:SEARCH_MAX_RESULTS]
        results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
            )
            for r in raw_results
        ]
        return SearchResponse(success=True, results=results)
    except Exception as exc:
        return SearchResponse(success=False, results=[], error=f"Web search failed: {exc}")


@app.on_event("shutdown")
async def shutdown():
    global _crawler
    if _crawler is not None:
        await _crawler.__aexit__(None, None, None)
        _crawler = None


if __name__ == "__main__":
    port = int(os.getenv("LISTEN_PORT", "4130"))
    uvicorn.run(app, host="0.0.0.0", port=port)
