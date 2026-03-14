#!/usr/bin/env python3
"""Muse Tool Server — lightweight HTTP service for companion tools.

Exposes:
  POST /tools/crawl   — crawl a URL, return markdown content
  POST /tools/search  — search the web via SearXNG, return results
  GET  /tools         — list available tools
  GET  /health        — health check

Keeps heavy dependencies (crawl4ai/Playwright) out of the main backend.
"""

import os

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Muse Tool Server")

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")
CRAWL_MAX_CHARS = int(os.getenv("CRAWL_MAX_CHARS", "16000"))
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


# --- Models ---

class CrawlRequest(BaseModel):
    url: str

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
        from crawl4ai import CrawlerRunConfig
        # Try to extract main content, falling back to full page
        run_config = CrawlerRunConfig(
            # Common article/main content selectors
            css_selector="article, main, [role='main'], .article-body, .post-content, .entry-content",
            word_count_threshold=20,
        )
        crawler = await _get_crawler()
        result = await crawler.arun(url=url, config=run_config)
        markdown = result.markdown or ""
        # If CSS selector matched nothing useful, retry without selector
        if len(markdown.strip()) < 100:
            result = await crawler.arun(url=url)
            markdown = result.markdown or ""
        if len(markdown) > CRAWL_MAX_CHARS:
            markdown = markdown[:CRAWL_MAX_CHARS] + "\n\n[Content truncated]"
        if not markdown.strip():
            return CrawlResponse(
                success=False, content="",
                error=f"Page at {url} returned no readable content.",
            )
        return CrawlResponse(success=True, content=markdown)
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
