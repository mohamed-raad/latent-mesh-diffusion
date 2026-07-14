"""
"Up-to-Date" Ingestion & Training Workflow.
Fetches docs from AI agent frameworks + Python/NodeJS lib docs + Q&A,
converts to clean markdown, creates datasets for mesh training.

Usage:
  uv run --no-sync --package noprop-mesh python scripts/ingest_docs.py
  uv run --no-sync --package noprop-mesh python scripts/ingest_docs.py --sources openai,python,nodejs --max-pages 20
"""
import os
import sys
import json
import re
import time
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ingested_data")
CACHE_DIR = os.path.join(DATA_DIR, "_cache")
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MeshDocIngester/1.0",
    "Accept": "text/html,application/xhtml+xml",
}


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _cache_key(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]


def _fetch(url: str, timeout: int = 15) -> str | None:
    cache_path = os.path.join(CACHE_DIR, _cache_key(url))
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    try:
        r = httpx.get(url, headers=_HEADERS, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
        content = r.text
        _ensure_dir(CACHE_DIR)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(content)
        time.sleep(0.5)
        return content
    except Exception as e:
        print(f"  [WARN] Failed to fetch {url}: {e}")
        return None


def _html_to_md(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.find("body") or soup
    md_text = md(str(main), heading_style="ATX", strip=["img", "svg"])
    md_text = re.sub(r"\n{3,}", "\n\n", md_text)
    md_text = re.sub(r"\[.*?\]\(.*?\)", "", md_text)
    lines = [l for l in md_text.split("\n") if len(l.strip()) > 10 or l.strip().startswith("#")]
    return "\n".join(lines).strip()


def _save_doc(topic: str, subtopic: str, idx: int, content: str):
    topic_dir = os.path.join(DATA_DIR, f"ingested_{topic}")
    _ensure_dir(topic_dir)
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", subtopic)[:40].strip("_")
    filename = f"{safe}_{idx:04d}.md"
    filepath = os.path.join(topic_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath


# ─── Source definitions ─────────────────────────────────────────────


def ingest_agent_frameworks(max_pages: int = 10) -> int:
    """Fetch docs from popular AI agent frameworks."""
    count = 0
    sources = [
        ("langchain", "https://python.langchain.com/docs/introduction/",
         ["https://python.langchain.com/docs/concepts/", "https://python.langchain.com/docs/tutorials/"]),
        ("crewai", "https://docs.crewai.com/introduction",
         ["https://docs.crewai.com/core-concepts/", "https://docs.crewai.com/how-to/"]),
        ("autogen", "https://microsoft.github.io/autogen/stable/",
         ["https://microsoft.github.io/autogen/stable/user-guide/index.html"]),
        ("semantic_kernel", "https://learn.microsoft.com/en-us/semantic-kernel/overview/",
         ["https://learn.microsoft.com/en-us/semantic-kernel/get-started/"]),
        ("crawlee", "https://crawlee.dev/docs/introduction",
         ["https://crawlee.dev/docs/quick-start", "https://crawlee.dev/examples/"]),
        ("llm_agents", "https://github.com/e2b-dev/ai-agents-sdk-docs",
         ["https://python.langchain.com/docs/modules/agents/"]),
    ]
    for name, main_url, extra_urls in sources:
        urls = [main_url] + extra_urls
        for i, url in enumerate(urls[:max_pages]):
            html = _fetch(url)
            if not html:
                continue
            text = _html_to_md(html)
            if len(text) < 200:
                continue
            chunks = _chunk_text(text, 2000)
            for ci, chunk in enumerate(chunks[:3]):
                doc = f"# Agent Framework: {name}\n## Source: {url}\n\n{chunk}\n\nTags: #ingested #agent #{name} #up_to_date"
                _save_doc("agent_frameworks", f"{name}_{i+1}", ci + 1, doc)
                count += 1
        print(f"  Agent '{name}': {count} chunks ingested")
    return count


def ingest_python_docs(max_pages: int = 10) -> int:
    """Fetch Python library documentation."""
    count = 0
    libs = [
        ("python_stdlib", "https://docs.python.org/3/tutorial/index.html",
         ["https://docs.python.org/3/library/index.html"]),
        ("fastapi", "https://fastapi.tiangolo.com/",
         ["https://fastapi.tiangolo.com/tutorial/"]),
        ("pydantic", "https://docs.pydantic.dev/latest/",
         ["https://docs.pydantic.dev/latest/concepts/models/"]),
        ("httpx", "https://www.python-httpx.org/",
         ["https://www.python-httpx.org/quickstart/"]),
        ("pandas", "https://pandas.pydata.org/docs/",
         ["https://pandas.pydata.org/docs/user_guide/index.html"]),
        ("numpy", "https://numpy.org/doc/stable/",
         ["https://numpy.org/doc/stable/user/absolute_beginners.html"]),
    ]
    for lib, main_url, extra_urls in libs:
        urls = [main_url] + extra_urls
        for url in urls[:max_pages]:
            html = _fetch(url)
            if not html:
                continue
            text = _html_to_md(html)
            if len(text) < 200:
                continue
            chunks = _chunk_text(text, 2000)
            for ci, chunk in enumerate(chunks[:3]):
                doc = f"# Python Library: {lib}\n## Source: {url}\n\n{chunk}\n\nTags: #ingested #python #{lib} #up_to_date"
                _save_doc("python_docs", f"{lib}_{url.split('/')[-2] if url.endswith('/') else 'main'}", ci + 1, doc)
                count += 1
        print(f"  Python '{lib}': chunks ingested")
    return count


def ingest_nodejs_docs(max_pages: int = 10) -> int:
    """Fetch NodeJS library documentation."""
    count = 0
    libs = [
        ("express", "https://expressjs.com/en/starter/hello-world.html",
         ["https://expressjs.com/en/guide/routing.html"]),
        ("nextjs", "https://nextjs.org/docs",
         ["https://nextjs.org/docs/app/building-your-application"]),
    ]
    for lib, main_url, extra_urls in libs:
        urls = [main_url] + extra_urls
        for url in urls[:max_pages]:
            html = _fetch(url)
            if not html:
                continue
            text = _html_to_md(html)
            if len(text) < 200:
                continue
            chunks = _chunk_text(text, 2000)
            for ci, chunk in enumerate(chunks[:3]):
                doc = f"# NodeJS Library: {lib}\n## Source: {url}\n\n{chunk}\n\nTags: #ingested #nodejs #{lib} #up_to_date"
                _save_doc("nodejs_docs", f"{lib}_{url.split('/')[-2] if url.endswith('/') else 'main'}", ci + 1, doc)
                count += 1
        print(f"  NodeJS '{lib}': chunks ingested")
    return count


def _fetch_arxiv_abstracts(query: str, max_results: int = 10) -> list[dict]:
    """Fetch real paper abstracts from arxiv API."""
    papers = []
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    try:
        xml_data = _fetch(url)
        if not xml_data:
            return papers
        import re as _re
        entries = _re.findall(r'<entry>(.*?)</entry>', xml_data, _re.DOTALL)
        for entry in entries:
            title = _re.search(r'<title>(.*?)</title>', entry, _re.DOTALL)
            summary = _re.search(r'<summary>(.*?)</summary>', entry, _re.DOTALL)
            paper_id = _re.search(r'<id>(.*?)</id>', entry)
            authors = _re.findall(r'<name>(.*?)</name>', entry)
            if title and summary:
                papers.append({
                    "title": title.group(1).strip(),
                    "summary": summary.group(1).strip().replace("\n", " "),
                    "id": paper_id.group(1).strip() if paper_id else "",
                    "authors": authors[:5],
                })
    except Exception as e:
        print(f"  [WARN] arxiv API error: {e}")
    return papers


def ingest_ai_research(max_pages: int = 10) -> int:
    """Fetch real AI/ML research papers from arxiv, HuggingFace, and AI blogs."""
    count = 0

    print("  Fetching real papers from arxiv (LLMs, AI agents, multimodal)...")
    for query in ["large+language+models", "AI+agents", "multimodal+learning", "diffusion+models"]:
        papers = _fetch_arxiv_abstracts(query, max_results=5)
        for i, p in enumerate(papers):
            authors = ", ".join(p["authors"]) if p["authors"] else "Unknown"
            doc = (
                f"# Paper: {p['title']}\n"
                f"## Authors: {authors}\n"
                f"## Source: {p['id']}\n\n"
                f"{p['summary']}\n\n"
                f"Tags: #ingested #ai #research #arxiv #paper #up_to_date\n"
            )
            _save_doc("ai_research", f"arxiv_{query[:20]}_{i+1}", 1, doc)
            count += 1
        print(f"    {len(papers)} papers from arxiv ({query})")

    sources = [
        ("huggingface_papers", "https://huggingface.co/papers",
         ["https://huggingface.co/blog"]),
        ("nvidia_research", "https://research.nvidia.com/publications",
         []),
        ("google_ai", "https://ai.googleblog.com/",
         []),
        ("openai_research", "https://openai.com/research",
         []),
        ("anthropic_research", "https://www.anthropic.com/research",
         []),
        ("deepmind_research", "https://deepmind.google/research/",
         []),
    ]
    for name, main_url, extra_urls in sources:
        urls = [main_url] + extra_urls
        for url in urls[:max_pages]:
            html = _fetch(url)
            if not html:
                continue
            text = _html_to_md(html)
            if len(text) < 200:
                continue
            chunks = _chunk_text(text, 2000)
            for ci, chunk in enumerate(chunks[:2]):
                doc = f"# AI Research: {name}\n## Source: {url}\n\n{chunk}\n\nTags: #ingested #ai #{name} #papers #up_to_date"
                _save_doc("ai_research", f"{name}_{ci}", 1, doc)
                count += 1
        print(f"    chunks ingested from {name}")

    return count


def ingest_q_and_a(max_pages: int = 15) -> int:
    """Fetch Q&A from Stack Overflow and GitHub discussions (top questions)."""
    count = 0
    topics = [
        ("python", "python", "python+asyncio+OR+python+decorators+OR+python+context+manager"),
        ("javascript", "javascript", "javascript+closures+OR+javascript+promises+OR+javascript+async"),
        ("agent_ai", "ai-agent", "langchain+OR+autogen+OR+crewai+OR+ai+agent"),
    ]
    for name, tag, query in topics:
        url = f"https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=votes&q={query}&site=stackoverflow&pagesize={min(max_pages, 10)}"
        try:
            r = httpx.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            items = data.get("items", [])[:max_pages]
            for i, item in enumerate(items):
                title = item.get("title", "Untitled")
                body_html = item.get("body", "")
                body_md = _html_to_md(body_html) if body_html else ""
                answer_url = f"https://api.stackexchange.com/2.3/questions/{item['question_id']}/answers?order=desc&sort=votes&site=stackoverflow&pagesize=3"
                answers = []
                try:
                    ar = httpx.get(answer_url, timeout=10)
                    ar.raise_for_status()
                    for a in ar.json().get("items", [])[:2]:
                        answers.append(_html_to_md(a.get("body", "")))
                except Exception:
                    pass
                doc = f"# Q&A: {title}\n## Tags: {tag}\n## Question\n\n{body_md}\n\n## Top Answers\n\n"
                if answers:
                    doc += "\n---\n".join(answers)
                else:
                    doc += "_No answers fetched_"
                doc += f"\n\nTags: #ingested #qa #{tag} #up_to_date #stackoverflow"
                _save_doc("qa", f"{name}_{i+1}", 1, doc)
                count += 1
            print(f"  Q&A '{name}': {len(items)} questions ingested")
        except Exception as e:
            print(f"  [WARN] Q&A '{name}' failed: {e}")
        time.sleep(1)
    return count


def _chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    chunks = []
    paragraphs = text.split("\n\n")
    current = ""
    for p in paragraphs:
        if len(current) + len(p) < max_chars:
            current += p + "\n\n"
        else:
            if current.strip():
                chunks.append(current.strip())
            current = p + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [text.strip()]


SOURCES = {
    "agent": ("AI Agent Frameworks", ingest_agent_frameworks),
    "python": ("Python Library Docs", ingest_python_docs),
    "nodejs": ("NodeJS Library Docs", ingest_nodejs_docs),
    "qa": ("Q&A (Stack Overflow)", ingest_q_and_a),
    "airesearch": ("AI/ML Research Papers", ingest_ai_research),
}


def run_all(sources: list[str] | None = None, max_pages: int = 10):
    _ensure_dir(DATA_DIR)
    print("=" * 60)
    print(" Up-to-Date Ingestion Workflow")
    print(" Fetches docs from AI frameworks, libraries, and Q&A")
    print(f" Output: {DATA_DIR}")
    print("=" * 60)
    print()

    source_keys = sources or list(SOURCES.keys())
    total = 0
    for key in source_keys:
        if key not in SOURCES:
            print(f"Unknown source '{key}', skipping")
            continue
        name, func = SOURCES[key]
        print(f"--- {name} ---")
        try:
            n = func(max_pages=max_pages)
            total += n
        except Exception as e:
            print(f"  [ERROR] {name} failed: {e}")
        print()

    print("=" * 60)
    print(f"Ingestion complete — {total} chunks in {DATA_DIR}")
    print("Train: train_from_text.bat ingested_data")
    print("=" * 60)
    return total


def main():
    parser = argparse.ArgumentParser(description="Up-to-Date Ingestion Workflow")
    parser.add_argument("--sources", nargs="+", choices=list(SOURCES.keys()) + ["all"],
                        default=["all"], help="Sources to ingest (default: all)")
    parser.add_argument("--max-pages", type=int, default=10,
                        help="Max pages per source (default: 10)")
    args = parser.parse_args()
    srcs = list(SOURCES.keys()) if "all" in (args.sources or ["all"]) else args.sources
    run_all(sources=srcs, max_pages=args.max_pages)


if __name__ == "__main__":
    main()
