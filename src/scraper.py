"""Scrape CDC MMWR reports for RAG ingestion."""
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data" / "mmwr"
BASE_URL  = "https://www.cdc.gov/mmwr"
INDEX_URL = "https://www.cdc.gov/mmwr/mmwr_wk/wk_cvol.html"

HEADERS = {"User-Agent": "Mozilla/5.0 (research project; public health RAG)"}


def fetch_recent_issues(n: int = 20) -> list[dict]:
    """Return up to n recent MMWR issue URLs and titles."""
    r = requests.get(INDEX_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    issues = []
    for a in soup.select("a[href*='/volumes/']")[:n * 2]:
        href = a.get("href", "")
        if not href.endswith(".htm") and not href.endswith(".html"):
            continue
        url = href if href.startswith("http") else "https://www.cdc.gov" + href
        issues.append({"title": a.get_text(strip=True), "url": url})
        if len(issues) >= n:
            break
    return issues


def scrape_issue_text(url: str) -> str:
    """Extract article body text from an MMWR HTML page."""
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body
    return main.get_text(separator="\n", strip=True) if main else ""


def scrape_and_save(n: int = 30):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    issues = fetch_recent_issues(n)
    print(f"Found {len(issues)} issues. Scraping...")
    for i, issue in enumerate(issues):
        slug = issue["url"].rstrip("/").split("/")[-1].replace(".htm", "")
        dest = DATA_DIR / f"{slug}.txt"
        if dest.exists():
            continue
        try:
            text = scrape_issue_text(issue["url"])
            dest.write_text(text, encoding="utf-8")
            print(f"  [{i+1}/{len(issues)}] {slug} ({len(text):,} chars)")
        except Exception as e:
            print(f"  [{i+1}] SKIP {issue['url']}: {e}")
        time.sleep(0.5)
    print("Done.")


if __name__ == "__main__":
    scrape_and_save(30)
