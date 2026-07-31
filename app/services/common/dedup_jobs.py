"""
Deterministic dedup — no LLM call needed. Runs after search_agent merges
results from multiple sources (JSearch, Exa), which can both return the same
posting under different URLs.
"""
from app.job_sources.base import Job


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def dedup_jobs(jobs: list[Job]) -> list[Job]:
    seen_urls: set[str] = set()
    seen_title_company: set[tuple[str, str]] = set()
    unique: list[Job] = []

    for job in jobs:
        if job.url in seen_urls:
            continue

        title_company = (_normalize(job.title), _normalize(job.company))
        if title_company in seen_title_company:
            continue

        seen_urls.add(job.url)
        seen_title_company.add(title_company)
        unique.append(job)

    return unique
