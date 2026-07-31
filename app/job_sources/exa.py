from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import requests

from app.config import settings
from app.job_sources.base import Job
from app.job_sources.contacts import extract_emails, extract_phones
from app.job_sources.extract import extract_job_fields

BASE_URL = "https://api.exa.ai/search"

# Exa has no built-in "date_posted" concept like JSearch — it takes an explicit
# ISO date range instead, so we translate the same recency labels into days back.
_DATE_POSTED_DAYS = {
    "today": 1,
    "3days": 3,
    "week": 7,
    "month": 30,
}


def _start_published_date(date_posted: str | None) -> str | None:
    days = _DATE_POSTED_DAYS.get(date_posted or "")
    if days is None:  # "all" or unrecognized -> no date filter
        return None
    start = datetime.now(timezone.utc) - timedelta(days=days)
    return start.strftime("%Y-%m-%dT%H:%M:%S.000Z")


class ExaSource:
    def search(self, query: str, location: str | None = None, date_posted: str | None = None) -> list[Job]:
        full_query = f"{query} jobs in {location}" if location else f"{query} jobs"

        payload = {
            "query": full_query,
            "type": "auto",
            "numResults": 10,
            "contents": {
                "text": {
                    "maxCharacters": 3000,
                    "verbosity": "standard",
                }
            },
        }
        start_published_date = _start_published_date(date_posted)
        if start_published_date:
            payload["startPublishedDate"] = start_published_date

        response = requests.post(
            BASE_URL,
            headers={"Authorization": f"Bearer {settings.exa_api_key}"},
            json=payload,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return []

        # Extraction is I/O-bound (LLM call per result) — run concurrently
        # instead of one-at-a-time, same fix as Tier-2's per-job calls.
        with ThreadPoolExecutor(max_workers=len(results)) as executor:
            jobs = executor.map(self._to_job, results)
        return [job for job in jobs if job is not None]

    def _to_job(self, result: dict) -> Job | None:
        content = result.get("text", "")
        fields = extract_job_fields(result["url"], content)

        if not fields.get("is_single_job_posting", True):
            return None  # listing/search-results page, not an actual job posting

        description = fields.get("description", "")
        return Job(
            id=result["url"],
            title=fields.get("title") or "Unknown",
            company=fields.get("company") or "Unknown",
            location=fields.get("location") or "Not specified",
            description=description,
            url=result["url"],
            employer_website=fields.get("employer_website"),
            source_platform="Exa",
            posted_date=result.get("publishedDate"),
            emails=extract_emails(description),
            phones=extract_phones(description),
        )
