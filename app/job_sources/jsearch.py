import requests

from app.config import settings
from app.job_sources.base import Job
from app.job_sources.contacts import extract_emails, extract_phones

BASE_URL = "https://api.openwebninja.com/jsearch/search-v2"


class JSearchSource:
    def __init__(self, country: str | None = None) -> None:
        self.country = country or settings.jsearch_country

    def search(
        self,
        query: str,
        location: str | None = None,
        date_posted: str = "week",
        num_pages: int = 2,  # confirmed: each page is billed as a separate request against the 200/month quota
    ) -> list[Job]:
        full_query = f"{query} in {location}" if location else query

        response = requests.get(
            BASE_URL,
            params={
                "query": full_query,
                "country": self.country,
                "date_posted": date_posted,
                "num_pages": num_pages,
            },
            headers={"x-api-key": settings.jsearch_api_key},
        )
        response.raise_for_status()
        raw_jobs = response.json().get("data", {}).get("jobs", [])

        return [self._to_job(raw) for raw in raw_jobs]

    def _to_job(self, raw: dict) -> Job:
        location_parts = [raw.get("job_city"), raw.get("job_state"), raw.get("job_country")]
        location = ", ".join(part for part in location_parts if part)

        description = raw.get("job_description", "")

        return Job(
            id=raw["job_id"],
            title=raw["job_title"],
            company=raw.get("employer_name", "Unknown"),
            location=location or "Not specified",
            description=description,
            url=raw.get("job_apply_link", ""),
            employer_website=raw.get("employer_website"),
            source_platform=raw.get("job_publisher"),
            posted_date=raw.get("job_posted_at_datetime_utc"),
            emails=extract_emails(description),
            phones=extract_phones(description),
        )
