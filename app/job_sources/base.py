from typing import Protocol

from pydantic import BaseModel, Field


class Job(BaseModel):
    id: str
    title: str
    company: str
    location: str
    description: str
    url: str  # apply / source link
    employer_website: str | None = None
    source_platform: str | None = None
    posted_date: str | None = None
    # Best-effort contact details scraped from the description text (job-board
    # APIs don't expose recruiter contacts as structured fields). Often empty.
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)


class JobSource(Protocol):
    def search(self, query: str, location: str | None = None) -> list[Job]: ...
