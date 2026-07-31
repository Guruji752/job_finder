"""
Shared LLM extraction for job sources that return raw web content instead of
structured fields (e.g. Exa) — unlike JSearch, which already gives us
title/company/description directly.
"""
from app.config import settings
from app.llm.client import get_llm_client
from app.llm.json_parse import parse_json_response

EXTRACT_PROMPT = """Extract job posting details from this web page content.

URL: {url}
Page content:
{content}

Return a single JSON object with exactly these keys:
- "title": string (the job title)
- "company": string (employer name; use "Unknown" if not stated)
- "location": string (job location; use "Not specified" if not stated)
- "description": string (the full job description, responsibilities, and requirements as available)
- "employer_website": string or null (the company's own website, if mentioned separately from the job board)
- "is_single_job_posting": boolean (true if this page describes ONE specific job; false if it's a listing/search-results page, a general careers page, or anything else that isn't one concrete role)

Respond with ONLY the JSON object, no markdown fences, no other text."""

_MAX_CONTENT_CHARS = 4000


def extract_job_fields(url: str, content: str) -> dict:
    prompt = EXTRACT_PROMPT.format(url=url, content=content[:_MAX_CONTENT_CHARS])
    completion = get_llm_client().chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = completion.choices[0].message.content
    return parse_json_response(raw)
