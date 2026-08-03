"""
Search Agent — replaces the direct JSearch-only call in retrieve_jobs.
Uses native tool calling: the LLM is given jsearch_search and exa_search as
tools and decides the arguments; we execute the real functions ourselves and
skip a second LLM round-trip since we don't need a synthesized answer, just
the structured Job lists back.
"""
import json
from concurrent.futures import ThreadPoolExecutor

from app.config import settings
from app.job_sources.base import Job
from app.job_sources.exa import ExaSource
from app.job_sources.jsearch import JSearchSource
from app.llm.client import get_llm_client
from app.services.graph.stategraph import JobSearchState

_DATE_POSTED_ENUM = ["today", "3days", "week", "month", "all"]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "jsearch_search",
            "description": "Search job postings on JSearch, which aggregates listings from LinkedIn, Indeed, Naukri, and other major job boards.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Job search query, e.g. role and key skills"},
                    "location": {"type": "string", "description": "Location to search in, if known"},
                    "date_posted": {
                        "type": "string",
                        "enum": _DATE_POSTED_ENUM,
                        "description": "How recent the job posting must be",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exa_search",
            "description": "Search the open web via Exa for individual job posting pages (company career sites, smaller boards) not covered by JSearch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Job search query, e.g. role and key skills"},
                    "location": {"type": "string", "description": "Location to search in, if known"},
                    "date_posted": {
                        "type": "string",
                        "enum": _DATE_POSTED_ENUM,
                        "description": "How recent the job posting must be",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

_TOOL_SOURCES = {
    "jsearch_search": JSearchSource(),
    "exa_search": ExaSource(),
}

SEARCH_AGENT_PROMPT = """Find job postings for this search.

Query: {query}
Location: {location}
Date posted: {date_posted}

Call BOTH the jsearch_search and exa_search tools with this query, location,
and date_posted so we cover both sources."""


def _run_tool_call(
    tool_call,
    fallback_query: str,
    fallback_location: str | None,
    fallback_date_posted: str,
) -> tuple[str, list[Job]]:
    name = tool_call.function.name
    source = _TOOL_SOURCES.get(name)
    if source is None:
        return name, []
    args = json.loads(tool_call.function.arguments)
    jobs = source.search(
        args.get("query", fallback_query),
        args.get("location", fallback_location),
        args.get("date_posted", fallback_date_posted),
    )
    return name, jobs


def search_agent(state: JobSearchState):
    query = state["query"]
    location = state.get("location")
    date_posted = state.get("date_posted", "week")

    completion = get_llm_client().chat.completions.create(
        model=settings.chat_model,
        messages=[{
            "role": "user",
            "content": SEARCH_AGENT_PROMPT.format(
                query=query, location=location or "not specified", date_posted=date_posted,
            ),
        }],
        tools=TOOLS,
        tool_choice="required",
    )
    tool_calls = completion.choices[0].message.tool_calls or []
    print(f"=== SEARCH_AGENT: LLM requested {len(tool_calls)} tool call(s): {[tc.function.name for tc in tool_calls]} ===")

    with ThreadPoolExecutor(max_workers=len(tool_calls) or 1) as executor:
        results = list(executor.map(
            lambda tc: _run_tool_call(tc, query, location, date_posted), tool_calls,
        ))

    for name, jobs in results:
        print(f"=== SEARCH_AGENT: {name} -> {len(jobs)} jobs ===")

    raw_jobs = [job for _, jobs in results for job in jobs]
    new_retry_count = state.get("retry_count", 0) + 1
    print(f"=== SEARCH_AGENT: query='{query}' -> fetched {len(raw_jobs)} jobs total (retry_count now {new_retry_count}) ===")

    return {
        # Dumped to dict — see stategraph.py's note on why real Job instances
        # can't be stored directly in checkpointed state.
        "raw_jobs": [job.model_dump() for job in raw_jobs],
        "retry_count": new_retry_count,
    }
