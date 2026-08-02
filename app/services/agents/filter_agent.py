"""
Filter Agent — narrows the paused Tier-1 shortlist (filtered_jobs) based on a
free-form /chat message, before the graph is ever resumed to rank_jobs.

Unlike search_agent (which forces both tools every call), this is a genuine
tool-SELECTION agent: the LLM picks exactly one of two tools per request —
a fast deterministic Python filter for objective criteria (location, company,
a keyword), or an LLM-judgment filter for anything fuzzier that a plain field
match can't express (seniority implied by the text, remote-friendliness,
etc). Only one tool call is expected; extras are ignored.
"""
import json

from app.config import settings
from app.job_sources.base import Job
from app.llm.client import get_llm_client
from app.llm.json_parse import parse_json_response

FILTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "filter_by_field",
            "description": "Filter jobs by a substring match on one structured field. Use for objective criteria: location, company name, or a specific keyword that should appear in the title/description. Fast and deterministic — prefer this whenever the criteria is a plain match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "enum": ["location", "company", "title", "description"], "description": "Which field to filter on"},
                    "value": {"type": "string", "description": "Substring to match, case-insensitive"},
                },
                "required": ["field", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_judgment",
            "description": "Filter jobs using semantic judgment over the full description. Use only when the criteria can't be expressed as a simple field match — e.g. seniority level implied by the text, remote-friendliness, team size, tone of the posting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "criteria": {"type": "string", "description": "The natural-language criteria to judge each job against"},
                },
                "required": ["criteria"],
            },
        },
    },
]

FILTER_ROUTE_PROMPT = """A candidate is narrowing down a shortlist of {count} jobs with this request:

"{message}"

Call exactly ONE of the available tools to apply this filter — pick filter_by_field if the request is an objective match on location/company/keyword, or filter_by_judgment if it requires reading and judging the job descriptions."""

JUDGMENT_PROMPT = """Judge each job below against this criteria: "{criteria}"

{jobs_block}

Return a single JSON object with exactly this key:
- "matching_indices": list of integers (0-based indices of jobs that satisfy the criteria)

Respond with ONLY the JSON object, no markdown fences, no other text."""

_JUDGMENT_BATCH_SIZE = 5


def _chunk(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _filter_by_field(jobs: list[Job], field: str, value: str) -> list[Job]:
    needle = value.lower()
    return [job for job in jobs if needle in getattr(job, field, "").lower()]


def _judge_batch(batch: list[Job], criteria: str) -> list[Job]:
    jobs_block = "\n\n".join(
        f"{i}: {job.title} at {job.company}\n{job.description[:500]}"
        for i, job in enumerate(batch)
    )
    completion = get_llm_client().chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": JUDGMENT_PROMPT.format(criteria=criteria, jobs_block=jobs_block)}],
    )
    raw = completion.choices[0].message.content
    indices = parse_json_response(raw).get("matching_indices", [])
    return [batch[i] for i in indices if isinstance(i, int) and 0 <= i < len(batch)]


def _filter_by_judgment(jobs: list[Job], criteria: str) -> list[Job]:
    # Same lesson as tier2's batching: keep batches small (5) to avoid the
    # LLM silently dropping items or degrading on a long single-shot list.
    matches = []
    for batch in _chunk(jobs, _JUDGMENT_BATCH_SIZE):
        matches.extend(_judge_batch(batch, criteria))
    return matches


def filter_shortlist(jobs: list[Job], message: str) -> list[Job]:
    completion = get_llm_client().chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": FILTER_ROUTE_PROMPT.format(count=len(jobs), message=message)}],
        tools=FILTER_TOOLS,
        tool_choice="required",
    )
    tool_calls = completion.choices[0].message.tool_calls or []
    if not tool_calls:
        print("=== FILTER_SHORTLIST: LLM made no tool call, leaving list unchanged ===")
        return jobs

    tool_call = tool_calls[0]
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    print(f"=== FILTER_SHORTLIST: LLM chose '{name}' with args={args} ===")

    if name == "filter_by_field":
        result = _filter_by_field(jobs, args["field"], args["value"])
    elif name == "filter_by_judgment":
        result = _filter_by_judgment(jobs, args["criteria"])
    else:
        result = jobs

    print(f"=== FILTER_SHORTLIST: {len(jobs)} -> {len(result)} jobs ===")
    return result
