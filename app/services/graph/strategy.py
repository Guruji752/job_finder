"""
Search-query reformulation — the reasoning behind the retry loop.

When a search returns too few jobs, this asks the LLM to rewrite the query into
a better one (broader, more standard job-title phrasing, relaxed location) so the
next search attempt is more likely to succeed. This is what turns the retry loop
from "try the same thing again" into an actual agentic decision.
"""
from app.config import settings
from app.llm.client import get_llm_client
from app.llm.json_parse import parse_json_response
from app.profile.digest import ProfileDigest

REFINE_PROMPT = """You are a job-search strategist. The previous search returned too few results, so the query needs to be improved.

Candidate context (keep suggestions realistic for this person):
- Skills: {skills}
- Seniority: {seniority}
- Domains: {domains}

Original query: "{query}"
Already tried (do NOT repeat any of these): {tried}

Write ONE improved job-search query more likely to return results. Good strategies:
- broaden an over-specific title (e.g. "senior python django microservices dev" -> "python developer")
- use common job-board phrasing for the title
- drop rare/niche keywords
- relax an overly narrow location

Return ONLY a JSON object: {{"query": "<improved query>"}} — no markdown, no other text."""


def reformulate_query(
    current_query: str,
    tried_queries: list[str],
    profile: ProfileDigest,
) -> str:
    """Ask the LLM for a better query. Falls back to the current query if parsing fails."""
    print("Query reform")
    prompt = REFINE_PROMPT.format(
        skills=", ".join(profile.skills),
        seniority=profile.seniority,
        domains=", ".join(profile.domains),
        query=current_query,
        tried=", ".join(tried_queries) if tried_queries else "none",
    )
    completion = get_llm_client().chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = completion.choices[0].message.content
    parsed = parse_json_response(raw)
    # If the model returns nothing usable, keep the current query — the retry cap
    # will still stop the loop, so we can't spin forever.
    return parsed.get("query") or current_query
