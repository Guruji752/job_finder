import operator
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

from app.job_sources.base import Job
from app.matching.tier2 import RankedJob
from app.profile.digest import ProfileDigest


# Define your state — this is the backbone of your entire agent.
# Note: these are the real object types the nodes store (not dicts). TypedDict
# doesn't enforce them at runtime, but honest hints keep the code readable.
class JobSearchState(TypedDict):
    query: str                                # current job search query (refine_query may rewrite it)
    location: str | None                      # optional location filter for the search
    num_pages: int                            # how many result pages to fetch (billed per page)
    date_posted: str                          # recency filter: today/3days/week/month/all
    # Every query we've already searched. operator.add is a reducer: when a node
    # returns {"tried_queries": [q]}, LangGraph APPENDS it instead of overwriting —
    # so this accumulates across retry loops and refine_query never repeats a query.
    tried_queries: Annotated[list[str], operator.add]
    max_score:int
    raw_jobs: list[Job]                       # jobs fetched from the source
    filtered_jobs: list[Job]                  # jobs after Tier-1 similarity ranking
    profile: ProfileDigest | None             # candidate digest (renamed from retrieve_profile)
    ranked_jobs: list[RankedJob]              # final Tier-2 gap-analysed results
    messages: Annotated[list, add_messages]   # conversation history (unused until later phases)
    retry_count: int                          # search-retry counter (advanced in a node)
    supervisor_decision: str                  # LLM's routing decision, read by route_supervisor
    is_done: bool                             # are we finished?