import operator
from typing import TypedDict, Annotated
from langgraph.graph import add_messages


# Define your state — this is the backbone of your entire agent.
# raw_jobs/filtered_jobs/ranked_jobs/profile are stored as plain dicts, NOT
# the real Job/ProfileDigest/RankedJob instances. This is required because
# the Redis checkpointer serializes state between steps, and LangChain's
# serde intentionally does NOT reconstruct arbitrary custom Pydantic classes
# on deserialize (arbitrary class reconstruction from external data is a
# security risk) — it hands back a raw envelope dict instead, which silently
# breaks any node that expects real attribute access (confirmed: this broke
# both Job.title and ProfileDigest.skills after a real Redis-backed resume).
# Convention: nodes reconstruct a typed instance locally when they need
# attribute access, and dump back to dict (.model_dump()) before returning.
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
    raw_jobs: list[dict]                      # jobs fetched from the source (dumped Job dicts)
    filtered_jobs: list[dict]                 # jobs after Tier-1 similarity ranking (dumped Job dicts)
    profile: dict | None                      # candidate digest (dumped ProfileDigest dict)
    ranked_jobs: list[dict]                   # final Tier-2 gap-analysed results (dumped RankedJob dicts)
    messages: Annotated[list, add_messages]   # conversation history (unused until later phases)
    retry_count: int                          # search-retry counter (advanced in a node)
    supervisor_decision: str                  # LLM's routing decision, read by route_supervisor
    is_done: bool                             # are we finished?
