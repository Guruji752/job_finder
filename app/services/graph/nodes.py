'''
Nodes for the job-search graph.

Rule to remember: a node READS from state and RETURNS a partial-state dict.
Whatever it returns is merged back into the shared state by LangGraph — that
(and only that) is how state changes. Routers/edges never mutate state.

Each node here is a thin wrapper over the pipeline functions you already built.
'''
from app.services.graph.stategraph import JobSearchState

from app.job_sources.jsearch import JSearchSource
from app.profile.digest import get_profile_digest
from app.matching.tier1 import rank_by_similarity
# tier2's rank_jobs shares a name with our rank_jobs node, so alias it to avoid shadowing.
from app.matching.tier2 import rank_jobs as gap_analyze_jobs
from app.services.graph.strategy import reformulate_query
from app.services.graph.dummy_source import load_dummy_jobs

# One source instance, reused across calls (same as main.py).
_job_source = JSearchSource()


def retrieve_profile(state: JobSearchState):
    # One-time setup: distill the candidate's resume into a ProfileDigest
    # (cached inside get_profile_digest, so re-entering is cheap).
    profile = get_profile_digest()
    return {"profile": profile}


def retrieve_jobs(state: JobSearchState):
    # Fetch raw jobs for the query. This is the node the retry-loop re-enters,
    # so it's also where we advance retry_count — the counter changes in a NODE,
    # which is the whole reason the loop can eventually terminate.
    jobs = _job_source.search(
        state["query"],
        state.get("location"),
        num_pages=state.get("num_pages", 2),
    )
    return {
        "raw_jobs": jobs,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def filter_jobs(state: JobSearchState):
    # Tier 1: cheap embedding-similarity ranking of ALL fetched jobs (no truncation).
    profile = state["profile"]
    ranked = rank_by_similarity(state["raw_jobs"], profile)
    return {"filtered_jobs": ranked}


def rank_jobs(state: JobSearchState):
    # Tier 2: expensive LLM gap analysis on the filtered jobs → RankedJob list,
    # sorted by match_score. This is a terminal node, so we also mark is_done.
    profile = state["profile"]
    ranked = gap_analyze_jobs(state["filtered_jobs"], profile)
    return {"ranked_jobs": ranked, "is_done": True}


def supervisor(state: JobSearchState):
    print("=== ENTERED SUPERVISOR ====")
    # Thin loop-back hub — the actual decision happens in route_supervisor (edges.py).
    # This node exists so rank_jobs and dummy_agent have somewhere to return control to.
    return {}


def dummy_agent(state: JobSearchState):
    print("=== ENTERED DUMMY AGENT ====")
    # Fallback specialist: too few real results, so supplement with a pre-analyzed
    # canned batch instead of paying for a second live source (e.g. Tavily).
    dummy_jobs = load_dummy_jobs()
    return {
        "ranked_jobs": state.get("ranked_jobs", []) + dummy_jobs,
        "dummy_used": True,
    }


def refine_query(state: JobSearchState):
    print("=== ENTERED IN REFINE QUERY ====")
    # Agentic step: the previous query returned too few jobs, so an LLM rewrites it
    # into a better one. We also record the OLD query in tried_queries (append reducer)
    # so the strategist never suggests a query we've already burned a search on.
    new_query = reformulate_query(
        state["query"],
        state.get("tried_queries", []),
        state["profile"],
    )
    return {"query": new_query, "tried_queries": [state["query"]]}
