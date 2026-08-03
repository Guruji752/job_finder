'''
Nodes for the job-search graph.

Rule to remember: a node READS from state and RETURNS a partial-state dict.
Whatever it returns is merged back into the shared state by LangGraph — that
(and only that) is how state changes. Routers/edges never mutate state.

Each node here is a thin wrapper over the pipeline functions you already built.

Second rule (see stategraph.py): raw_jobs/filtered_jobs/ranked_jobs/profile
are stored in state as plain dicts, not real Job/ProfileDigest/RankedJob
instances — required for the Redis checkpointer. Nodes reconstruct a typed
instance locally when they need attribute access, and dump back to dict
(.model_dump()) before returning.
'''
from app.services.graph.stategraph import JobSearchState

from app.job_sources.base import Job
from app.profile.digest import ProfileDigest, get_profile_digest
from app.matching.tier1 import rank_by_similarity
# tier2's rank_jobs shares a name with our rank_jobs node, so alias it to avoid shadowing.
from app.matching.tier2 import rank_jobs as gap_analyze_jobs
from app.services.graph.strategy import reformulate_query
from app.services.graph.judge_result import judge_results
from app.services.agents.search_agent import search_agent
from app.services.common.dedup_jobs import dedup_jobs



def retrieve_profile(state: JobSearchState):
    # One-time setup: distill the candidate's resume into a ProfileDigest
    # (cached inside get_profile_digest, so re-entering is cheap).
    profile = get_profile_digest()
    print(f"=== RETRIEVE_PROFILE: skills={profile.skills}, seniority={profile.seniority}, domains={profile.domains}, years_experience={profile.years_experience} ===")
    return {"profile": profile.model_dump()}


def retrieve_jobs(state: JobSearchState):
    # Fetch raw jobs for the query. This is the node the retry-loop re-enters,
    # so it's also where we advance retry_count — the counter changes in a NODE,
    # which is the whole reason the loop can eventually terminate.
    results = search_agent(state)
    print(f"=== RETRIEVE_JOBS: query='{state['query']}' -> fetched {len(results.get('raw_jobs', []))} jobs (retry_count now {results['retry_count']}) ===")
    return results

def filter_jobs(state: JobSearchState):
    # Tier 1: cheap embedding-similarity ranking of ALL fetched jobs (no truncation).
    profile = ProfileDigest(**state["profile"])
    raw_jobs = [Job(**j) for j in state["raw_jobs"]]
    ranked = rank_by_similarity(raw_jobs, profile)
    print(f"=== FILTER_JOBS: {len(ranked)} jobs after Tier-1 similarity ranking ===")
    return {"filtered_jobs": [job.model_dump() for job in ranked]}


def rank_jobs(state: JobSearchState):
    # Tier 2: expensive LLM gap analysis on the filtered jobs → RankedJob list,
    # sorted by match_score. This is a terminal node, so we also mark is_done.
    profile = ProfileDigest(**state["profile"])
    filtered_jobs = [Job(**j) for j in state["filtered_jobs"]]
    ranked = gap_analyze_jobs(filtered_jobs, profile)
    print(f"=== RANK_JOBS: {len(ranked)} jobs after Tier-2 gap analysis ===")
    return {"ranked_jobs": [rj.model_dump() for rj in ranked], "is_done": True}


def supervisor(state: JobSearchState):
    print("=== ENTERED SUPERVISOR ====")
    # First visit, nothing to judge yet — skip the LLM call entirely and just kick
    # off the search. No point paying for reasoning when there's no data to reason about.
    if not state.get('ranked_jobs'):
        return {"supervisor_decision": "retrieve_profile"}

    # TEMP: quality judgment disabled for now — accept whatever ranked_jobs we got
    # on the first pass instead of retrying on low match_score. Commented out
    # (not removed) so it's easy to re-enable later.
    print(f"=== SUPERVISOR: accepting results as-is (ranked_jobs={len(state.get('ranked_jobs', []))}) ===")
    return {"supervisor_decision": "end"}

    # decision, max_score = judge_results(ProfileDigest(**state["profile"]), state)
    #
    # retry_count = state.get('retry_count', 0)
    # print(f"=== SUPERVISOR: LLM decided '{decision}' (retry_count={retry_count}, ranked_jobs={len(state.get('ranked_jobs', []))}, max_score={max_score}) ===")
    #
    # if retry_count > 2:
    #     # Hard cap — never let the LLM keep restarting the search past this point,
    #     # regardless of what it decides. Same principle as should_retry's cap.
    #     decision = "end"
    #     print(f"=== SUPERVISOR: retry_count cap hit, overriding to '{decision}' ===")
    #
    # return {"supervisor_decision": decision, "max_score": max_score}


def refine_query(state: JobSearchState):
    print("=== ENTERED IN REFINE QUERY ====")
    # Agentic step: the previous query returned too few jobs, so an LLM rewrites it
    # into a better one. We also record the OLD query in tried_queries (append reducer)
    # so the strategist never suggests a query we've already burned a search on.
    new_query = reformulate_query(
        state["query"],
        state.get("tried_queries", []),
        ProfileDigest(**state["profile"]),
    )
    return {"query": new_query, "tried_queries": [state["query"]]}

def filter_duplicate(state:JobSearchState):
    print("===== Filtered Duplicate Jobs =======")
    jobs = [Job(**j) for j in state['raw_jobs']]
    unique = dedup_jobs(jobs)
    return {"raw_jobs": [job.model_dump() for job in unique]}
