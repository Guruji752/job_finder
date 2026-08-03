import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.job_sources.base import Job
from app.matching.tier2 import RankedJob
from app.services.agents.filter_agent import filter_shortlist
from app.services.graph.edges import graph_app

app = FastAPI(title="Job Finder")


class SearchRequest(BaseModel):
    query: str
    location: str | None = None
    num_pages: int = 2
    date_posted: str = "week"  # one of: today, 3days, week, month, all


class SearchStartResponse(BaseModel):
    thread_id: str
    filtered_jobs: list[Job]  # cheap Tier-1 shortlist, awaiting approval before Tier-2


class ChatRequest(BaseModel):
    thread_id: str
    message: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search")
def search(request: SearchRequest) -> SearchStartResponse:
    print("===== GRAPH INVOKE ======")
    # The endpoint hands the query to the LangGraph agent, which runs:
    # retrieve_profile → retrieve_jobs → filter_jobs → (retry if too few) →
    # then PAUSES before rank_jobs (interrupt_before, see edges.py) so the
    # expensive Tier-2 gap analysis only runs after explicit approval.
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    paused_state = graph_app.invoke(
        {
            "query": request.query,
            "location": request.location,
            "num_pages": request.num_pages,
            "date_posted": request.date_posted,
            "retry_count": 0,
        },
        config=config,
    )
    print(f"=== GRAPH PAUSED before rank_jobs, thread_id={thread_id} ===")
    return SearchStartResponse(thread_id=thread_id, filtered_jobs=paused_state["filtered_jobs"])


@app.post("/chat")
def chat(request: ChatRequest) -> SearchStartResponse:
    print(f"===== CHAT FILTER, thread_id={request.thread_id} ======")
    config = {"configurable": {"thread_id": request.thread_id}}
    # Graph is paused at this point (interrupt_before rank_jobs) — read the
    # current shortlist straight from the checkpoint, no invoke() needed.
    # State stores plain dicts (see stategraph.py) — reconstruct real Job
    # instances for filter_shortlist, then dump back before persisting.
    current_jobs = [Job(**j) for j in graph_app.get_state(config).values["filtered_jobs"]]
    narrowed = filter_shortlist(current_jobs, request.message)
    narrowed_dicts = [job.model_dump() for job in narrowed]
    # Patch the checkpoint directly. This does NOT run any graph nodes — the
    # graph stays paused at the same interrupt point (before rank_jobs), so
    # /chat can be called repeatedly to keep narrowing before ever resuming.
    graph_app.update_state(config, {"filtered_jobs": narrowed_dicts})
    return SearchStartResponse(thread_id=request.thread_id, filtered_jobs=narrowed_dicts)


@app.post("/search/{thread_id}/reset")
def reset_filter(thread_id: str) -> SearchStartResponse:
    print(f"===== RESET FILTER, thread_id={thread_id} ======")
    config = {"configurable": {"thread_id": thread_id}}
    # Every /chat narrowing writes a NEW checkpoint rather than overwriting —
    # the original shortlist is still in this thread's history. Every
    # checkpoint from the original pause through any number of narrowings has
    # next == ("rank_jobs",); walking oldest-to-newest (history iterates
    # newest-first, so we keep overwriting) lands on the very first one —
    # the shortlist as it was right after filter_jobs, before any /chat call.
    original_snapshot = None
    for snapshot in graph_app.get_state_history(config):
        if snapshot.next == ("rank_jobs",):
            original_snapshot = snapshot
    if original_snapshot is None:
        raise HTTPException(404, "No paused search found for this thread_id")

    original_jobs = original_snapshot.values["filtered_jobs"]
    # Restoring it via update_state writes a fresh checkpoint with the old
    # values — genuine checkpoint "rewind", not just a read.
    graph_app.update_state(config, {"filtered_jobs": original_jobs})
    return SearchStartResponse(thread_id=thread_id, filtered_jobs=original_jobs)


@app.post("/search/{thread_id}/continue")
def continue_search(thread_id: str) -> list[RankedJob]:
    print(f"===== RESUMING GRAPH, thread_id={thread_id} ======")
    config = {"configurable": {"thread_id": thread_id}}
    # Passing None as input tells LangGraph to resume from the last checkpoint
    # for this thread_id instead of starting a new run.
    final_state = graph_app.invoke(None, config=config)
    return final_state["ranked_jobs"]
