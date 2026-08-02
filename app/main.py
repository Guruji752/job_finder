import uuid

from fastapi import FastAPI
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
    current_jobs = graph_app.get_state(config).values["filtered_jobs"]
    narrowed = filter_shortlist(current_jobs, request.message)
    # Patch the checkpoint directly. This does NOT run any graph nodes — the
    # graph stays paused at the same interrupt point (before rank_jobs), so
    # /chat can be called repeatedly to keep narrowing before ever resuming.
    graph_app.update_state(config, {"filtered_jobs": narrowed})
    return SearchStartResponse(thread_id=request.thread_id, filtered_jobs=narrowed)


@app.post("/search/{thread_id}/continue")
def continue_search(thread_id: str) -> list[RankedJob]:
    print(f"===== RESUMING GRAPH, thread_id={thread_id} ======")
    config = {"configurable": {"thread_id": thread_id}}
    # Passing None as input tells LangGraph to resume from the last checkpoint
    # for this thread_id instead of starting a new run.
    final_state = graph_app.invoke(None, config=config)
    return final_state["ranked_jobs"]
