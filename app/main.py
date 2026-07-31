from fastapi import FastAPI
from pydantic import BaseModel

from app.matching.tier2 import RankedJob
from app.services.graph.edges import graph_app

app = FastAPI(title="Job Finder")


class SearchRequest(BaseModel):
    query: str
    location: str | None = None
    num_pages: int = 2
    date_posted: str = "week"  # one of: today, 3days, week, month, all


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search")
def search(request: SearchRequest) -> list[RankedJob]:
    print("===== GRAPH INVOKE ======")
    # The endpoint no longer runs the pipeline by hand — it hands the query to the
    # LangGraph agent, which does: retrieve_profile → retrieve_jobs → filter_jobs
    # → (retry search if too few) → rank_jobs, then leaves the final list in state.
    final_state = graph_app.invoke(
        {
            "query": request.query,
            "location": request.location,
            "num_pages": request.num_pages,
            "date_posted": request.date_posted,
            "retry_count": 0,
        }
    )
    return final_state["ranked_jobs"]
