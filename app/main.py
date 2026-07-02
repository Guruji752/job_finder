from fastapi import FastAPI
from pydantic import BaseModel

from app.job_sources.jsearch import JSearchSource
from app.matching.tier1 import rank_by_similarity
from app.matching.tier2 import RankedJob, rank_jobs
from app.profile.digest import get_profile_digest

app = FastAPI(title="Job Finder")

job_source = JSearchSource()


class SearchRequest(BaseModel):
    query: str
    location: str | None = None
    date_posted: str = "week"
    num_pages: int = 2
    # How many jobs get the full LLM gap analysis. None = all of them (accurate
    # missing_skills + score for every job, but one LLM call each — slower/pricier).
    # Set an int to cap it (cheaper): capped jobs are still returned, unanalyzed.
    analyze_top: int | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search")
def search(request: SearchRequest) -> list[RankedJob]:
    # 1. Fetch raw jobs from the job source.
    jobs = job_source.search(
        request.query,
        request.location,
        date_posted=request.date_posted,
        num_pages=request.num_pages,
    )

    # 2. Profile digest (cached across requests — describes the candidate, not the search).
    profile = get_profile_digest()

    # 3. Tier 1: rank ALL jobs by cheap local embedding similarity (no truncation).
    ranked = rank_by_similarity(jobs, profile)

    # 4. Tier 2: LLM gap analysis. By default on every job (analyze_top=None), so
    #    all are ranked by match_score with missing_skills. An int caps analysis to
    #    the top-N most-similar; the rest are still returned, unanalyzed.
    n = request.analyze_top if request.analyze_top is not None else len(ranked)
    analyzed = rank_jobs(ranked[:n], profile)
    rest = [RankedJob(job=job) for job in ranked[n:]]
    return analyzed + rest
