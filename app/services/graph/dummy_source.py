"""
Dummy Agent's data — a pre-analyzed, RankedJob-shaped batch loaded from disk
instead of hitting JSearch + the LLM. Used by the Supervisor as a fallback when
the real Search Agent comes back with too few jobs, without spending on a second
live source (e.g. Tavily).
"""
import json

from app.matching.tier2 import RankedJob

DUMMY_JOBS_PATH = "dummy/jobs.json"


def load_dummy_jobs() -> list[RankedJob]:
    with open(DUMMY_JOBS_PATH) as f:
        raw = json.load(f)
    return [RankedJob.model_validate(entry) for entry in raw]
