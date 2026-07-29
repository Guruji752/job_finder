from pydantic import BaseModel

from app.config import settings
from app.job_sources.base import Job
from app.llm.client import get_llm_client
from app.llm.json_parse import parse_json_response
from app.profile.digest import ProfileDigest

GAP_PROMPT = """You are a careful technical recruiter. Compare this candidate against the job below and return a JSON assessment.

CANDIDATE:
- Skills: {skills}
- Seniority: {seniority}
- Domains: {domains}

JOB:
Title: {title}
Company: {company}
Description: {description}

Return a single JSON object with exactly these keys:
- "match_score": integer 0-100 (how well the candidate fits this specific role)
- "matched_skills": list of strings (skills the candidate has that this job requires)
- "missing_skills": list of strings (skills this job requires that the candidate lacks)
- "verdict": string (one honest sentence — do NOT inflate; if it's a weak fit, say so)

Be conservative: only count a skill as matched if the job actually asks for it. Respond with ONLY the JSON object, no markdown fences, no other text."""

# Job descriptions can be long; cap to keep the prompt focused and cheaper.
_MAX_DESC_CHARS = 2000


class GapAnalysis(BaseModel):
    match_score: int
    matched_skills: list[str]
    missing_skills: list[str]
    verdict: str


class RankedJob(BaseModel):
    job: Job
    analysis: GapAnalysis | None = None  # None = listed but not deep-analyzed (Tier-2 skipped)


def analyze_job(job: Job, profile: ProfileDigest) -> GapAnalysis:
    prompt = GAP_PROMPT.format(
        skills=", ".join(profile.skills),
        seniority=profile.seniority,
        domains=", ".join(profile.domains),
        title=job.title,
        company=job.company,
        description=job.description[:_MAX_DESC_CHARS],
    )
    completion = get_llm_client().chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = completion.choices[0].message.content
    return GapAnalysis.model_validate(parse_json_response(raw))


def rank_jobs(jobs: list[Job], profile: ProfileDigest) -> list[RankedJob]:
    ranked = [RankedJob(job=job, analysis=analyze_job(job, profile)) for job in jobs]
    ranked.sort(key=lambda r: r.analysis.match_score, reverse=True)
    return ranked
