from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from app.config import settings
from app.job_sources.base import Job
from app.llm.client import get_llm_client
from app.llm.json_parse import parse_json_array, parse_json_response
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

GAP_PROMPT_BATCH = """You are a careful technical recruiter. Compare this candidate against EACH of the following jobs and return a JSON array of assessments, one object per job, in the same order as the jobs are listed below.

CANDIDATE:
- Skills: {skills}
- Seniority: {seniority}
- Domains: {domains}

JOBS:
{jobs_block}

Return a JSON array with exactly {n} objects (same order as the jobs above). Each object must have exactly these keys:
- "match_score": integer 0-100 (how well the candidate fits this specific role)
- "matched_skills": list of strings (skills the candidate has that this job requires)
- "missing_skills": list of strings (skills this job requires that the candidate lacks)
- "verdict": string (one honest sentence — do NOT inflate; if it's a weak fit, say so)

Be conservative: only count a skill as matched if the job actually asks for it. Respond with ONLY the JSON array, no markdown fences, no other text."""

# Job descriptions can be long; cap to keep the prompt focused and cheaper.
_MAX_DESC_CHARS = 2000

# Batches larger than this start silently dropping jobs and losing skill-tag
# precision (confirmed empirically: batch of 5 was clean, batch of 10 dropped
# a job and started copying raw job-description phrases instead of skill tags).
_BATCH_SIZE = 5


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


def _chunk(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def analyze_batch(jobs: list[Job], profile: ProfileDigest) -> list[GapAnalysis]:
    jobs_block = "\n\n".join(
        f"Job {i + 1}:\nTitle: {job.title}\nCompany: {job.company}\nDescription: {job.description[:_MAX_DESC_CHARS]}"
        for i, job in enumerate(jobs)
    )
    prompt = GAP_PROMPT_BATCH.format(
        skills=", ".join(profile.skills),
        seniority=profile.seniority,
        domains=", ".join(profile.domains),
        jobs_block=jobs_block,
        n=len(jobs),
    )
    completion = get_llm_client().chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = completion.choices[0].message.content
    parsed = parse_json_array(raw)

    if len(parsed) != len(jobs):
        # Model silently dropped/added an entry — don't trust the alignment
        # between jobs and results, fall back to per-job analysis instead.
        raise ValueError(f"batch returned {len(parsed)} results for {len(jobs)} jobs")

    return [GapAnalysis.model_validate(item) for item in parsed]


def _analyze_chunk(jobs: list[Job], profile: ProfileDigest) -> list[GapAnalysis]:
    try:
        return analyze_batch(jobs, profile)
    except Exception as e:
        # Batch call failed or came back misaligned — fall back to analyzing
        # this chunk's jobs one at a time instead of losing them entirely.
        print(f"=== RANK_JOBS: batch of {len(jobs)} failed ({e}), falling back to per-job analysis ===")
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            return list(executor.map(lambda job: analyze_job(job, profile), jobs))


def rank_jobs(jobs: list[Job], profile: ProfileDigest) -> list[RankedJob]:
    if not jobs:
        return []

    chunks = _chunk(jobs, _BATCH_SIZE)
    print(f"=== RANK_JOBS: {len(jobs)} jobs split into {len(chunks)} batch(es) of up to {_BATCH_SIZE} ===")

    # Batches are independent LLM calls — run them concurrently, same as
    # before, just at batch granularity instead of one call per job.
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        chunk_results = list(executor.map(lambda chunk: _analyze_chunk(chunk, profile), chunks))

    ranked = [
        RankedJob(job=job, analysis=analysis)
        for chunk, analyses in zip(chunks, chunk_results)
        for job, analysis in zip(chunk, analyses)
    ]
    ranked.sort(key=lambda r: r.analysis.match_score, reverse=True)
    return ranked
