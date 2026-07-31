from app.config import settings
from app.llm.client import get_llm_client
from app.llm.json_parse import parse_json_response
from app.profile.digest import ProfileDigest
from app.services.graph.stategraph import JobSearchState

SUPERVISOR_PROMPT = """You are a Supervisor deciding whether current job search results are good enough.

CANDIDATE: {skills} | {seniority} | {domains}

RESULTS SO FAR ({job_count} jobs):
{results_summary}

Best match_score this cycle: {current_max_score}
Best match_score seen across all previous cycles: {previous_max_score}

retry_count: {retry_count}  is_done:{is_done}

Decide ONE of:
- "retrieve_profile" — results are weak, search again
- "end" — results are good enough, stop

As a rough guide: aim for at least 4-5  matches.
If this cycle's best score did NOT improve on the previous best, retrying again is
unlikely to help — prefer "end" instead of "retrieve_profile".

Return ONLY a JSON object: {{"decision": "<one of the above>"}}"""


def judge_results(
        profile:ProfileDigest,
        job:JobSearchState
)-> tuple[str, int]:
  """Ask the LLM for a judgment which step will be better option.

  Returns (decision, updated_max_score) — the caller must store updated_max_score
  back into state, otherwise the next cycle has no memory of the previous best and
  can never tell whether retrying actually helped.
  """
  print("==== Supervisor Judgmenet ====")

  ranked_jobs = job['ranked_jobs']
  previous_max_score = job.get('max_score', 0)
  ranked_jobs_count = len(ranked_jobs)
  skills = ", ".join(profile.skills)
  seniority=profile.seniority
  domains=", ".join(profile.domains)
  is_done = job.get('is_done', False)
  results_summary = "\n".join(
    f"{r.job.title}: score={r.analysis.match_score}, verdict={r.analysis.verdict}"
    for r in ranked_jobs
    )
  scores_list = [r.analysis.match_score for r in ranked_jobs]
  current_max_score = max(scores_list) if scores_list else 0
  updated_max_score = max(previous_max_score, current_max_score)

  prompt = SUPERVISOR_PROMPT.format(
    job_count = ranked_jobs_count,
    skills = skills,
    seniority = seniority,
    domains = domains,
    retry_count = job.get('retry_count', 0),
    is_done = is_done,
    results_summary = results_summary,
    current_max_score = current_max_score,
    previous_max_score = previous_max_score,
  )

  completion = get_llm_client().chat.completions.create(
    model=settings.chat_model,
    messages=[{"role":"user","content":prompt}],
  )

  raw = completion.choices[0].message.content
  parsed = parse_json_response(raw)
  decision = parsed.get("decision") or "end"
  return decision, updated_max_score


  
  
  
  