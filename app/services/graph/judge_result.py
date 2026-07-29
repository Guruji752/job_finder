from app.config import settings
from app.llm.client import get_llm_client
from app.llm.json_parse import parse_json_response
from app.profile.digest import ProfileDigest
from app.services.graph.stategraph import JobSearchState

SUPERVISOR_PROMPT = """You are a Supervisor deciding whether current job search results are good enough.

CANDIDATE: {skills} | {seniority} | {domains}

RESULTS SO FAR ({job_count} jobs):
{results_summary}

dummy_used: {dummy_used}   retry_count: {retry_count}  is_done:{is_done}

Decide ONE of:
- "retrieve_profile" — results are weak, search again
- "dummy_agent" — search exhausted, use fallback (only if dummy_used is false)
- "end" — results are good enough, stop

As a rough guide: aim for at least 4-5 solid matches (match_score roughly 60+)
before considering the search complete. Use judgment on quality vs quantity —
2 excellent matches can be better than 8 mediocre ones, but 1 mediocre match
alone is not enough.

Return ONLY a JSON object: {{"decision": "<one of the above>"}}"""

def judge_results(
        profile:ProfileDigest,
        job:JobSearchState
)->str:
  """Ask the LLM for a judgment which step will be better option"""  
  print("==== Supervisor Judgmenet ====")

  ranked_jobs = job['ranked_jobs']
  ranked_jobs_count = len(ranked_jobs)
  skills = ", ".join(profile.skills)
  seniority=profile.seniority
  domains=", ".join(profile.domains)
  dummy_used = job.get('dummy_used', False)
  is_done = job.get('is_done', False)
  results_summary = "\n".join(
    f"{r.job.title}: score={r.analysis.match_score}, verdict={r.analysis.verdict}"
    for r in ranked_jobs
    )
  prompt = SUPERVISOR_PROMPT.format(
    job_count = ranked_jobs_count,
    skills = skills,
    seniority = seniority,
    domains = domains,
    dummy_used = dummy_used,
    retry_count = job.get('retry_count', 0),
    is_done = is_done,
    results_summary = results_summary
  )
  
  completion = get_llm_client().chat.completions.create(
    model=settings.chat_model,
    messages=[{"role":"user","content":prompt}],
  )

  raw = completion.choices[0].message.content
  parsed = parse_json_response(raw)
  return parsed.get("decision") or "end"


  
  
  
  