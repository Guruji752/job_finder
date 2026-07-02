from pydantic import BaseModel

from app.config import settings
from app.llm.client import get_llm_client
from app.llm.json_parse import parse_json_response
from app.rag.client import RAGClient

QUESTIONS = [
    "What programming languages, frameworks, and technical skills does the candidate have?",
    "What is the candidate's total years of professional work experience, and what "
    "seniority level (junior/mid/senior/lead) does that represent?",
    "What industries or business domains has the candidate worked in?",
    "What job titles has the candidate held throughout their career, in order?",
]

DISTILL_PROMPT = """Based on the following Q&A about a candidate's resume, produce a single JSON object with exactly these keys:
- "skills": list of strings (programming languages, frameworks, tools)
- "years_experience": number (total years of professional experience)
- "seniority": string (e.g. "junior", "mid", "senior", "lead") — base this primarily
  on the candidate's most recent / highest job title, not just computed years. A
  title containing "Senior" or "Lead" means the seniority should reflect that,
  even if the computed years alone might suggest otherwise.
- "domains": list of strings (industries/business domains worked in)

Respond with ONLY the JSON object, no markdown code fences, no other text.

{context}"""


class ProfileDigest(BaseModel):
    skills: list[str]
    years_experience: float
    seniority: str
    domains: list[str]


def build_profile_digest(rag_client: RAGClient | None = None) -> ProfileDigest:
    rag_client = rag_client or RAGClient()

    qa_pairs = []
    for question in QUESTIONS:
        response = rag_client.query(question)
        qa_pairs.append(f"Q: {question}\nA: {response.answer}")

    context = "\n\n".join(qa_pairs)

    llm_client = get_llm_client()
    completion = llm_client.chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": DISTILL_PROMPT.format(context=context)}],
    )

    raw = completion.choices[0].message.content
    return ProfileDigest.model_validate(parse_json_response(raw))


_cached_digest: ProfileDigest | None = None


def get_profile_digest(force_refresh: bool = False) -> ProfileDigest:
    """Return the candidate's profile digest, building it once and caching it.

    The digest describes the candidate (not the search), so it's reused across
    requests — avoids re-hitting RAG + the LLM on every /search call. Call with
    force_refresh=True after the resume changes.
    """
    global _cached_digest
    if _cached_digest is None or force_refresh:
        _cached_digest = build_profile_digest()
    return _cached_digest
