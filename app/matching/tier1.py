import numpy as np
from huggingface_hub import InferenceClient

from app.config import settings
from app.job_sources.base import Job
from app.profile.digest import ProfileDigest

# BAAI/bge models are trained to expect this instruction prefix on the *query*
# side of a retrieval pair (the profile) — not on the documents being searched
# (job postings), which are embedded as-is.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# feature_extraction can only take so much text per call; matches the RAG service.
_MAX_CHARS = 1500

_client = InferenceClient(token=settings.hf_token)


def _embed_one(text: str) -> np.ndarray:
    # BGE via feature_extraction returns token-level embeddings (seq_len, dim);
    # mean-pool across tokens to get a single sentence vector — same as the RAG service.
    result = _client.feature_extraction(text[:_MAX_CHARS], model=settings.embedding_model)
    arr = np.array(result)
    if arr.ndim == 1:
        return arr
    elif arr.ndim == 2:
        return arr.mean(axis=0)
    else:
        return arr[0].mean(axis=0)


def _profile_text(profile: ProfileDigest) -> str:
    return (
        f"Skills: {', '.join(profile.skills)}. "
        f"Domains: {', '.join(profile.domains)}. "
        f"Seniority: {profile.seniority}."
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def rank_by_similarity(jobs: list[Job], profile: ProfileDigest) -> list[Job]:
    """All jobs, sorted most-similar-first. No truncation."""
    profile_embedding = _embed_one(QUERY_INSTRUCTION + _profile_text(profile))
    job_embeddings = [_embed_one(f"{job.title}. {job.description}") for job in jobs]

    scored_jobs = sorted(
        zip(jobs, (_cosine_similarity(profile_embedding, je) for je in job_embeddings)),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [job for job, _ in scored_jobs]


def filter_by_similarity(jobs: list[Job], profile: ProfileDigest, top_n: int = 15) -> list[Job]:
    return rank_by_similarity(jobs, profile)[:top_n]
