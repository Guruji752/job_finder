"""
RAGAS evaluation of the profile-digest RAG step — scores retrieve_profile's
Q&A (RAGClient answering QUESTIONS from digest.py) for quality.

Faithfulness and AnswerRelevancy are reference-free — they run automatically
against real RAG output, no ground truth needed. ContextPrecision and
ContextRecall both need a ground-truth reference answer per question; fill
in REFERENCE_ANSWERS below with the actual correct answer from your real
resume. Any question left out of that dict just skips both for it.

Standalone script, not wired into the running app:
    uv run python -m app.eval.ragas_eval
"""
import sys
import types

# ragas (as of 0.4.3) unconditionally imports ChatVertexAI from a legacy
# langchain-community shim that no longer ships in the installed version —
# we never use Vertex AI, so stub the module just to satisfy the import.
_dummy_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
_dummy_vertexai.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = _dummy_vertexai

from openai import AsyncOpenAI
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from app.config import settings
from app.matching.tier1 import _embed_one
from app.profile.digest import QUESTIONS
from app.rag.client import RAGClient

# Ground truth, written from the actual resume (Resume2026.pdf).
REFERENCE_ANSWERS: dict[str, str] = {
    "What programming languages, frameworks, and technical skills does the candidate have?": (
        "Languages: Python, JavaScript, SQL, Bash, Rust, C++, Perl. "
        "Frameworks: FastAPI, Django, DRF, gRPC, React, Flask, Pandas, Scikit-learn. "
        "DevOps Tools: Kubernetes, Docker, Git, Jenkins, CI/CD, Nginx. "
        "AI/ML: LightFM, CrewAI, LLM Integration, RAG, MCP, Multi-Agent Systems, Prompt Engineering. "
        "Database: MySQL, PostgreSQL, DynamoDB, MongoDB, VectorDB, ElasticSearch. "
        "Core Technical Skills: System Design, Microservices, Distributed Systems, "
        "Event-Driven Architecture, Scalability, Caching, Message Queues. "
        "Cloud Services (AWS): Lambda, Step Functions, S3, Glue, SES, SNS, EventBridge, "
        "API Gateway, Redshift, EC2."
    ),
    "What is the candidate's total years of professional work experience, and what "
    "seniority level (junior/mid/senior/lead) does that represent?": (
        "Total professional experience is approximately 6 years, spanning Tradeindia "
        "(Mar 2020 - Dec 2022), SupplyCopia (Dec 2022 - Jun 2023), Edge (Jun 2023 - Feb 2024), "
        "Infinite Computer Solution (Jun 2024 - Feb 2026), and Verticurl Pvt Ltd (Feb 2026 - "
        "present). The candidate's current title, stated explicitly in the resume header, is "
        "'Senior Software Engineer/AI Engineer' — so the seniority level is Senior, not mid-level."
    ),
    "What industries or business domains has the candidate worked in?": (
        "OTT / Media Entertainment, Workforce Management / HR-Tech, Healthcare Supply Chain, "
        "and B2B E-Commerce / CRM / Sales Automation — these are the four domains explicitly "
        "listed under the resume's Domain Knowledge section."
    ),
    "What job titles has the candidate held throughout their career, in order?": (
        "The resume does not state a separate job title for each role. Only one title appears "
        "anywhere: 'Senior Software Engineer/AI Engineer', given once in the header as the "
        "candidate's current/overall title. Each work-experience entry (Verticurl, Infinite "
        "Computer Solution, Edge, SupplyCopia, Tradeindia) lists the company and project names "
        "only, with no distinct job title per entry."
    ),
}


class HFEmbeddings(BaseRagasEmbedding):
    """Wraps tier1.py's existing HF Inference API embedding — reused instead
    of RAGAS's built-in 'huggingface' provider, which uses local
    sentence-transformers (would download a model locally)."""

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return _embed_one(text).tolist()

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return self.embed_text(text, **kwargs)


def run_eval():
    # score() always runs the async ascore() internally (calls
    # self.llm.agenerate(), never generate()) — needs an async-capable
    # client, so this is separate from app.llm.client.get_llm_client()
    # (sync, used everywhere else in the app).
    async_client = AsyncOpenAI(base_url=settings.hf_base_url, api_key=settings.hf_token)
    llm = llm_factory(settings.chat_model, client=async_client)
    embeddings = HFEmbeddings()

    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)

    rag_client = RAGClient()
    results = []

    for question in QUESTIONS:
        response = rag_client.query(question)
        print(f"\n=== {question} ===")
        print(f"answer: {response.answer[:1000]}")

        faith_result = faithfulness.score(
            user_input=question,
            response=response.answer,
            retrieved_contexts=response.sources,
        )
        relevancy_result = answer_relevancy.score(
            user_input=question,
            response=response.answer,
        )
        print(f"faithfulness: {faith_result.value:.2f}")
        print(f"answer_relevancy: {relevancy_result.value:.2f}")

        entry = {
            "question": question,
            "faithfulness": faith_result.value,
            "answer_relevancy": relevancy_result.value,
        }

        if question in REFERENCE_ANSWERS:
            precision_result = context_precision.score(
                user_input=question,
                reference=REFERENCE_ANSWERS[question],
                retrieved_contexts=response.sources,
            )
            recall_result = context_recall.score(
                user_input=question,
                reference=REFERENCE_ANSWERS[question],
                retrieved_contexts=response.sources,
            )
            print(f"context_precision: {precision_result.value:.2f}")
            print(f"context_recall: {recall_result.value:.2f}")
            entry["context_precision"] = precision_result.value
            entry["context_recall"] = recall_result.value
        else:
            print("context_precision/context_recall: skipped (no reference answer set)")

        results.append(entry)

    print("\n=== AGGREGATE ===")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        scores = [r[metric] for r in results if metric in r]
        if scores:
            print(f"{metric}: avg={sum(scores) / len(scores):.2f} (n={len(scores)})")


if __name__ == "__main__":
    run_eval()
