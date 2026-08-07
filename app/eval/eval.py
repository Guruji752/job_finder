import sys
import types
_dummy_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
_dummy_vertexai.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = _dummy_vertexai
import asyncio
from ragas import SingleTurnSample
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_huggingface import HuggingFaceEndpoint
from app.llm.client import get_llm_client
from app.rag.client import RAGClient
from app.config import settings
from app.profile.digest import QUESTIONS


llm = LangchainLLMWrapper(
    HuggingFaceEndpoint(
        repo_id=settings.chat_model,
        huggingfacehub_api_token=settings.hf_token
    )
)

faithfulness = Faithfulness(llm=llm)

sample = SingleTurnSample(
    user_input="Find AI Engineer jobs in Delhi",
    retrieved_contexts=[
        "Senior AI Engineer at Meesho Delhi. Requires Python and LangGraph.",
        "ML Engineer at Sarvam Delhi. Requires RAG and Python.",
    ],
    response="Meesho is hiring Senior AI Engineer in Delhi.",
)

async def run():
    score = await faithfulness.single_turn_ascore(sample)
    print(f"Faithfulness: {score:.4f}")

asyncio.run(run())



