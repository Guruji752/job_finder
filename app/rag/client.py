import requests
from pydantic import BaseModel

from app.config import settings


class RAGResponse(BaseModel):
    answer: str
    sources: list[str]


class RAGClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or settings.rag_base_url

    def query(self, question: str, top_k: int = 5) -> RAGResponse:
        response = requests.post(
            f"{self.base_url}/query",
            json={"question": question, "top_k": top_k},
        )
        response.raise_for_status()
        data = response.json()
        return RAGResponse(answer=data["answer"], sources=data["sources"])
