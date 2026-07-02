from openai import OpenAI

from app.config import settings


def get_llm_client() -> OpenAI:
    return OpenAI(base_url=settings.hf_base_url, api_key=settings.hf_token)
