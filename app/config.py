from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Job source (JSearch / OpenWeb Ninja) ---
    jsearch_api_key: str = ""
    jsearch_country: str = "in"

    # --- Job sources (Exa) ---
    exa_api_key: str = ""

    # --- RAG service (existing, external) ---
    rag_base_url: str = "http://localhost:8001"

    # --- Hugging Face Inference Providers ---
    hf_token: str = ""
    hf_base_url: str = "https://router.huggingface.co/v1"

    # --- Models ---
    chat_model: str = "Qwen/Qwen2.5-72B-Instruct"
    embedding_model: str = "BAAI/bge-large-en-v1.5"


settings = Settings()
