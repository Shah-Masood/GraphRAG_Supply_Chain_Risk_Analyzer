from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
 
 
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
 
    # ── PostgreSQL ──────────────────────────────────────────────────────────────
    database_url: str
    db_pool_min_size: int = 2
    db_pool_max_size: int = 10
 
    # ── ChromaDB ────────────────────────────────────────────────────────────────
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "supply_chain_docs"
 
    # ── Neo4j ───────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j_password"
 
    # ── OpenAI ──────────────────────────────────────────────────────────────────
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o"
 
    # ── NewsAPI ─────────────────────────────────────────────────────────────────
    news_api_key: str
    news_api_base_url: str = "https://newsapi.org/v2"
 
    # ── App ─────────────────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
 
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"
 
 
@lru_cache
def get_settings() -> Settings:
    """Cached singleton — import and call this everywhere."""
    return Settings()