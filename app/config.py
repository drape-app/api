from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    supabase_url: str
    supabase_service_key: str
    supabase_jwt_secret: str
    gemini_api_key: str
    cloudinary_url: str
    redis_url: str = "redis://redis:6379/0"
    modal_token_id: str = ""
    modal_token_secret: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
