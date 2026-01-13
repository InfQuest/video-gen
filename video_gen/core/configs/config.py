from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the project root directory (where .env is located)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        populate_by_name=True,
        alias_generator=lambda field_name: field_name.upper(),
        extra="ignore",
    )

    openai_api_key: str = ""
    volcengine_api_key: Optional[str] = None

    # AWS credentials for S3
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None


settings = Settings()
