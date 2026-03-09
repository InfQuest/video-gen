from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

# Get the project root directory (where secrets.yaml is located)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SECRETS_FILE = PROJECT_ROOT / "secrets.yaml"


class Settings(BaseModel):
    openai_api_key: str = ""
    volcengine_api_key: Optional[str] = None

    # AWS credentials for S3
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None


def load_settings() -> Settings:
    """Load settings from secrets.yaml file.

    Returns:
        Settings: Loaded settings

    Raises:
        FileNotFoundError: If secrets.yaml does not exist
        yaml.YAMLError: If secrets.yaml is invalid
    """
    if not SECRETS_FILE.exists():
        raise FileNotFoundError(
            f"secrets.yaml not found at {SECRETS_FILE}. "
            "Please create a secrets.yaml file in the project root."
        )

    with open(SECRETS_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError("secrets.yaml is empty")

    # Flatten nested structure if needed
    # Supports both flat structure and nested structure like:
    # api_keys:
    #   openai_api_key: xxx
    # aws:
    #   aws_access_key_id: xxx
    flat_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat_data.update(value)
        else:
            flat_data[key] = value

    return Settings(**flat_data)


settings = load_settings()
