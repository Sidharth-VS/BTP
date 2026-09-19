from pathlib import Path
from typing import Any, Dict
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> "AppSettings":
        path = Path(yaml_path)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**data)