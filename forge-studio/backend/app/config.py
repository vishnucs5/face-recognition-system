from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Server
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    FRONTEND_URL: str = "http://localhost:5173"

    # AI Services
    COMFYUI_URL: str = "http://127.0.0.1:8188"
    OLLAMA_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen3-coder:30b"

    # Modes
    DEMO_MODE: bool = False

    # Directories
    DATA_DIR: str = "./data"
    MODELS_DIR: str = "./models"
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/database/forge.db"

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @property
    def data_path(self) -> Path:
        path = Path(self.DATA_DIR)
        return path if path.is_absolute() else self.base_dir / path

    @property
    def generated_path(self) -> Path:
        path = self.data_path / "generated"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def uploads_path(self) -> Path:
        path = self.data_path / "uploads"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def database_path(self) -> Path:
        path = self.data_path / "database"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def resolved_database_url(self) -> str:
        if self.DATABASE_URL.startswith("sqlite+aiosqlite:///"):
            prefix = "sqlite+aiosqlite:///"
            db_rel = self.DATABASE_URL[len(prefix):]
            p = Path(db_rel)
            if not p.is_absolute():
                abs_p = (self.base_dir / p).resolve()
            else:
                abs_p = p.resolve()
            abs_p.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite+aiosqlite:///{abs_p.as_posix()}"
        return self.DATABASE_URL

    @property
    def models_path(self) -> Path:
        path = Path(self.MODELS_DIR)
        return path if path.is_absolute() else self.base_dir / path

    @property
    def checkpoints_path(self) -> Path:
        path = self.models_path / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def loras_path(self) -> Path:
        path = self.models_path / "loras"
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
