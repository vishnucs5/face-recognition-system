from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from app.config import settings

from sqlalchemy import text

connect_args = {}
if "sqlite" in settings.resolved_database_url:
    connect_args = {"timeout": 30.0}

engine = create_async_engine(
    settings.resolved_database_url,
    echo=False,
    future=True,
    connect_args=connect_args,
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()


async def init_db() -> None:
    import app.models  # noqa: F401 - Register models with Base.metadata
    settings.database_path.mkdir(parents=True, exist_ok=True)
    async with engine.begin() as conn:
        if "sqlite" in settings.resolved_database_url:
            await conn.execute(text("PRAGMA journal_mode=WAL;"))
            await conn.execute(text("PRAGMA busy_timeout=30000;"))
        await conn.run_sync(Base.metadata.create_all)
        if "sqlite" in settings.resolved_database_url:
            try:
                await conn.execute(text("ALTER TABLE generations ADD COLUMN comfy_prompt_id VARCHAR(100);"))
            except Exception:
                pass
            try:
                await conn.execute(text("ALTER TABLE generations ADD COLUMN completed_at TIMESTAMP;"))
            except Exception:
                pass
            for col, col_type in [
                ("mask_image_path", "VARCHAR(500)"),
                ("controlnet_type", "VARCHAR(50)"),
                ("controlnet_model", "VARCHAR(255)"),
                ("controlnet_strength", "FLOAT"),
                ("controlnet_start_percent", "FLOAT"),
                ("controlnet_end_percent", "FLOAT"),
                ("controlnet_preprocessor", "VARCHAR(50)"),
                ("started_at", "TIMESTAMP"),
                ("priority", "INTEGER DEFAULT 2"),
                ("estimated_vram_mb", "INTEGER"),
                ("estimated_duration_sec", "FLOAT"),
                ("client_request_id", "VARCHAR(36)"),
                ("generation_hash", "VARCHAR(64)"),
                ("retry_count", "INTEGER DEFAULT 0"),
                ("preset_name", "VARCHAR(100)"),
                ("preset_version", "INTEGER"),
                ("settings_snapshot_json", "TEXT"),
                ("gpu_telemetry_snapshot_json", "TEXT"),
                ("workflow_graph_hash", "VARCHAR(64)"),
                ("asset_hashes_json", "TEXT"),
            ]:
                try:
                    await conn.execute(text(f"ALTER TABLE generations ADD COLUMN {col} {col_type};"))
                except Exception:
                    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session
