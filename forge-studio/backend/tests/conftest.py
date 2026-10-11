import pytest_asyncio
from app.database import init_db


_db_initialized = False

@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    global _db_initialized
    if not _db_initialized:
        await init_db()
        _db_initialized = True
