from typing import Literal
from fastapi import APIRouter, Body, Query, Response

from app.schemas.diagnostics import (
    BenchmarkResult,
    BenchmarkRunRequest,
    DiagnosticsReport,
)
from app.services.benchmark_service import benchmark_service
from app.services.diagnostics_service import diagnostics_service

router = APIRouter(prefix="/diagnostics", tags=["Diagnostics"])


@router.get("/system", response_model=DiagnosticsReport)
async def get_system_diagnostics(refresh: bool = False) -> DiagnosticsReport:
    """
    Run or fetch cached 9-point system hardware & dependency diagnostics.
    """
    return await diagnostics_service.run_diagnostics(force_refresh=refresh)


@router.post("/benchmark/start", response_model=BenchmarkResult)
async def start_benchmark(request: BenchmarkRunRequest = Body(default_factory=BenchmarkRunRequest)) -> BenchmarkResult:
    """
    Execute hardware benchmark suite with GPU safety isolation lease.
    """
    return await benchmark_service.run_benchmark(request)


@router.get("/benchmark/results", response_model=list[BenchmarkResult])
async def list_benchmark_results() -> list[BenchmarkResult]:
    """
    List all stored benchmark execution runs.
    """
    return benchmark_service.get_results()


@router.get("/benchmark/export")
async def export_benchmark_results(format: Literal["json", "csv"] = Query("json")):
    """
    Export benchmark results as a downloadable JSON or CSV file.
    """
    content = benchmark_service.export_results(format=format)
    if format.lower() == "csv":
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="benchmark_results.csv"'},
        )
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="benchmark_results.json"'},
    )
