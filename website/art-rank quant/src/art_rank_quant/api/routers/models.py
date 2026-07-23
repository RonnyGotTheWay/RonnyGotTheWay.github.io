from fastapi import APIRouter, HTTPException, status

from art_rank_quant.api.schemas.common import Envelope
from art_rank_quant.api.schemas.models import BootstrapAccepted, ModelMonitoring, TrainingRunSummary
from art_rank_quant.api.services.model_bootstrap import (
    TrainingConflictError,
    get_model_bootstrap_service,
    get_short_term_bootstrap_service,
)
from art_rank_quant.api.services.research_runtime import get_research_runtime_service
from art_rank_quant.api.services.short_term import get_short_term_runtime_service

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/health", response_model=Envelope[ModelMonitoring])
def health() -> Envelope[ModelMonitoring]:
    service = get_research_runtime_service()
    monitoring = service.monitoring()
    _, metadata = service.latest_prediction()
    return Envelope(meta=metadata, data=monitoring)


@router.post("/bootstrap", response_model=BootstrapAccepted, status_code=status.HTTP_202_ACCEPTED)
def bootstrap_models() -> BootstrapAccepted:
    try:
        return get_model_bootstrap_service().start()
    except TrainingConflictError as exc:
        raise HTTPException(status_code=409, detail=exc.detail) from exc


@router.get("/runs/{run_id}", response_model=TrainingRunSummary)
def bootstrap_run(run_id: str) -> TrainingRunSummary:
    try:
        return get_model_bootstrap_service().get(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Bootstrap run not found") from exc


@router.post("/runs/{run_id}/cancel", response_model=TrainingRunSummary)
def cancel_bootstrap_run(run_id: str) -> TrainingRunSummary:
    try:
        return get_model_bootstrap_service().cancel(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Bootstrap run not found") from exc


@router.post("/short-term/bootstrap", response_model=BootstrapAccepted, status_code=status.HTTP_202_ACCEPTED)
def bootstrap_short_term_models() -> BootstrapAccepted:
    try:
        return get_short_term_bootstrap_service().start()
    except TrainingConflictError as exc:
        raise HTTPException(status_code=409, detail=exc.detail) from exc


@router.get("/short-term/runs/{run_id}", response_model=TrainingRunSummary)
def short_term_run(run_id: str) -> TrainingRunSummary:
    try:
        return get_short_term_bootstrap_service().get(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Short-term run not found") from exc


@router.post("/short-term/runs/{run_id}/cancel", response_model=TrainingRunSummary)
def cancel_short_term_run(run_id: str) -> TrainingRunSummary:
    try:
        return get_short_term_bootstrap_service().cancel(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Short-term run not found") from exc


@router.get("/short-term/health", response_model=Envelope[ModelMonitoring])
def short_term_health() -> Envelope[ModelMonitoring]:
    service = get_short_term_runtime_service()
    monitoring = service.monitoring()
    _, metadata = service.latest_prediction()
    return Envelope(meta=metadata, data=monitoring)
