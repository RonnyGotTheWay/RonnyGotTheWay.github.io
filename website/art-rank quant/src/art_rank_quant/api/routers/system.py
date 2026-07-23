from fastapi import APIRouter, HTTPException

from art_rank_quant.api.schemas.common import Envelope
from art_rank_quant.api.schemas.market import RefreshRun
from art_rank_quant.api.services.research_runtime import get_research_runtime_service
from art_rank_quant.common.config import get_settings

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/status", response_model=Envelope[dict[str, str]])
def status() -> Envelope[dict[str, str]]:
    settings = get_settings()
    provider = settings.market_data_provider.lower()
    publication = (
        "licensed_personal_research"
        if provider != "eastmoney" or settings.eastmoney_permission_confirmed
        else "configuration_required"
    )
    return Envelope(
        data={
            "api": "healthy",
            "database": "local",
            "data": provider,
            "publication": publication,
        }
    )


@router.get("/runs/{run_id}", response_model=RefreshRun)
def run_status(run_id: str) -> RefreshRun:
    if not run_id.startswith("refresh-") or "/" in run_id or ".." in run_id:
        raise HTTPException(status_code=422, detail="Invalid refresh run id")
    try:
        return get_research_runtime_service().get_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Refresh run not found") from exc
