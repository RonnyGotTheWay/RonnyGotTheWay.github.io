from fastapi import APIRouter

from art_rank_quant.api.schemas.common import Envelope
from art_rank_quant.api.schemas.weekly import WeeklyReport
from art_rank_quant.api.services.weekly_report import get_weekly_report_service

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/weekly/latest", response_model=Envelope[WeeklyReport])
def latest_weekly_report() -> Envelope[WeeklyReport]:
    report, metadata = get_weekly_report_service().latest()
    return Envelope(meta=metadata, data=report)
