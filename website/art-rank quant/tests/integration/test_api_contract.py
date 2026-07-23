from fastapi.testclient import TestClient

from art_rank_quant.api.main import app
from art_rank_quant.api.schemas.common import VersionMetadata
from art_rank_quant.api.schemas.models import TrainingRunSummary
from art_rank_quant.api.schemas.weekly import WeeklyReport
from art_rank_quant.api.services.model_bootstrap import TrainingConflictError


def test_versioned_rankings_contract() -> None:
    response = TestClient(app).get("/api/v1/stocks/rankings?limit=3")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["data"]) == 3
    assert {"data_version", "feature_version", "model_version", "run_id", "is_stale"} <= set(payload["meta"])


def test_realtime_endpoint_rejects_invalid_symbols_without_scraping() -> None:
    response = TestClient(app).get("/api/v1/market/quotes?symbols=https://example.com/quote")
    assert response.status_code == 422


def test_short_term_health_and_prediction_contracts_are_isolated() -> None:
    client = TestClient(app)

    health = client.get("/api/v1/models/short-term/health")
    prediction = client.get("/api/v1/stocks/predictions/short-term/latest")

    assert health.status_code == 200
    assert health.json()["data"]["mode"] == "short_term"
    assert health.json()["data"]["experimental"] is True
    assert prediction.status_code == 200
    assert prediction.json()["data"]["prediction_mode"] == "short_term"
    assert prediction.json()["data"]["training_window_days"] == 15
    assert prediction.json()["data"]["experimental"] is True


def test_short_term_bootstrap_returns_structured_conflict_for_long_run(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class ConflictingService:
        def start(self):  # type: ignore[no-untyped-def]
            raise TrainingConflictError(
                TrainingRunSummary(run_id="bootstrap-active", status="running", mode="long_term")
            )

    monkeypatch.setattr(
        "art_rank_quant.api.routers.models.get_short_term_bootstrap_service",
        lambda: ConflictingService(),
    )
    response = TestClient(app).post("/api/v1/models/short-term/bootstrap")

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "long_term_training_active"
    assert response.json()["detail"]["blocking_run_id"] == "bootstrap-active"


def test_weekly_report_contract_when_not_available(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class MissingWeeklyService:
        def latest(self):  # type: ignore[no-untyped-def]
            report = WeeklyReport(
                status="not_available", message="missing", generated_at="2026-07-22T00:00:00Z"
            )
            return report, VersionMetadata(is_stale=True)

    monkeypatch.setattr("art_rank_quant.api.routers.reports.get_weekly_report_service", MissingWeeklyService)
    response = TestClient(app).get("/api/v1/reports/weekly/latest")

    assert response.status_code == 200
    assert response.json()["data"]["status"] == "not_available"
    assert response.json()["meta"]["is_stale"] is True
