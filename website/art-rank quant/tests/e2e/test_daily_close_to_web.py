from fastapi.testclient import TestClient

from art_rank_quant.api.main import app


def test_research_site_source_contract_is_read_only() -> None:
    client = TestClient(app)
    assert client.get("/health").json()["mode"] == "research-only"
    assert client.post("/api/v1/stocks/rankings").status_code == 405
