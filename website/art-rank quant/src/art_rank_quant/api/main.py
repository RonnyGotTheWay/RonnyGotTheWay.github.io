from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from art_rank_quant import __version__
from art_rank_quant.api.routers import backtest, market, models, portfolio, reports, stocks, style, system

app = FastAPI(title="ART-Rank Quant API", version=__version__, docs_url="/docs", redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

for router in (
    market.router,
    style.router,
    stocks.router,
    portfolio.router,
    backtest.router,
    models.router,
    system.router,
    reports.router,
):
    app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "version": __version__, "mode": "research-only"}
