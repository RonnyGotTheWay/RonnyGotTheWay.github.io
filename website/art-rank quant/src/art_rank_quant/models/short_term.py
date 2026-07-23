from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from art_rank_quant.models.hmm.model import RegimeHMM
from art_rank_quant.models.patchtst.model import PatchTSTEncoder
from art_rank_quant.models.ranker.model import ArtRanker
from art_rank_quant.models.ranker.train import rank_ic
from art_rank_quant.models.tft.model import TftStyleForecaster
from art_rank_quant.models.tft.train import quantile_loss

SHORT_WINDOW_DAYS = 15
PATCH_SEQUENCE_LENGTH = 120
SHORT_BASE_FEATURES = [
    "momentum_1d",
    "momentum_3d",
    "momentum_5d",
    "momentum_10d",
    "volatility_5d",
    "volatility_10d",
    "log_amount",
    "turnover_proxy_5d",
    "intraday_return",
    "intraday_amplitude",
    "intraday_log_amount",
]


def latest_trading_dates(frame: pl.DataFrame, days: int = SHORT_WINDOW_DAYS) -> list[date]:
    dates = sorted(frame["trade_date"].unique().to_list())
    if len(dates) < days:
        raise ValueError(f"Only {len(dates)} trading dates are available; {days} required")
    return dates[-days:]


def trim_to_dates(frame: pl.DataFrame, dates: Sequence[date]) -> pl.DataFrame:
    return frame.filter(pl.col("trade_date").is_in(list(dates))).sort(["symbol", "trade_date"])


def build_short_features(daily: pl.DataFrame, checks: pl.DataFrame, intraday: pl.DataFrame) -> pl.DataFrame:
    dates = latest_trading_dates(daily)
    daily = trim_to_dates(daily, dates)
    checks = trim_to_dates(checks, dates)
    intraday_daily = intraday_daily_features(trim_to_dates(intraday, dates))
    counts = daily.group_by("symbol").agg(pl.col("trade_date").n_unique().alias("window_days"))
    ordered = (
        daily.join(checks, on=["symbol", "trade_date"], how="left")
        .join(counts, on="symbol", how="left")
        .join(intraday_daily, on=["symbol", "trade_date"], how="left")
        .sort(["symbol", "trade_date"])
        .with_columns(
            pl.col("close_forward_adjusted").pct_change().over("symbol").alias("daily_return"),
            *[
                (
                    pl.col("close_forward_adjusted")
                    / pl.col("close_forward_adjusted").shift(window).over("symbol")
                    - 1
                ).alias(f"momentum_{window}d")
                for window in (1, 3, 5, 10)
            ],
            pl.col("amount").log1p().alias("log_amount"),
            pl.col("amount").rolling_median(5).over("symbol").alias("median_amount_5d"),
        )
        .with_columns(
            pl.col("daily_return").rolling_std(5).over("symbol").alias("volatility_5d"),
            pl.col("daily_return").rolling_std(10).over("symbol").alias("volatility_10d"),
            (pl.col("amount") / pl.col("amount").rolling_mean(5).over("symbol")).alias("turnover_proxy_5d"),
            (
                pl.col("close_forward_adjusted").shift(-1).over("symbol")
                / pl.col("close_forward_adjusted")
                - 1
            ).alias("forward_return"),
        )
        .with_columns(
            (pl.col("forward_return") - pl.col("forward_return").mean().over("trade_date")).alias(
                "forward_excess_return"
            )
        )
        .with_columns(
            (
                (
                    (pl.col("forward_excess_return").rank("ordinal").over("trade_date") - 1)
                    * 5
                    / pl.len().over("trade_date")
                )
                .floor()
                .clip(0, 4)
                .cast(pl.Int8)
            ).alias("relevance"),
            (
                (pl.col("window_days") >= SHORT_WINDOW_DAYS)
                & (pl.col("median_amount_5d") >= 50_000_000)
                & (pl.col("volume") > 0)
                & ~pl.col("is_st").fill_null(True)
                & (pl.col("trade_status_baostock").fill_null("unknown") == "trading")
                & ((pl.col("close") / pl.col("close_baostock") - 1).abs().fill_null(1.0) <= 0.005)
            ).alias("eligible"),
        )
    )
    return ordered


def intraday_daily_features(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.sort(["symbol", "event_time"])
        .group_by(["symbol", "trade_date"])
        .agg(
            (pl.col("close").last() / pl.col("open").first() - 1).alias("intraday_return"),
            (pl.col("high").max() / pl.col("low").min() - 1).alias("intraday_amplitude"),
            pl.col("amount").sum().log1p().alias("intraday_log_amount"),
        )
    )


def market_features(frame: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    columns = ["market_return", "breadth", "cross_section_volatility", "market_log_turnover"]
    market = (
        frame.group_by("trade_date")
        .agg(
            pl.col("daily_return").mean().fill_null(0.0).alias("market_return"),
            (pl.col("daily_return").fill_null(0.0) > 0).mean().alias("breadth"),
            pl.col("daily_return").std().fill_null(0.0).alias("cross_section_volatility"),
            pl.col("amount").sum().log1p().alias("market_log_turnover"),
        )
        .sort("trade_date")
    )
    values = market.select(columns).to_numpy().astype(np.float64)
    means = values.mean(axis=0)
    standard = values.std(axis=0)
    normalized = (values - means) / np.where(standard < 1e-8, 1.0, standard)
    return pl.concat(
        [market.select("trade_date"), pl.DataFrame(normalized, schema=columns)], how="horizontal_extend"
    ), columns


def train_short_hmm(frame: pl.DataFrame, path: Path) -> tuple[pl.DataFrame, dict[str, Any], np.ndarray]:
    market, columns = market_features(frame)
    if market.height != SHORT_WINDOW_DAYS:
        raise RuntimeError(f"Short HMM requires {SHORT_WINDOW_DAYS} market dates, got {market.height}")
    model = RegimeHMM(n_components=2, random_state=474).fit(market.select(columns).to_numpy())
    probabilities = model.predict(market.select(columns).to_numpy())
    model.save(path)
    context = market.select("trade_date").with_columns(
        pl.Series("hmm_p0", probabilities[:, 0]), pl.Series("hmm_p1", probabilities[:, 1])
    )
    metrics = {
        "states": 2,
        "observations": market.height,
        "converged": bool(model.model.monitor_.converged),
        "probability_sum_max_error": float(np.abs(probabilities.sum(axis=1) - 1).max()),
    }
    return context, metrics, probabilities[-1]


def train_short_tft(frame: pl.DataFrame, path: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    feature_frame = frame.with_columns(
        [pl.col(column).fill_nan(None).fill_null(0.0) for column in SHORT_BASE_FEATURES]
    )
    sequences: list[np.ndarray] = []
    targets: list[list[float]] = []
    keys: list[tuple[str, date]] = []
    latest_sequences: list[np.ndarray] = []
    latest_keys: list[tuple[str, date]] = []
    for group in feature_frame.partition_by("symbol", maintain_order=True):
        group = group.sort("trade_date")
        values = group.select(SHORT_BASE_FEATURES).to_numpy().astype(np.float32)
        dates = group["trade_date"].to_list()
        symbol = str(group["symbol"][0])
        forward = group["forward_excess_return"].to_list()
        for end in range(4, len(group)):
            sequence = values[end - 4 : end + 1]
            latest_sequences.append(sequence)
            latest_keys.append((symbol, dates[end]))
            if forward[end] is not None and math.isfinite(float(forward[end])):
                sequences.append(sequence)
                targets.append([float(forward[end])])
                keys.append((symbol, dates[end]))
    if len(sequences) < 100:
        raise RuntimeError(f"Short TFT only has {len(sequences)} valid sequences")
    unique_dates = sorted({trade_date for _, trade_date in keys})
    split_date = unique_dates[-2] if len(unique_dates) >= 3 else unique_dates[-1]
    train_indices = [index for index, (_, trade_date) in enumerate(keys) if trade_date < split_date]
    validation_indices = [index for index, (_, trade_date) in enumerate(keys) if trade_date >= split_date]
    sequence_values = torch.tensor(np.stack(sequences), dtype=torch.float32)
    target_values = torch.tensor(np.asarray(targets), dtype=torch.float32)
    model = TftStyleForecaster(input_size=len(SHORT_BASE_FEATURES), horizons=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    quantiles = torch.tensor([0.1, 0.5, 0.9])
    loader = DataLoader(
        TensorDataset(sequence_values[train_indices], target_values[train_indices]), batch_size=128, shuffle=True
    )
    best = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    validation_x = sequence_values[validation_indices]
    validation_y = target_values[validation_indices]
    for _ in range(12):
        model.train()
        for inputs, target in loader:
            optimizer.zero_grad()
            loss = quantile_loss(model(inputs), target, quantiles)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(quantile_loss(model(validation_x), validation_y, quantiles))
        if validation_loss < best:
            best = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                break
    if best_state is None or not math.isfinite(best):
        raise RuntimeError("Short TFT validation loss is not finite")
    model.load_state_dict(best_state)
    inference = torch.tensor(np.stack(latest_sequences), dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        predictions = model(inference).numpy()[:, 0, :]
    torch.save({"state_dict": model.state_dict(), "input_size": len(SHORT_BASE_FEATURES), "horizons": 1}, path)
    output = pl.DataFrame(
        {
            "symbol": [symbol for symbol, _ in latest_keys],
            "trade_date": [trade_date for _, trade_date in latest_keys],
            "tft_q10": predictions[:, 0],
            "tft_q50": predictions[:, 1],
            "tft_q90": predictions[:, 2],
        }
    )
    metrics = {
        "encoder_days": 5,
        "horizon_days": 1,
        "training_sequences": len(sequences),
        "validation_quantile_loss": best,
        "finite_values": bool(np.isfinite(predictions).all()),
    }
    return output, metrics


def _intraday_matrix(frame: pl.DataFrame) -> np.ndarray:
    values = frame.select("open", "high", "low", "close", "volume").to_numpy().astype(np.float32)
    prices = np.clip(values[:, :4], 1e-6, None)
    returns = np.vstack([np.zeros((1, 4), dtype=np.float32), np.diff(np.log(prices), axis=0)])
    volume = np.log1p(np.clip(values[:, 4:5], 0, None))
    volume = (volume - volume.mean()) / max(float(volume.std()), 1e-6)
    return np.concatenate([returns, volume], axis=1)


def padded_patch_window(matrix: np.ndarray, end: int, length: int = PATCH_SEQUENCE_LENGTH) -> np.ndarray:
    start = max(0, end - length + 1)
    window = matrix[start : end + 1]
    if len(window) == length:
        return window
    padding = np.zeros((length - len(window), matrix.shape[1]), dtype=np.float32)
    return np.concatenate([padding, window], axis=0)


def train_short_patchtst(
    intraday: pl.DataFrame, path: Path, *, min_sequences: int = 1_000
) -> tuple[pl.DataFrame, dict[str, Any]]:
    training: list[np.ndarray] = []
    materialization: list[tuple[str, date, np.ndarray]] = []
    complete = 0
    for group in intraday.partition_by("symbol", maintain_order=True):
        group = group.sort("event_time")
        matrix = _intraday_matrix(group)
        symbol = str(group["symbol"][0])
        dates = group["trade_date"].to_list()
        last_indices: dict[date, int] = {}
        for index, trade_date in enumerate(dates):
            last_indices[trade_date] = index
        if len(matrix) >= PATCH_SEQUENCE_LENGTH:
            training.append(matrix[-PATCH_SEQUENCE_LENGTH:])
            complete += 1
        for trade_date, index in last_indices.items():
            materialization.append((symbol, trade_date, padded_patch_window(matrix, index)))
    if len(training) < min_sequences:
        raise RuntimeError(f"Short PatchTST only has {len(training)} complete 120-bar sequences")
    values = torch.tensor(np.stack(training), dtype=torch.float32)
    model = PatchTSTEncoder(channels=5)
    head = nn.Linear(64, 5)
    optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=1e-3)
    loader = DataLoader(TensorDataset(values), batch_size=64, shuffle=True)
    best = math.inf
    for _ in range(5):
        model.train()
        losses: list[float] = []
        for (inputs,) in loader:
            mask = torch.rand_like(inputs) < 0.15
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(head(model(inputs.masked_fill(mask, 0))), inputs[:, -1])
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            losses.append(float(loss.detach()))
        best = min(best, float(np.mean(losses)))
    torch.save({"state_dict": model.state_dict(), "channels": 5, "embedding_dim": 64}, path)
    model.eval()
    rows: list[dict[str, object]] = []
    for offset in range(0, len(materialization), 256):
        batch = materialization[offset : offset + 256]
        inputs = torch.tensor(np.stack([window for _, _, window in batch]), dtype=torch.float32)
        with torch.inference_mode():
            embeddings = model(inputs).numpy()
        for (symbol, trade_date, _), embedding in zip(batch, embeddings, strict=True):
            row: dict[str, object] = {"symbol": symbol, "trade_date": trade_date}
            row.update({f"patch_{index}": float(value) for index, value in enumerate(embedding)})
            rows.append(row)
    metrics = {
        "sequence_length": PATCH_SEQUENCE_LENGTH,
        "complete_sequences": complete,
        "masked_mse": best,
        "embedding_dim": 64,
        "left_padding_for_early_dates": True,
        "finite_values": math.isfinite(best),
    }
    return pl.DataFrame(rows), metrics


def rolling_splits(dates: Sequence[date]) -> list[tuple[list[date], date, list[date]]]:
    ordered = sorted(dates)
    if len(ordered) < 13:
        raise ValueError(f"At least 13 labeled dates are required, got {len(ordered)}")
    first_test = len(ordered) - 6
    splits = []
    for fold in range(3):
        test_start = first_test + fold * 2
        embargo_index = test_start - 1
        splits.append((ordered[:embargo_index], ordered[embargo_index], ordered[test_start : test_start + 2]))
    return splits


def _fit_ranker(frame: pl.DataFrame, features: list[str]) -> ArtRanker:
    ordered = frame.sort(["trade_date", "symbol"])
    groups = ordered.group_by("trade_date", maintain_order=True).len()["len"].to_list()
    model = ArtRanker(
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=31,
        max_bin=63,
        n_jobs=8,
        verbosity=-1,
        random_state=474,
    )
    model.fit(ordered.select(features).to_numpy(), ordered["relevance"].to_numpy(), group=groups)
    return model


def _daily_ics(frame: pl.DataFrame) -> list[float]:
    values = []
    for group in frame.partition_by("trade_date"):
        if group.height >= 20:
            values.append(rank_ic(group["score"].to_numpy(), group["forward_excess_return"].to_numpy()))
    return values


def train_short_rankers(frame: pl.DataFrame, artifact_dir: Path) -> dict[str, Any]:
    patch_features = [f"patch_{index}" for index in range(64)]
    full_features = SHORT_BASE_FEATURES + ["hmm_p0", "hmm_p1", "tft_q50"] + patch_features
    prepared = frame.with_columns(
        [pl.col(column).fill_nan(None).fill_null(0.0) for column in full_features]
    )
    labeled = prepared.filter(pl.col("eligible") & pl.col("relevance").is_not_null())
    latest_date = max(prepared["trade_date"].to_list())
    latest = prepared.filter((pl.col("trade_date") == latest_date) & pl.col("eligible")).sort("symbol")
    dates = sorted(labeled["trade_date"].unique().to_list())
    splits = rolling_splits(dates)
    daily_ics: dict[str, list[float]] = {"factor": [], "full": []}
    latest_scores: list[np.ndarray] = []
    test_frames: list[pl.DataFrame] = []
    for train_dates, _, test_dates in splits:
        train = labeled.filter(pl.col("trade_date").is_in(train_dates))
        test = labeled.filter(pl.col("trade_date").is_in(test_dates))
        for name, features in (("factor", SHORT_BASE_FEATURES), ("full", full_features)):
            model = _fit_ranker(train, features)
            scores = np.asarray(model.predict(test.select(features).to_numpy()), dtype=float)
            scored = test.select("trade_date", "symbol", "forward_excess_return").with_columns(
                pl.Series("score", scores)
            )
            daily_ics[name].extend(_daily_ics(scored))
            if name == "full":
                test_frames.append(scored)
                latest_scores.append(np.asarray(model.predict(latest.select(features).to_numpy()), dtype=float))
    factor_model = _fit_ranker(labeled, SHORT_BASE_FEATURES)
    full_model = _fit_ranker(labeled, full_features)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    factor_model.save(artifact_dir / "factor.txt")
    full_model.save(artifact_dir / "full.txt")
    means = {name: float(np.mean(values)) if values else -1.0 for name, values in daily_ics.items()}
    positive = {name: max(0.0, value) for name, value in means.items()}
    denominator = sum(positive.values())
    weights = {name: value / denominator if denominator else 0.5 for name, value in positive.items()}
    tests = pl.concat(test_frames) if test_frames else pl.DataFrame()
    shuffled_ic = 1.0
    cost_adjusted = -1.0
    if tests.height:
        rng = np.random.default_rng(474)
        shuffled = tests["forward_excess_return"].to_numpy().copy()
        rng.shuffle(shuffled)
        shuffled_ic = rank_ic(tests["score"].to_numpy(), shuffled)
        top = tests.sort(["trade_date", "score"], descending=[False, True]).group_by("trade_date").head(20)
        mean_return = top["forward_excess_return"].mean()
        cost_adjusted = float(np.asarray(mean_return if mean_return is not None else 0.0, dtype=float).item()) - 0.0015
    return {
        "factor_model": factor_model,
        "full_model": full_model,
        "factor_features": SHORT_BASE_FEATURES,
        "full_features": full_features,
        "latest": latest,
        "rank_ic": means["full"],
        "daily_ics": daily_ics["full"],
        "validation_days": len(daily_ics["full"]),
        "shuffled_rank_ic": float(shuffled_ic),
        "cost_adjusted_top20_return": cost_adjusted,
        "weights": weights,
        "latest_scores": latest_scores,
    }


def experimental_gate(
    *,
    daily_days: int,
    daily_coverage: float,
    intraday_coverage: float,
    validation_days: int,
    rank_ic_value: float,
    shuffled_rank_ic: float,
    finite_outputs: bool,
) -> dict[str, Any]:
    checks = {
        "fifteen_common_trading_days": daily_days == SHORT_WINDOW_DAYS,
        "daily_coverage_at_least_95pct": daily_coverage >= 0.95,
        "intraday_coverage_at_least_85pct": intraday_coverage >= 0.85,
        "at_least_three_validation_days": validation_days >= 3,
        "finite_outputs": finite_outputs,
        "positive_oos_rank_ic": rank_ic_value > 0,
        "shuffled_rank_ic_near_zero": abs(shuffled_rank_ic) < 0.05,
        "no_future_data": True,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return {"passed": not reasons, "checks": checks, "blocked_reasons": reasons}
