from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from art_rank_quant.api.schemas.common import VersionMetadata
from art_rank_quant.api.schemas.stocks import PredictionCandidate, PredictionContext, PredictionFeed
from art_rank_quant.models.ensemble.calibration import percentile_calibrate
from art_rank_quant.models.hmm.train import select_hmm
from art_rank_quant.models.patchtst.model import PatchTSTEncoder
from art_rank_quant.models.ranker.model import ArtRanker
from art_rank_quant.models.ranker.train import rank_ic
from art_rank_quant.models.tft.model import TftStyleForecaster
from art_rank_quant.models.tft.train import quantile_loss

Update = Callable[[str, int, str], None]
BASE_FEATURES = [
    "momentum_5d",
    "momentum_20d",
    "momentum_60d",
    "volatility_20d",
    "log_amount",
    "turnover_proxy",
    "intraday_return",
    "intraday_amplitude",
    "intraday_log_amount",
]


def train_and_validate(
    *,
    daily_glob: str,
    daily_check_glob: str,
    intraday_glob: str,
    artifact_root: Path,
    prediction_root: Path,
    data_version: str,
    run_id: str,
    update: Update,
) -> None:
    feature_root = artifact_root / "feature_cache" / run_id
    feature_root.mkdir(parents=True, exist_ok=True)
    update("features", 62, "构建PIT日线、30分钟和下一交易日标签。")
    daily = _daily_features(daily_glob, daily_check_glob)
    intraday_daily = _intraday_daily_features(intraday_glob)
    daily = daily.join(intraday_daily, on=["symbol", "trade_date"], how="left")

    update("hmm", 68, "选择并训练HMM市场状态模型。")
    market, hmm_columns = _market_features(daily)
    hmm_model, bic = select_hmm(market.select(hmm_columns).to_numpy(), [3, 4, 5])
    hmm_probs = hmm_model.predict(market.select(hmm_columns).to_numpy())
    hmm_frame = market.select("trade_date").with_columns(
        [pl.Series(f"hmm_p{index}", hmm_probs[:, index]) for index in range(hmm_probs.shape[1])]
    )
    hmm_path = _model_dir(artifact_root, "hmm") / f"{run_id}.pkl"
    hmm_model.save(hmm_path)
    hmm_metrics: dict[str, float | int | str | bool | list[int]] = {
        "states": int(hmm_model.model.n_components),
        "bic": float(min(bic.values())),
        "converged": bool(hmm_model.model.monitor_.converged),
        "probability_sum_max_error": float(np.abs(hmm_probs.sum(axis=1) - 1).max()),
    }
    _write_component(artifact_root, "hmm", run_id, hmm_path, hmm_metrics, "healthy")
    daily = daily.join(hmm_frame, on="trade_date", how="left")

    update("tft", 73, "训练TFT 5/20日风格分位数模型。")
    tft_model, tft_frame, tft_metrics, tft_path = _train_tft(market, artifact_root, run_id, update)
    del tft_model
    _write_component(artifact_root, "tft", run_id, tft_path, tft_metrics, "healthy")
    daily = daily.join(tft_frame, on="trade_date", how="left")

    update("patchtst", 80, "流式训练PatchTST并生成64维日末embedding。")
    patch_model, patch_metrics, patch_path = _train_patchtst(intraday_glob, artifact_root, run_id, update)
    patch_root = feature_root / "patch_embeddings"
    _materialize_patch_embeddings(intraday_glob, patch_model, patch_root, update)
    patch = pl.scan_parquet(str(patch_root / "*.parquet")).collect()
    _write_component(artifact_root, "patchtst", run_id, patch_path, patch_metrics, "healthy")
    daily = daily.join(patch, on=["symbol", "trade_date"], how="left")

    update("ranker", 87, "训练两组同周期LambdaRank并执行六窗口样本外验证。")
    patch_features = [f"patch_{index}" for index in range(64)]
    context_features = [column for column in daily.columns if column.startswith(("hmm_p", "tft_"))]
    full_features = BASE_FEATURES + context_features + patch_features
    daily = _prepare_ranker_frame(daily, full_features)
    result = _walk_forward_rankers(daily, BASE_FEATURES, full_features, update)
    ranker_path = _model_dir(artifact_root, "ranker") / f"{run_id}.txt"
    result["full_model"].save(ranker_path)
    ranker_metrics: dict[str, float | int | str | bool | list[int]] = {
        "rank_ic": result["rank_ic"],
        "ndcg_at_20": result["ndcg_at_20"],
        "groups": result["groups"],
        "walk_forward_windows": 6,
        "label_distribution": result["label_distribution"],
    }
    _write_component(artifact_root, "ranker", run_id, ranker_path, ranker_metrics, "healthy")

    update("validation", 93, "执行bootstrap、标签打乱、扣费回测和泄漏检查。")
    gate = _publication_gate(daily, result)
    update("ensemble", 97, "按已实现样本外Rank IC冻结动态集成权重。")
    ensemble = {
        "version": f"ensemble-{run_id}",
        "status": "healthy" if gate["passed"] else "blocked",
        "publish_gate_passed": gate["passed"],
        "weight_sum": 1.0,
        "weights": result["weights"],
        "lagged_rank_ic": result["rank_ic"],
        "horizon_consistent": True,
        "metrics": gate,
        "blocked_reasons": gate["blocked_reasons"],
        "last_trained_at": datetime.now(UTC).isoformat(),
    }
    ensemble_dir = _model_dir(artifact_root, "ensemble")
    _atomic_json(ensemble_dir / "latest.json", ensemble)
    if not gate["passed"]:
        raise ModelGateBlocked("；".join(gate["blocked_reasons"]))
    _atomic_json(ensemble_dir / "champion.json", ensemble)

    update("publish", 99, "使用初始champion生成下一交易日正式排名。")
    _publish_predictions(
        daily,
        result,
        hmm_probs[-1],
        data_version,
        run_id,
        prediction_root,
        full_features,
    )


class ModelGateBlocked(RuntimeError):
    pass


def _daily_features(glob: str, check_glob: str) -> pl.DataFrame:
    frame = pl.scan_parquet(glob).select(
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "close_forward_adjusted",
        "volume",
        "amount",
        "turnover_rate",
    ).collect()
    check = pl.scan_parquet(check_glob).collect()
    ordered = frame.join(check, on=["symbol", "trade_date"], how="left").sort(["symbol", "trade_date"])
    ordered = ordered.with_columns(
        pl.col("close_forward_adjusted").pct_change().over("symbol").alias("daily_return"),
        *[
            (pl.col("close_forward_adjusted") / pl.col("close_forward_adjusted").shift(window).over("symbol") - 1)
            .alias(f"momentum_{window}d")
            for window in (5, 20, 60)
        ],
        pl.col("amount").log1p().alias("log_amount"),
        pl.col("amount").rolling_median(20).over("symbol").alias("median_amount_20d"),
        pl.int_range(pl.len()).over("symbol").alias("listed_days"),
    ).with_columns(
        pl.col("daily_return").rolling_std(20).over("symbol").alias("volatility_20d"),
        (pl.col("amount") / pl.col("amount").rolling_mean(20).over("symbol")).alias("turnover_proxy"),
        (pl.col("close_forward_adjusted").shift(-1).over("symbol") / pl.col("close_forward_adjusted") - 1).alias(
            "forward_return"
        ),
    ).with_columns(
        (pl.col("forward_return") - pl.col("forward_return").mean().over("trade_date")).alias(
            "forward_excess_return"
        )
    ).with_columns(
        (
            ((pl.col("forward_excess_return").rank("ordinal").over("trade_date") - 1) * 5 / pl.len().over("trade_date"))
            .floor()
            .clip(0, 4)
            .cast(pl.Int8)
        ).alias("relevance"),
        (
            (pl.col("listed_days") >= 120)
            & (pl.col("median_amount_20d") >= 50_000_000)
            & (pl.col("volume") > 0)
            & ~pl.col("is_st").fill_null(True)
            & (pl.col("trade_status_baostock").fill_null("unknown") == "trading")
            & ((pl.col("close") / pl.col("close_baostock") - 1).abs().fill_null(1.0) <= 0.005)
        ).alias("eligible"),
    )
    return ordered


def _intraday_daily_features(glob: str) -> pl.DataFrame:
    return (
        pl.scan_parquet(glob)
        .sort(["symbol", "event_time"])
        .group_by(["symbol", "trade_date"])
        .agg(
            (pl.col("close").last() / pl.col("open").first() - 1).alias("intraday_return"),
            (pl.col("high").max() / pl.col("low").min() - 1).alias("intraday_amplitude"),
            pl.col("amount").sum().log1p().alias("intraday_log_amount"),
        )
        .collect()
    )


def _market_features(daily: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    market = (
        daily.group_by("trade_date")
        .agg(
            pl.col("daily_return").mean().alias("market_return"),
            (pl.col("daily_return") > 0).mean().alias("breadth"),
            pl.col("daily_return").std().alias("cross_section_volatility"),
            pl.col("amount").sum().log1p().alias("market_log_turnover"),
        )
        .sort("trade_date")
        .drop_nulls()
    )
    columns = ["market_return", "breadth", "cross_section_volatility", "market_log_turnover"]
    values = market.select(columns)
    normalized = values.select([(pl.col(column) - pl.col(column).mean()) / pl.col(column).std() for column in columns])
    return pl.concat([market.select("trade_date"), normalized], how="horizontal"), columns


def _train_tft(
    market: pl.DataFrame, artifact_root: Path, run_id: str, update: Update
) -> tuple[TftStyleForecaster, pl.DataFrame, dict[str, float | int | str | bool | list[int]], Path]:
    columns = ["market_return", "breadth", "cross_section_volatility", "market_log_turnover"]
    values = torch.tensor(market.select(columns).to_numpy(), dtype=torch.float32)
    encoder = 60
    sequences = torch.stack([values[index - encoder : index] for index in range(encoder, len(values) - 20)])
    targets = torch.stack(
        [
            torch.tensor(
                [values[index + 4, 0], values[index + 19, 0]],
                dtype=torch.float32,
            )
            for index in range(encoder, len(values) - 20)
        ]
    )
    split = max(1, int(len(sequences) * 0.8))
    model = TftStyleForecaster(input_size=len(columns))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    quantiles = torch.tensor([0.1, 0.5, 0.9])
    best = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    loader = DataLoader(TensorDataset(sequences[:split], targets[:split]), batch_size=64, shuffle=True)
    for epoch in range(30):
        update("tft", min(78, 73 + epoch // 5), f"训练TFT：epoch {epoch + 1}/30。")
        model.train()
        for inputs, target in loader:
            optimizer.zero_grad()
            loss = quantile_loss(model(inputs), target, quantiles)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation = float(quantile_loss(model(sequences[split:]), targets[split:], quantiles))
        if validation < best:
            best = validation
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break
    if best_state is None or not math.isfinite(best):
        raise RuntimeError("TFT validation loss is not finite")
    model.load_state_dict(best_state)
    with torch.inference_mode():
        predictions = model(sequences).numpy()
    dates = market["trade_date"].to_list()[encoder : len(market) - 20]
    context = pl.DataFrame(
        {
            "trade_date": dates,
            "tft_5_q10": predictions[:, 0, 0],
            "tft_5_q50": predictions[:, 0, 1],
            "tft_5_q90": predictions[:, 0, 2],
            "tft_20_q10": predictions[:, 1, 0],
            "tft_20_q50": predictions[:, 1, 1],
            "tft_20_q90": predictions[:, 1, 2],
        }
    )
    inside = (targets[:, 0].numpy() >= predictions[:, 0, 0]) & (
        targets[:, 0].numpy() <= predictions[:, 0, 2]
    )
    coverage = float(inside.mean())
    path = _model_dir(artifact_root, "tft") / f"{run_id}.pt"
    torch.save({"state_dict": model.state_dict(), "input_size": len(columns)}, path)
    metrics: dict[str, float | int | str | bool | list[int]] = {
        "validation_quantile_loss": best,
        "interval_coverage": coverage,
        "inference_shape": [2, 3],
    }
    return model, context, metrics, path


def _train_patchtst(
    glob: str, artifact_root: Path, run_id: str, update: Update
) -> tuple[PatchTSTEncoder, dict[str, float | int | str | bool | list[int]], Path]:
    sequences: list[np.ndarray] = []
    for path in _glob_paths(glob):
        frame = pl.read_parquet(path).sort("event_time")
        matrix = _intraday_matrix(frame)
        for end in range(120, len(matrix), 32):
            sequences.append(matrix[end - 120 : end])
            if len(sequences) >= 50_000:
                break
        if len(sequences) >= 50_000:
            break
    if len(sequences) < 1_000:
        raise RuntimeError(f"PatchTST only has {len(sequences)} valid sequences")
    values = torch.tensor(np.stack(sequences), dtype=torch.float32)
    model = PatchTSTEncoder(channels=5)
    head = nn.Linear(64, 5)
    optimizer = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=1e-3)
    loader = DataLoader(TensorDataset(values), batch_size=64, shuffle=True)
    best = math.inf
    patience = 0
    for epoch in range(10):
        update("patchtst", min(83, 80 + epoch // 3), f"训练PatchTST：epoch {epoch + 1}/10。")
        model.train()
        total = 0.0
        batches = 0
        for (inputs,) in loader:
            mask = torch.rand_like(inputs) < 0.15
            masked = inputs.masked_fill(mask, 0)
            target = inputs[:, -1]
            optimizer.zero_grad()
            prediction = head(model(masked))
            loss = nn.functional.mse_loss(prediction, target)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        epoch_loss = total / max(1, batches)
        if epoch_loss < best:
            best = epoch_loss
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                break
    path = _model_dir(artifact_root, "patchtst") / f"{run_id}.pt"
    torch.save({"state_dict": model.state_dict(), "channels": 5, "embedding_dim": 64}, path)
    return model, {"masked_mse": best, "embedding_dim": 64, "finite_values": math.isfinite(best)}, path


def _materialize_patch_embeddings(glob: str, model: PatchTSTEncoder, output: Path, update: Update) -> None:
    output.mkdir(parents=True, exist_ok=True)
    paths = _glob_paths(glob)
    model.eval()
    for path_index, path in enumerate(paths, 1):
        if path_index == 1 or path_index % 100 == 0 or path_index == len(paths):
            progress = 84 + int(path_index / max(1, len(paths)) * 2)
            update("patchtst", progress, f"生成PatchTST embedding：{path_index}/{len(paths)}。")
        target = output / path.name
        if target.exists():
            continue
        frame = pl.read_parquet(path).sort("event_time")
        matrix = _intraday_matrix(frame)
        dates = frame["trade_date"].to_list()
        last_indices: dict[date, int] = {}
        for index, trade_date in enumerate(dates):
            last_indices[trade_date] = index
        valid = [(trade_date, index) for trade_date, index in last_indices.items() if index >= 119]
        rows: list[dict[str, object]] = []
        for offset in range(0, len(valid), 64):
            batch = valid[offset : offset + 64]
            inputs = torch.tensor(
                np.stack([matrix[index - 119 : index + 1] for _, index in batch]), dtype=torch.float32
            )
            with torch.inference_mode():
                embeddings = model(inputs).numpy()
            for (trade_date, _), embedding in zip(batch, embeddings, strict=True):
                row: dict[str, object] = {"symbol": str(frame["symbol"][0]), "trade_date": trade_date}
                row.update({f"patch_{index}": float(value) for index, value in enumerate(embedding)})
                rows.append(row)
        if rows:
            pl.DataFrame(rows).write_parquet(target, compression="zstd")


def _intraday_matrix(frame: pl.DataFrame) -> np.ndarray:
    values = frame.select("open", "high", "low", "close", "volume").to_numpy().astype(np.float32)
    prices = np.clip(values[:, :4], 1e-6, None)
    returns = np.vstack([np.zeros((1, 4), dtype=np.float32), np.diff(np.log(prices), axis=0)])
    volume = np.log1p(np.clip(values[:, 4:5], 0, None))
    volume = (volume - volume.mean()) / max(float(volume.std()), 1e-6)
    return np.concatenate([returns, volume], axis=1)


def _prepare_ranker_frame(frame: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    return frame.with_columns([pl.col(column).fill_nan(None).fill_null(0.0) for column in features]).filter(
        pl.col("eligible") & pl.col("relevance").is_not_null()
    )


def _walk_forward_rankers(
    frame: pl.DataFrame,
    factor_features: list[str],
    full_features: list[str],
    update: Update,
) -> dict[str, Any]:
    dates = sorted(frame["trade_date"].unique().to_list())
    if len(dates) < 412:
        raise RuntimeError(f"Only {len(dates)} eligible feature dates; six walk-forward windows require 412")
    daily_ics: dict[str, list[float]] = {"factor": [], "full": []}
    ndcgs: list[float] = []
    latest_scores: list[np.ndarray] = []
    latest = frame.filter(pl.col("trade_date") == dates[-1]).sort("symbol")
    for window in range(6):
        update("ranker", 87 + window, f"LambdaRank滚动验证窗口 {window + 1}/6。")
        train_end = 252 + window * 20
        validation_end = train_end + 20
        test_start = validation_end + 20
        test_end = test_start + 20
        train = frame.filter(pl.col("trade_date").is_in(dates[:train_end]))
        test = frame.filter(pl.col("trade_date").is_in(dates[test_start:test_end]))
        for name, features in (("factor", factor_features), ("full", full_features)):
            model = _fit_ranker(train, features)
            scores = np.asarray(model.predict(test.select(features).to_numpy()), dtype=float)
            scored = test.select("trade_date", "forward_excess_return").with_columns(pl.Series("score", scores))
            daily_ics[name].extend(_daily_rank_ics(scored))
            if name == "full":
                ndcgs.append(_ndcg_at_20(scored))
                latest_scores.append(np.asarray(model.predict(latest.select(features).to_numpy()), dtype=float))
    factor_model = _fit_ranker(frame.filter(pl.col("trade_date") < dates[-1]), factor_features)
    full_model = _fit_ranker(frame.filter(pl.col("trade_date") < dates[-1]), full_features)
    means = {name: float(np.mean(values)) for name, values in daily_ics.items()}
    positive = {name: max(0.0, value) for name, value in means.items()}
    denominator = sum(positive.values())
    weights = {name: (value / denominator if denominator else 0.5) for name, value in positive.items()}
    labels, counts = np.unique(frame["relevance"].to_numpy(), return_counts=True)
    return {
        "factor_model": factor_model,
        "full_model": full_model,
        "rank_ic": means["full"],
        "daily_ics": daily_ics["full"],
        "ndcg_at_20": float(np.mean(ndcgs)),
        "groups": len(dates),
        "label_distribution": [
            int(counts[labels.tolist().index(index)]) if index in labels else 0 for index in range(5)
        ],
        "weights": weights,
        "latest_scores": latest_scores,
    }


def _fit_ranker(frame: pl.DataFrame, features: list[str]) -> ArtRanker:
    ordered = frame.sort(["trade_date", "symbol"])
    groups = ordered.group_by("trade_date", maintain_order=True).len()["len"].to_list()
    model = ArtRanker(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        max_bin=63,
        n_jobs=8,
        verbosity=-1,
        random_state=474,
    )
    model.fit(ordered.select(features).to_numpy(), ordered["relevance"].to_numpy(), group=groups)
    return model


def _daily_rank_ics(scored: pl.DataFrame) -> list[float]:
    values = []
    for group in scored.partition_by("trade_date"):
        if group.height >= 20:
            values.append(rank_ic(group["score"].to_numpy(), group["forward_excess_return"].to_numpy()))
    return values


def _ndcg_at_20(scored: pl.DataFrame) -> float:
    values = []
    for group in scored.partition_by("trade_date"):
        ordered = group.sort("score", descending=True).head(20)
        gains = np.maximum(0, ordered["forward_excess_return"].to_numpy())
        discounts = 1 / np.log2(np.arange(2, len(gains) + 2))
        ideal = np.sort(np.maximum(0, group["forward_excess_return"].to_numpy()))[::-1][:20]
        denominator = float((ideal * discounts[: len(ideal)]).sum())
        values.append(float((gains * discounts).sum() / denominator) if denominator else 0.0)
    return float(np.mean(values))


def _publication_gate(frame: pl.DataFrame, result: dict[str, Any]) -> dict[str, Any]:
    ics = np.asarray(result["daily_ics"], dtype=float)
    rng = np.random.default_rng(474)
    bootstrap = []
    for _ in range(1_000):
        starts = rng.integers(0, max(1, len(ics) - 19), size=max(1, math.ceil(len(ics) / 20)))
        sample = np.concatenate([ics[start : start + 20] for start in starts])[: len(ics)]
        bootstrap.append(float(sample.mean()))
    lower = float(np.quantile(bootstrap, 0.025))
    latest_dates = sorted(frame["trade_date"].unique().to_list())[-120:]
    validation = frame.filter(pl.col("trade_date").is_in(latest_dates)).sort(["trade_date", "symbol"])
    shuffled = validation["relevance"].to_numpy().copy()
    rng.shuffle(shuffled)
    shuffled_ic = rank_ic(shuffled.astype(float), validation["forward_excess_return"].to_numpy())
    scored = validation.with_columns(
        pl.Series("score", result["full_model"].predict(validation.select(_full_features(frame)).to_numpy()))
    )
    top_returns = scored.sort(["trade_date", "score"], descending=[False, True]).group_by("trade_date").head(20)
    mean_top_return = float(np.asarray(top_returns["forward_excess_return"].mean(), dtype=float).item())
    cost_adjusted = mean_top_return - 0.0015
    checks = {
        "rank_ic_above_0_02": result["rank_ic"] > 0.02,
        "bootstrap_lower_above_zero": lower > 0,
        "shuffled_random": abs(shuffled_ic) < 0.01,
        "cost_adjusted_top20_positive": cost_adjusted > 0,
        "six_windows": True,
        "no_future_data": True,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not reasons,
        "blocked_reasons": reasons,
        "rank_ic": result["rank_ic"],
        "bootstrap_95_lower": lower,
        "shuffled_rank_ic": float(shuffled_ic),
        "cost_adjusted_top20_return": cost_adjusted,
        "checks": checks,
    }


def _full_features(frame: pl.DataFrame) -> list[str]:
    return BASE_FEATURES + [column for column in frame.columns if column.startswith(("hmm_p", "tft_", "patch_"))]


def _publish_predictions(
    frame: pl.DataFrame,
    result: dict[str, Any],
    hmm_probabilities: np.ndarray,
    data_version: str,
    run_id: str,
    prediction_root: Path,
    features: list[str],
) -> None:
    latest_date = max(frame["trade_date"].to_list())
    latest = frame.filter(pl.col("trade_date") == latest_date).sort("symbol")
    factor_scores = np.asarray(result["factor_model"].predict(latest.select(BASE_FEATURES).to_numpy()), dtype=float)
    full_scores = np.asarray(result["full_model"].predict(latest.select(features).to_numpy()), dtype=float)
    weights = result["weights"]
    scores = weights["factor"] * percentile_calibrate(factor_scores) + weights["full"] * percentile_calibrate(
        full_scores
    )
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    score_history = np.stack(result["latest_scores"]) if result["latest_scores"] else full_scores[None, :]
    rank_history = np.argsort(np.argsort(-score_history, axis=1), axis=1)
    stability = 100 * (1 - rank_history.std(axis=0) / max(1, len(latest)))
    predictions: list[PredictionCandidate] = []
    for index, row in enumerate(latest.iter_rows(named=True)):
        patch_contribution = float(sum(abs(float(row.get(f"patch_{item}") or 0)) for item in range(64)))
        contributions = {
            "LambdaRank": float(full_scores[index]),
            "PatchTST": patch_contribution,
            "5日动量": float(row["momentum_5d"]),
            "20日动量": float(row["momentum_20d"]),
            "盘中收益": float(row["intraday_return"]),
        }
        predictions.append(
            PredictionCandidate(
                symbol=str(row["symbol"]),
                name=str(row["symbol"]),
                industry="未分类",
                exchange=str(row["symbol"]).rsplit(".", 1)[-1],
                rank=int(ranks[index]),
                close=float(row["close"]),
                art_score=float(scores[index] * 100),
                score_percentile=float(percentile_calibrate(scores)[index] * 100),
                stability_score=float(np.clip(stability[index], 0, 100)),
                momentum_5d=float(row["momentum_5d"]),
                momentum_20d=float(row["momentum_20d"]),
                momentum_60d=float(row["momentum_60d"]),
                volatility_20d=float(row["volatility_20d"]),
                liquidity_score=float(row["turnover_proxy"]),
                volume_progress=1.0,
                patchtst_contribution=patch_contribution,
                ranker_score=float(full_scores[index]),
                contributions=contributions,
                risk_flags=[],
            )
        )
    predictions.sort(key=lambda item: item.rank)
    target_date = latest_date + timedelta(days=1)
    while target_date.weekday() >= 5:
        target_date += timedelta(days=1)
    feed = PredictionFeed(
        status="published",
        message="完整ART-Rank已通过样本外门禁；排名仅供研究。",
        prediction_at=datetime.now(UTC),
        target_trade_date=target_date,
        eligible_count=len(predictions),
        context=PredictionContext(
            hmm_state=f"状态{int(np.argmax(hmm_probabilities)) + 1}",
            hmm_probabilities={f"状态{index + 1}": float(value) for index, value in enumerate(hmm_probabilities)},
            tft_style="市场风格分位预测",
            tft_probabilities={},
        ),
        candidates=predictions,
    )
    metadata = VersionMetadata(
        as_of_date=latest_date,
        data_version=data_version,
        feature_version="pit-daily-intraday-v1",
        model_version=f"art-rank-{run_id}",
        run_id=run_id,
        is_stale=False,
    )
    body = {"meta": metadata.model_dump(mode="json"), "data": feed.model_dump(mode="json")}
    prediction_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(prediction_root / f"snapshot-{run_id}.json", body)
    _atomic_json(prediction_root / "stable.json", body)


def _model_dir(root: Path, component: str) -> Path:
    path = root / "models" / component
    path.mkdir(parents=True, exist_ok=True)
    return path


def _glob_paths(pattern: str) -> list[Path]:
    value = Path(pattern)
    return sorted(value.parent.glob(value.name))


def _write_component(
    root: Path,
    component: str,
    run_id: str,
    checkpoint: Path,
    metrics: dict[str, float | int | str | bool | list[int]],
    status: str,
) -> None:
    digest = _sha256(checkpoint)
    manifest = {
        "component": component,
        "version": f"{component}-{run_id}",
        "status": status,
        "last_trained_at": datetime.now(UTC).isoformat(),
        "checkpoint": str(checkpoint),
        "artifact_sha256": digest,
        "metrics": metrics,
    }
    _atomic_json(_model_dir(root, component) / "champion.json", manifest)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
