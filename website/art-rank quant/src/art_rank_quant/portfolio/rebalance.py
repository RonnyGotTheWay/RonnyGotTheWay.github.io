from __future__ import annotations


def rebalance_orders(
    current: dict[str, float], target: dict[str, float], nav: float, prices: dict[str, float]
) -> list[dict[str, float | int | str]]:
    orders: list[dict[str, float | int | str]] = []
    for symbol in sorted(set(current) | set(target)):
        delta_value = (target.get(symbol, 0.0) - current.get(symbol, 0.0)) * nav
        quantity = int(delta_value / prices[symbol] / 100) * 100
        if quantity:
            orders.append({"symbol": symbol, "side": "buy" if quantity > 0 else "sell", "quantity": abs(quantity)})
    return orders
