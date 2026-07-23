from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd


def get_akshare_history(
    symbol: str,
    days: int = 15,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    使用AKShare获取A股最近若干个交易日的数据。

    参数
    ----------
    symbol:
        六位股票代码，例如：
        000001：平安银行
        600519：贵州茅台
        688279：峰岹科技

    days:
        返回最近多少个有效交易日。

    adjust:
        ""：不复权
        "qfq"：前复权
        "hfq"：后复权
    """

    # 使用北京时间确定当前日期
    end_date = datetime.now(
        ZoneInfo("Asia/Shanghai")
    ).date()

    # 多获取一些自然日，避免周末、节假日和停牌影响
    start_date = end_date - timedelta(days=max(60, days * 4))

    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        adjust=adjust,
    )

    if df.empty:
        raise ValueError(f"没有获取到股票 {symbol} 的行情数据")

    # 统一为英文列名，方便后续建模和数据库存储
    column_mapping = {
        "日期": "date",
        "股票代码": "code",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "price_change",
        "换手率": "turnover",
    }

    df = df.rename(columns=column_mapping)

    if "code" not in df.columns:
        df["code"] = symbol

    df["date"] = pd.to_datetime(df["date"])

    numeric_columns = [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
        "pct_change",
        "price_change",
        "turnover",
    ]

    existing_numeric_columns = [
        column for column in numeric_columns
        if column in df.columns
    ]

    df[existing_numeric_columns] = df[
        existing_numeric_columns
    ].apply(pd.to_numeric, errors="coerce")

    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"])
        .tail(days)
        .reset_index(drop=True)
    )

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["source"] = "akshare"

    return df


if __name__ == "__main__":
    data = get_akshare_history(
        symbol="000001",
        days=15,
        adjust="qfq",
    )

    print(data)

    data.to_csv(
        "data/akshare_000001_last_15_days.csv",
        index=False,
        encoding="utf-8-sig",
    )