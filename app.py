from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "web_data"
SUMMARY_PATH = DATA_DIR / "portfolio_summary.json"
HOLDINGS_PATH = DATA_DIR / "portfolio_holdings.csv"
HISTORY_PATH = DATA_DIR / "portfolio_history.csv"

RISK_ASSET_CLASSES = {"개별주식", "지수", "장기채권"}
SAFE_ASSET_CLASSES = {"현금", "예금", "단기채권"}

st.set_page_config(page_title="포트폴리오 대시보드", page_icon="📊", layout="wide")


def num(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def fmt_krw(value: Any) -> str:
    return f"{num(value):,.0f}원"


def fmt_pct(value: Any) -> str:
    return f"{num(value) * 100:,.2f}%"


def fmt_num(value: Any, digits: int = 2) -> str:
    return f"{num(value):,.{digits}f}"


def text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def generated_at(value: str | None) -> str:
    if not value:
        return "-"
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return value


def rows(summary: dict[str, Any], key: str) -> list[dict[str, Any]]:
    data = summary.get("summary", {}).get(key, [])
    return data if isinstance(data, list) else []


def table(summary: dict[str, Any], key: str) -> pd.DataFrame:
    return pd.DataFrame(rows(summary, key))


def group_value(summary: dict[str, Any], key: str, label_col: str, label: str, value_col: str) -> float:
    for row in rows(summary, key):
        if text(row.get(label_col)) == label:
            return num(row.get(value_col))
    return 0.0


def group_pct(summary: dict[str, Any], key: str, label_col: str, label: str) -> float:
    return group_value(summary, key, label_col, label, "pct_total_assets")


def long_bond_pct(summary: dict[str, Any]) -> float:
    return group_pct(summary, "by_asset_class1", "asset_class1", "장기채권")


def short_bond_pct(summary: dict[str, Any]) -> float:
    return group_pct(summary, "by_asset_class1", "asset_class1", "단기채권")


def clean_holdings(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    numeric_cols = [
        "quantity",
        "current_price_foreign",
        "current_price_krw",
        "previous_close_foreign",
        "previous_close_krw",
        "day_change_price_foreign",
        "day_change_price_krw",
        "day_change_pct",
        "day_change_amount_krw",
        "market_value_krw",
        "pnl_rate",
        "pnl_amount_krw",
        "pe",
        "forward_pe",
        "pb",
        "roe",
        "debt_ratio",
        "dividend_yield",
    ]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def clean_history(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for col in result.columns:
        if col not in ["date", "generated_at"]:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result.dropna(subset=["date"]).sort_values("date") if "date" in result.columns else result


@st.cache_data(ttl=60, show_spinner=False)
def load_summary_local(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=60, show_spinner=False)
def load_holdings_local(path: str) -> pd.DataFrame:
    return clean_holdings(pd.read_csv(path))


@st.cache_data(ttl=60, show_spinner=False)
def load_history_local(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return clean_history(pd.read_csv(path))


@st.cache_data(ttl=60, show_spinner=False)
def load_summary_url(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


@st.cache_data(ttl=60, show_spinner=False)
def load_holdings_url(url: str) -> pd.DataFrame:
    return clean_holdings(pd.read_csv(url))


@st.cache_data(ttl=60, show_spinner=False)
def load_history_url(url: str) -> pd.DataFrame:
    if not url:
        return pd.DataFrame()
    return clean_history(pd.read_csv(url))


def value_delta(history_df: pd.DataFrame, value_col: str, days: int | None = None) -> tuple[float | None, float | None]:
    if history_df.empty or value_col not in history_df.columns:
        return None, None
    data = history_df.dropna(subset=[value_col]).sort_values("date")
    if len(data) < 2:
        return None, None
    latest = data.iloc[-1]
    if days is None:
        previous = data.iloc[-2]
    else:
        cutoff = latest["date"] - pd.Timedelta(days=days)
        candidates = data[data["date"] <= cutoff]
        previous = candidates.iloc[-1] if not candidates.empty else data.iloc[0]
    change = num(latest[value_col]) - num(previous[value_col])
    base = num(previous[value_col])
    return change, change / base if base else None


def show_metric(label: str, value: Any, delta_value: float | None = None, delta_pct: float | None = None) -> None:
    if delta_value is None:
        st.metric(label, fmt_krw(value))
    else:
        st.metric(label, fmt_krw(value), f"{fmt_krw(delta_value)} ({fmt_pct(delta_pct)})")


def filtered_holdings(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    with st.sidebar:
        st.header("필터")
        for col, label in [("owner", "소유자"), ("asset_class1", "자산구분1"), ("broker", "증권사/계좌기관")]:
            values = sorted([x for x in result.get(col, pd.Series(dtype=str)).dropna().unique()])
            selected = st.multiselect(label, values, default=values)
            if selected and col in result.columns:
                result = result[result[col].isin(selected)]
        keyword = st.text_input("종목명/티커 검색", "")
        if keyword.strip():
            pattern = re.escape(keyword.strip())
            name = result.get("name", pd.Series("", index=result.index)).fillna("").str.contains(pattern, case=False, regex=True)
            ticker = result.get("ticker", pd.Series("", index=result.index)).fillna("").str.contains(pattern, case=False, regex=True)
            result = result[name | ticker]
    return result


def show_percent_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return
    rename = {
        "asset_class1": "자산구분1",
        "asset_class2": "자산구분2",
        "investment_currency": "투자환종",
        "listing_currency": "상장통화",
        "hedge_flag": "환헤지",
        "owner": "소유자",
        "amount_krw_total": "총자산 기준 금액",
        "pct_total_assets": "총자산 비중",
    }
    display = df.drop(
        columns=["amount_krw_investment_base", "pct_investment_base"],
        errors="ignore",
    ).rename(columns=rename).copy()
    column_config = {}
    for col in ["총자산 기준 금액"]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").fillna(0)
            column_config[col] = st.column_config.NumberColumn(col, format="%.0f원")
    for col in ["총자산 비중"]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").fillna(0) * 100
            column_config[col] = st.column_config.NumberColumn(col, format="%.2f%%")
    st.dataframe(display, use_container_width=True, hide_index=True, column_config=column_config)


def show_asset_class1_pie(summary: dict[str, Any]) -> None:
    asset1_df = table(summary, "by_asset_class1")
    if asset1_df.empty or "asset_class1" not in asset1_df.columns or "pct_total_assets" not in asset1_df.columns:
        st.info("원형그래프를 표시할 데이터가 없습니다.")
        return
    chart_df = asset1_df[["asset_class1", "pct_total_assets"]].copy()
    chart_df["pct_total_assets"] = pd.to_numeric(chart_df["pct_total_assets"], errors="coerce").fillna(0)
    chart_df = chart_df[chart_df["pct_total_assets"] > 0]
    if chart_df.empty:
        st.info("원형그래프를 표시할 데이터가 없습니다.")
        return
    chart_df = chart_df.rename(columns={"asset_class1": "자산구분1", "pct_total_assets": "비중"})
    chart_df["비중"] = chart_df["비중"] * 100
    st.vega_lite_chart(
        chart_df,
        {
            "mark": {"type": "arc", "innerRadius": 35},
            "encoding": {
                "theta": {"field": "비중", "type": "quantitative"},
                "color": {"field": "자산구분1", "type": "nominal", "legend": {"orient": "bottom"}},
                "tooltip": [
                    {"field": "자산구분1", "type": "nominal"},
                    {"field": "비중", "type": "quantitative", "format": ".2f", "title": "비중(%)"},
                ],
            },
        },
        use_container_width=True,
    )


def show_checkpoint_metrics(summary: dict[str, Any]) -> None:
    asset_pct = {}
    asset1_df = table(summary, "by_asset_class1")
    if not asset1_df.empty:
        for _, row in asset1_df.iterrows():
            asset_pct[text(row.get("asset_class1"))] = num(row.get("pct_total_assets"))
    st.subheader("핵심 비중 체크포인트 · 총자산 기준")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("주식+지수", fmt_pct(asset_pct.get("개별주식", 0) + asset_pct.get("지수", 0)))
    r2.metric("장기채권", fmt_pct(long_bond_pct(summary)))
    r3.metric("단기채권", fmt_pct(short_bond_pct(summary)))
    r4.metric("현금", fmt_pct(asset_pct.get("현금", 0)))
    r5.metric("금/헷지", fmt_pct(asset_pct.get("헷지", 0)))


def show_risk_safe_allocation(summary: dict[str, Any]) -> None:
    asset_rows = rows(summary, "by_asset_class1")
    total_asset = num(summary.get("totals", {}).get("total_asset_krw"))

    risk_amount = sum(
        num(row.get("amount_krw_total"))
        for row in asset_rows
        if text(row.get("asset_class1")) in RISK_ASSET_CLASSES
    )
    safe_amount = sum(
        num(row.get("amount_krw_total"))
        for row in asset_rows
        if text(row.get("asset_class1")) in SAFE_ASSET_CLASSES
    )
    other_amount = max(total_asset - risk_amount - safe_amount, 0)

    risk_pct = risk_amount / total_asset if total_asset else 0
    safe_pct = safe_amount / total_asset if total_asset else 0
    other_pct = other_amount / total_asset if total_asset else 0

    st.subheader("위험자산·안전자산 비중")
    metric_cols = st.columns(3 if other_pct > 0.00005 else 2)
    metric_cols[0].metric("위험자산", fmt_pct(risk_pct))
    metric_cols[0].caption(fmt_krw(risk_amount))
    metric_cols[1].metric("안전자산", fmt_pct(safe_pct))
    metric_cols[1].caption(fmt_krw(safe_amount))
    if other_pct > 0.00005:
        metric_cols[2].metric("기타·미분류", fmt_pct(other_pct))
        metric_cols[2].caption(fmt_krw(other_amount))

    st.caption("위험자산: 개별주식·지수·장기채권 / 안전자산: 현금·예금·단기채권 / 기타·미분류: 금(헷지)")


def show_history(history_df: pd.DataFrame) -> None:
    st.subheader("포트폴리오 추이")
    if history_df.empty:
        st.info("아직 portfolio_history.csv가 없습니다. 로컬 자동화 패치를 적용한 뒤 업로드를 한 번 실행하면 추이가 표시됩니다.")
        return
    chart_cols = [c for c in ["total_asset_krw"] if c in history_df.columns]
    chart_window = history_df.sort_values("date").tail(60).copy()
    if chart_cols:
        chart_window["date"] = pd.to_datetime(chart_window["date"], errors="coerce")
        chart_window = chart_window.dropna(subset=["date"])
        chart_values = chart_window[chart_cols].apply(pd.to_numeric, errors="coerce")
        min_value = float(chart_values.min().min())
        max_value = float(chart_values.max().max())
        span = max_value - min_value
        padding = max(span * 0.15, max(abs(max_value) * 0.003, 100000))
        y_min = min_value - padding
        y_max = max_value + padding
        chart_data = chart_window[["date"] + chart_cols].melt("date", var_name="series", value_name="amount_krw")
        chart_data["series"] = chart_data["series"].map(
            {
                "total_asset_krw": "총자산",
            }
        ).fillna(chart_data["series"])
        st.vega_lite_chart(
            chart_data,
            {
                "mark": {"type": "line", "point": True, "strokeWidth": 3},
                "encoding": {
                    "x": {
                        "field": "date",
                        "type": "temporal",
                        "axis": {
                            "title": "날짜",
                            "format": "%Y-%m-%d",
                            "labelAngle": -35,
                            "tickCount": min(len(chart_window), 8),
                        },
                    },
                    "y": {
                        "field": "amount_krw",
                        "type": "quantitative",
                        "scale": {"domain": [y_min, y_max], "zero": False},
                        "axis": {"title": "금액(원)", "format": ",.0f"},
                    },
                    "color": {
                        "field": "series",
                        "type": "nominal",
                        "title": "",
                        "scale": {"range": ["#0f766e"]},
                    },
                    "tooltip": [
                        {"field": "date", "type": "temporal", "title": "날짜", "format": "%Y-%m-%d"},
                        {"field": "series", "type": "nominal", "title": "구분"},
                        {"field": "amount_krw", "type": "quantitative", "title": "금액", "format": ",.0f"},
                    ],
                },
                "height": 420,
            },
            use_container_width=True,
        )
    display = history_df.tail(30).drop(
        columns=[c for c in history_df.columns if "investment_base" in c],
        errors="ignore",
    ).copy()
    for col in [c for c in display.columns if c.endswith("pct") or c.endswith("_pct")]:
        display[col] = display[col].map(fmt_pct)
    for col in [c for c in display.columns if c.endswith("krw") or c.endswith("_krw")]:
        display[col] = display[col].map(fmt_krw)
    st.dataframe(display.sort_values("date", ascending=False), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 투자기준 (2026-10-13 확정) — 기준 변경 시 아래 상수만 수정
# ---------------------------------------------------------------------------
POLICY_DATE = "2026-10-13"
POLICY_TARGET_RETURN = 0.067  # 서울 아파트 장기 평균 상승률 (KB 시계열 1986~2023)

# 자산구분1 → 정책 버킷
POLICY_BUCKET_MAP = {
    "지수": "주식",
    "개별주식": "주식",
    "장기채권": "장기채",
    "헷지": "금",
    "단기채권": "단기채·현금·예금",
    "현금": "단기채·현금·예금",
    "예금": "단기채·현금·예금",
}
# (버킷, 목표, 하한, 상한) — 투자기준 총액 대비
POLICY_ALLOCATION = [
    ("주식", 0.70, 0.65, 0.75),
    ("장기채", 0.10, 0.05, 0.10),
    ("금", 0.10, 0.07, 0.13),
    ("단기채·현금·예금", 0.10, 0.10, 0.15),
]
# 주식 내부 (주식 = 100)
POLICY_EQUITY_SPLIT = [
    ("지수", 0.80, 0.75, 0.85),
    ("개별주식", 0.20, 0.15, 0.25),
]
POLICY_EQUAL_WEIGHT_CLASS2 = "S&P500동일가중"
POLICY_EQUAL_WEIGHT_MAX_OF_EQUITY = 0.20
POLICY_USD_MIN = 0.50
POLICY_SINGLE_STOCK_MAX = 0.10
POLICY_SINGLE_STOCK_REVIEW_LOSS = -0.30
# 투자기준에서 제외할 행: 이름/자산구분2에 아래 키워드 포함 (주택청약저축)
POLICY_EXCLUDE_KEYWORDS = ["청약"]


def policy_base_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """투자기준 대상 행: 주택청약 제외, 적금 포함 (엑셀의 include 플래그와 무관하게 판정)."""
    result = df.copy()
    result["market_value_krw"] = pd.to_numeric(result.get("market_value_krw"), errors="coerce").fillna(0)
    label = (
        result.get("name", pd.Series("", index=result.index)).fillna("").astype(str)
        + " "
        + result.get("asset_class2", pd.Series("", index=result.index)).fillna("").astype(str)
    )
    excluded = label.apply(lambda s: any(k in s for k in POLICY_EXCLUDE_KEYWORDS))
    result = result[~excluded & (result["market_value_krw"] != 0)].copy()
    result["policy_bucket"] = result["asset_class1"].map(POLICY_BUCKET_MAP).fillna("미분류")
    return result


def policy_status(current: float, low: float, high: float) -> str:
    if current < low - 1e-9:
        return "🔻 하한 미달"
    if current > high + 1e-9:
        return "🔺 상한 초과"
    return "✅ 범위 내"


def policy_target_table(amounts: dict[str, float], base: float, spec: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    records = []
    for name, target, low, high in spec:
        current = amounts.get(name, 0.0)
        pct = current / base if base else 0.0
        target_amount = base * target
        if pct < low:
            band_adjust = base * low - current
        elif pct > high:
            band_adjust = base * high - current
        else:
            band_adjust = 0.0
        records.append(
            {
                "구분": name,
                "현재 금액": current,
                "현재 비중": pct * 100,
                "목표 비중": target * 100,
                "허용범위": f"{low * 100:.0f}~{high * 100:.0f}%",
                "목표 금액": target_amount,
                "목표까지 조정액": target_amount - current,
                "범위 복귀 최소 조정액": band_adjust,
                "상태": policy_status(pct, low, high),
            }
        )
    return pd.DataFrame(records)


POLICY_TABLE_CONFIG = {
    "현재 금액": st.column_config.NumberColumn("현재 금액", format="%,.0f원"),
    "현재 비중": st.column_config.NumberColumn("현재 비중", format="%.1f%%"),
    "목표 비중": st.column_config.NumberColumn("목표 비중", format="%.0f%%"),
    "목표 금액": st.column_config.NumberColumn("목표 금액", format="%,.0f원"),
    "목표까지 조정액": st.column_config.NumberColumn("목표까지 조정액 (+매수 / −매도)", format="%+,.0f원"),
    "범위 복귀 최소 조정액": st.column_config.NumberColumn("범위 복귀 최소 조정액", format="%+,.0f원"),
}


def show_policy_rules() -> None:
    with st.expander(f"📋 투자기준 원문 ({POLICY_DATE} 확정)", expanded=False):
        st.markdown(
            f"""
**전제** · 투자기간 10년(2026.10~2036), 10년 후 서울 내집마련 · 출금 없음 · 월 적립 200만원(적금 100 + 현금 100)
· 비상금 별도 고려 안 함 · 계좌: ISA 2개(명의별) + 해외직접투자 1개

**목표수익률** · 연 {POLICY_TARGET_RETURN * 100:.1f}% (서울 아파트 장기 평균, KB 시계열 1986~2023) · 매년 서울 아파트 매매가격지수(한국부동산원)와 비교

**투자기준 총액** · 투자자산 + 적금 포함 / 주택청약저축 제외

**자산 목표비중** · 주식 70%(65~75) · 장기채 10%(5~10) · 금 10%(7~13) · 단기채·현금·예금 10%(10~15)
· 고물가·금리상승 국면: 장기채 하단(5%), 단기채·물가채 상단(15%)으로 기울일 수 있음

**주식 내부** · 지수 80%(75~85) / 개별주식 20%(15~25) · 동일가중은 주식의 20% 이내(하한 없음), 신규 지수 매수는 시가총액가중으로 · 미국 외 지역 지수 미편입

**리밸런싱** · 허용범위 이탈 시 조정 · 월 적립금은 부족 자산에 우선 투입 · 주식 70% 이행은 6~12개월 분할
· 목표비중은 시황으로 바꾸지 않고 연 1회(가을, J.P. Morgan LTCMA 발표 후) 재검토

**통화** · 달러 노출 투자기준의 {POLICY_USD_MIN * 100:.0f}% 이상 (환헤지 ETF는 원화로 계산)

**개별주식** · 종목당 투자기준의 {POLICY_SINGLE_STOCK_MAX * 100:.0f}% 초과 시 리밸런싱 검토 · {POLICY_SINGLE_STOCK_REVIEW_LOSS * 100:.0f}% 도달 시 투자 논리 재검토(자동 매도 아님)

**지수 ETF** · 손절 기준 없음

**설정하지 않은 항목** · 목표 주택·필요 자기자본 · 주택 구입 전 위험자산 축소 일정 · 비상금
"""
        )


def show_policy_tab(holdings: pd.DataFrame) -> None:
    base_df = policy_base_holdings(holdings)
    base = float(base_df["market_value_krw"].sum())
    if base <= 0:
        st.info("투자기준 금액을 계산할 데이터가 없습니다.")
        return

    st.subheader("투자기준 점검")
    st.caption(
        "투자기준 총액 = 전체 보유자산 − 주택청약저축 (적금 포함). 보유내역 필터와 무관하게 전체 행 기준으로 계산합니다. "
        "조정액: + 매수 필요 / − 매도 필요"
    )
    show_policy_rules()

    bucket_amounts = base_df.groupby("policy_bucket")["market_value_krw"].sum().to_dict()
    equity_df = base_df[base_df["policy_bucket"] == "주식"]
    equity_total = float(equity_df["market_value_krw"].sum())
    equity_amounts = equity_df.groupby("asset_class1")["market_value_krw"].sum().to_dict()
    equal_weight = float(equity_df.loc[equity_df["asset_class2"] == POLICY_EQUAL_WEIGHT_CLASS2, "market_value_krw"].sum())
    usd_amount = float(base_df.loc[base_df["investment_currency"] == "달러", "market_value_krw"].sum())
    usd_pct = usd_amount / base

    alloc_df = policy_target_table(bucket_amounts, base, POLICY_ALLOCATION)
    equity_split_df = policy_target_table(equity_amounts, equity_total, POLICY_EQUITY_SPLIT)

    # 개별종목 (티커 기준 합산, 소유자 통합)
    stock_df = base_df[base_df["asset_class1"] == "개별주식"].copy()
    stock_df["purchase_amount_krw"] = pd.to_numeric(stock_df.get("purchase_amount_krw"), errors="coerce").fillna(0)
    stock_df["key"] = stock_df["ticker"].fillna(stock_df["name"])
    stocks = (
        stock_df.groupby("key")
        .agg(name=("name", "first"), market_value_krw=("market_value_krw", "sum"), purchase_amount_krw=("purchase_amount_krw", "sum"))
        .reset_index()
    )
    stocks["weight"] = stocks["market_value_krw"] / base
    stocks["pnl_rate"] = stocks.apply(
        lambda r: r["market_value_krw"] / r["purchase_amount_krw"] - 1 if r["purchase_amount_krw"] else 0.0, axis=1
    )

    # 상단 요약 지표
    alerts = int((alloc_df["상태"] != "✅ 범위 내").sum()) + int((equity_split_df["상태"] != "✅ 범위 내").sum())
    alerts += int(equal_weight / equity_total > POLICY_EQUAL_WEIGHT_MAX_OF_EQUITY) if equity_total else 0
    alerts += int(usd_pct < POLICY_USD_MIN)
    alerts += int((stocks["weight"] > POLICY_SINGLE_STOCK_MAX).sum())
    alerts += int((stocks["pnl_rate"] <= POLICY_SINGLE_STOCK_REVIEW_LOSS).sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("투자기준 총액", fmt_krw(base))
    k2.metric("주식 비중 (목표 70%)", fmt_pct(bucket_amounts.get("주식", 0) / base))
    k3.metric("달러 노출 (기준 ≥50%)", fmt_pct(usd_pct))
    k4.metric("점검 필요 항목", f"{alerts}건")

    st.markdown("#### ① 자산 목표비중 · 투자기준 총액 대비")
    st.dataframe(alloc_df, use_container_width=True, hide_index=True, column_config=POLICY_TABLE_CONFIG)
    st.caption(
        "목표까지 조정액: 목표비중을 정확히 맞출 때의 금액 / 범위 복귀 최소 조정액: 허용범위 경계까지만 맞출 때의 금액(범위 내면 0). "
        "주식 70% 이행은 6~12개월 분할 매수."
    )

    st.markdown("#### ② 주식 내부 비중 · 주식 합계 대비")
    st.dataframe(equity_split_df, use_container_width=True, hide_index=True, column_config=POLICY_TABLE_CONFIG)
    ew_pct = equal_weight / equity_total if equity_total else 0.0
    target_equity = base * POLICY_ALLOCATION[0][1]
    stock_at_target = equity_amounts.get("개별주식", 0.0) / target_equity if target_equity else 0.0
    st.caption(
        f"현재 주식 합계 기준 수치입니다. 주식이 목표 {POLICY_ALLOCATION[0][1] * 100:.0f}%({fmt_krw(target_equity)})에 도달하면, "
        f"지수 추가 매수만으로 개별주식 비중이 {stock_at_target * 100:.1f}%가 됩니다 — 주식 확대 중에는 개별주식 매도보다 지수 매수로 맞추는 것을 우선."
    )
    st.caption(
        f"동일가중(S&P500동일가중): 주식의 {ew_pct * 100:.1f}% · 상한 {POLICY_EQUAL_WEIGHT_MAX_OF_EQUITY * 100:.0f}% "
        f"{'✅' if ew_pct <= POLICY_EQUAL_WEIGHT_MAX_OF_EQUITY else '🔺 상한 초과'} — 시가총액가중과의 비율은 고정하지 않음, 신규 지수 매수는 시가총액가중으로."
    )

    st.markdown("#### ③ 규칙 점검")
    checks = [
        {
            "규칙": "달러 노출 ≥ 50% (환헤지 ETF는 원화)",
            "현재": f"{usd_pct * 100:.1f}% ({fmt_krw(usd_amount)})",
            "상태": "✅ 충족" if usd_pct >= POLICY_USD_MIN else "🔻 미달",
        },
        {
            "규칙": "동일가중 ≤ 주식의 20%",
            "현재": f"{ew_pct * 100:.1f}% ({fmt_krw(equal_weight)})",
            "상태": "✅ 충족" if ew_pct <= POLICY_EQUAL_WEIGHT_MAX_OF_EQUITY else "🔺 초과",
        },
    ]
    st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)

    st.markdown("#### ④ 개별종목 점검 · 종목당 10% 초과 시 리밸런싱 검토 / −30% 도달 시 투자논리 재검토")
    if stocks.empty:
        st.info("개별주식 보유 내역이 없습니다.")
    else:
        stock_view = stocks.sort_values("market_value_krw", ascending=False).copy()
        stock_view["비중 점검"] = stock_view["weight"].apply(lambda w: "🔺 10% 초과 · 리밸런싱 검토" if w > POLICY_SINGLE_STOCK_MAX else "✅")
        stock_view["손실 점검"] = stock_view["pnl_rate"].apply(
            lambda p: "⚠️ −30% 도달 · 투자논리 재검토" if p <= POLICY_SINGLE_STOCK_REVIEW_LOSS else "✅"
        )
        stock_view = stock_view.rename(
            columns={"key": "티커", "name": "종목명", "market_value_krw": "평가금액", "weight": "투자기준 비중", "pnl_rate": "수익률"}
        )[["티커", "종목명", "평가금액", "투자기준 비중", "수익률", "비중 점검", "손실 점검"]]
        stock_view["투자기준 비중"] = stock_view["투자기준 비중"] * 100
        stock_view["수익률"] = stock_view["수익률"] * 100
        st.dataframe(
            stock_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "평가금액": st.column_config.NumberColumn("평가금액", format="%,.0f원"),
                "투자기준 비중": st.column_config.NumberColumn("투자기준 비중", format="%.2f%%"),
                "수익률": st.column_config.NumberColumn("수익률(합산)", format="%+.1f%%"),
            },
        )
        st.caption("같은 티커는 소유자·계좌를 합산합니다. 지수 ETF는 손절 기준 없음.")

    with st.expander("투자기준 포함 행 상세"):
        detail = base_df[["owner", "account", "policy_bucket", "asset_class1", "asset_class2", "investment_currency", "name", "market_value_krw"]].rename(
            columns={
                "owner": "소유자",
                "account": "계좌",
                "policy_bucket": "정책 버킷",
                "asset_class1": "자산구분1",
                "asset_class2": "자산구분2",
                "investment_currency": "투자환종",
                "name": "종목명",
                "market_value_krw": "평가금액",
            }
        )
        st.dataframe(
            detail.sort_values(["정책 버킷", "평가금액"], ascending=[True, False]),
            use_container_width=True,
            hide_index=True,
            column_config={"평가금액": st.column_config.NumberColumn("평가금액", format="%,.0f원")},
        )


st.sidebar.title("데이터 소스")
source = st.sidebar.radio("읽기 방식", ["로컬 web_data", "URL 직접 입력"], index=0)

try:
    if source == "URL 직접 입력":
        summary_url = st.sidebar.text_input("portfolio_summary.json URL", "")
        holdings_url = st.sidebar.text_input("portfolio_holdings.csv URL", "")
        history_url = st.sidebar.text_input("portfolio_history.csv URL", "")
        if not summary_url or not holdings_url:
            st.warning("URL 방식은 JSON URL과 CSV URL을 모두 입력해야 합니다.")
            st.stop()
        summary_json = load_summary_url(summary_url)
        holdings_df = load_holdings_url(holdings_url)
        history_df = load_history_url(history_url)
    else:
        if not SUMMARY_PATH.exists() or not HOLDINGS_PATH.exists():
            st.error("web_data 폴더에 portfolio_summary.json / portfolio_holdings.csv 파일이 없습니다.")
            st.stop()
        summary_json = load_summary_local(str(SUMMARY_PATH))
        holdings_df = load_holdings_local(str(HOLDINGS_PATH))
        history_df = load_history_local(str(HISTORY_PATH))
except Exception as exc:
    st.error(f"데이터를 읽는 중 오류가 발생했습니다: {exc}")
    st.stop()

filtered_df = filtered_holdings(holdings_df)

totals = summary_json.get("totals", {})
fx = summary_json.get("fx", {})

st.title("포트폴리오 대시보드")
st.caption("포트폴리오.xlsx → Python 가격/환율 업데이트 → web_data CSV·JSON 생성 → Streamlit 표시")
st.write(f"기준 파일: `{summary_json.get('source_file', '-')}` / 생성시각: `{generated_at(summary_json.get('generated_at'))}`")

st.caption("대시보드의 모든 비중과 체크포인트는 총자산 기준입니다.")

prev_total_delta, prev_total_delta_pct = value_delta(history_df, "total_asset_krw")
thirty_total_delta, thirty_total_delta_pct = value_delta(history_df, "total_asset_krw", days=30)
m1, m2, m3 = st.columns(3)
with m1:
    show_metric("총자산", totals.get("total_asset_krw"), prev_total_delta, prev_total_delta_pct)
with m2:
    show_metric("최근 30일 총자산", totals.get("total_asset_krw"), thirty_total_delta, thirty_total_delta_pct)
with m3:
    st.metric("보유행 수", f"{int(num(totals.get('holding_count'))):,}개")

f1, f2, f3 = st.columns(3)
f1.metric("USD/KRW", fmt_num(fx.get("usd_krw_sell_rate"), 2))
f2.metric("JPY 100엔/KRW", fmt_num(fx.get("jpy_100_krw_sell_rate"), 2))
f3.metric("JPY 1엔/KRW", fmt_num(fx.get("jpy_1_krw_sell_rate"), 4))

st.divider()

tab_summary, tab_policy, tab_holdings, tab_history = st.tabs(["요약", "투자기준", "보유내역", "추이"])

with tab_policy:
    show_policy_tab(holdings_df)

with tab_summary:
    show_checkpoint_metrics(summary_json)
    show_risk_safe_allocation(summary_json)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("자산구분1 비중 원형그래프")
        show_asset_class1_pie(summary_json)
    with c2:
        st.subheader("자산구분1 비중표")
        show_percent_table(table(summary_json, "by_asset_class1"))
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("투자환종 비중")
        show_percent_table(table(summary_json, "by_investment_currency"))
    with c4:
        st.subheader("상장통화 비중")
        show_percent_table(table(summary_json, "by_listing_currency"))
    st.subheader("자산구분1·2 상세")
    show_percent_table(table(summary_json, "by_asset_class1_2"))

with tab_holdings:
    st.subheader("보유내역")
    st.write(f"필터 적용 후 {len(filtered_df):,}개 행")
    cols = [
        "owner",
        "broker",
        "account",
        "asset_class1",
        "asset_class2",
        "investment_currency",
        "listing_currency",
        "hedge_flag",
        "ticker",
        "name",
        "quantity",
        "current_price_foreign",
        "current_price_krw",
        "previous_close_foreign",
        "previous_close_krw",
        "day_change_price_foreign",
        "day_change_price_krw",
        "day_change_pct",
        "day_change_amount_krw",
        "market_value_krw",
        "pnl_rate",
        "pnl_amount_krw",
        "pe",
        "forward_pe",
        "pb",
        "roe",
        "debt_ratio",
        "dividend_yield",
    ]
    cols = [c for c in cols if c in filtered_df.columns]
    display_df = filtered_df[cols].copy()
    if "pnl_rate" in display_df.columns:
        display_df["pnl_rate_display"] = display_df["pnl_rate"] * 100
        display_df = display_df.drop(columns=["pnl_rate"])
    if "day_change_pct" in display_df.columns:
        display_df["day_change_pct_display"] = display_df["day_change_pct"] * 100
        display_df = display_df.drop(columns=["day_change_pct"])
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.download_button("보유내역 CSV 다운로드", filtered_df.to_csv(index=False).encode("utf-8-sig"), "portfolio_holdings.csv", "text/csv")
    st.download_button("요약 JSON 다운로드", json.dumps(summary_json, ensure_ascii=False, indent=2).encode("utf-8"), "portfolio_summary.json", "application/json")

with tab_history:
    show_history(history_df)
    if not history_df.empty:
        st.download_button("추이 CSV 다운로드", history_df.to_csv(index=False).encode("utf-8-sig"), "portfolio_history.csv", "text/csv")
