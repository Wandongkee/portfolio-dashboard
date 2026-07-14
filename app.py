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
    return group_value(summary, key, label_col, label, "pct_investment_base")


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
        "amount_krw_investment_base": "투자기준 금액",
        "pct_total_assets": "총자산 비중",
        "pct_investment_base": "투자기준 비중",
    }
    display = df.rename(columns=rename).copy()
    column_config = {}
    for col in ["총자산 기준 금액", "투자기준 금액"]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").fillna(0)
            column_config[col] = st.column_config.NumberColumn(col, format="%.0f원")
    for col in ["총자산 비중", "투자기준 비중"]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").fillna(0) * 100
            column_config[col] = st.column_config.NumberColumn(col, format="%.2f%%")
    st.dataframe(display, use_container_width=True, hide_index=True, column_config=column_config)


def show_asset_class1_pie(summary: dict[str, Any]) -> None:
    asset1_df = table(summary, "by_asset_class1")
    if asset1_df.empty or "asset_class1" not in asset1_df.columns or "pct_investment_base" not in asset1_df.columns:
        st.info("원형그래프를 표시할 데이터가 없습니다.")
        return
    chart_df = asset1_df[["asset_class1", "pct_investment_base"]].copy()
    chart_df["pct_investment_base"] = pd.to_numeric(chart_df["pct_investment_base"], errors="coerce").fillna(0)
    chart_df = chart_df[chart_df["pct_investment_base"] > 0]
    if chart_df.empty:
        st.info("원형그래프를 표시할 데이터가 없습니다.")
        return
    chart_df = chart_df.rename(columns={"asset_class1": "자산구분1", "pct_investment_base": "비중"})
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
            asset_pct[text(row.get("asset_class1"))] = num(row.get("pct_investment_base"))
    st.subheader("핵심 비중 체크포인트")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("주식+지수", fmt_pct(asset_pct.get("개별주식", 0) + asset_pct.get("지수", 0)))
    r2.metric("장기채권", fmt_pct(long_bond_pct(summary)))
    r3.metric("단기채권", fmt_pct(short_bond_pct(summary)))
    r4.metric("현금", fmt_pct(asset_pct.get("현금", 0)))
    r5.metric("금/헷지", fmt_pct(asset_pct.get("헷지", 0)))


def show_history(history_df: pd.DataFrame) -> None:
    st.subheader("포트폴리오 추이")
    if history_df.empty:
        st.info("아직 portfolio_history.csv가 없습니다. 로컬 자동화 패치를 적용한 뒤 업로드를 한 번 실행하면 추이가 표시됩니다.")
        return
    chart_cols = [c for c in ["total_asset_krw", "investment_base_krw"] if c in history_df.columns]
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
                "investment_base_krw": "투자기준 총액",
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
                        "scale": {"range": ["#0f766e", "#dc2626"]},
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
    display = history_df.tail(30).copy()
    for col in [c for c in display.columns if c.endswith("pct") or c.endswith("_pct")]:
        display[col] = display[col].map(fmt_pct)
    for col in [c for c in display.columns if c.endswith("krw") or c.endswith("_krw")]:
        display[col] = display[col].map(fmt_krw)
    st.dataframe(display.sort_values("date", ascending=False), use_container_width=True, hide_index=True)


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

if summary_json.get("method_note"):
    with st.expander("계산 기준"):
        st.write(summary_json["method_note"])

prev_total_delta, prev_total_delta_pct = value_delta(history_df, "total_asset_krw")
thirty_total_delta, thirty_total_delta_pct = value_delta(history_df, "total_asset_krw", days=30)
prev_base_delta, prev_base_delta_pct = value_delta(history_df, "investment_base_krw")

m1, m2, m3, m4 = st.columns(4)
with m1:
    show_metric("총자산", totals.get("total_asset_krw"), prev_total_delta, prev_total_delta_pct)
with m2:
    show_metric("투자기준 총액", totals.get("investment_base_krw"), prev_base_delta, prev_base_delta_pct)
with m3:
    show_metric("최근 30일 총자산", totals.get("total_asset_krw"), thirty_total_delta, thirty_total_delta_pct)
with m4:
    st.metric("보유행 수", f"{int(num(totals.get('holding_count'))):,}개")

f1, f2, f3 = st.columns(3)
f1.metric("USD/KRW", fmt_num(fx.get("usd_krw_sell_rate"), 2))
f2.metric("JPY 100엔/KRW", fmt_num(fx.get("jpy_100_krw_sell_rate"), 2))
f3.metric("JPY 1엔/KRW", fmt_num(fx.get("jpy_1_krw_sell_rate"), 4))

st.divider()

tab_summary, tab_holdings, tab_history = st.tabs(["요약", "보유내역", "추이"])

with tab_summary:
    show_checkpoint_metrics(summary_json)
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
        "include_in_investment_base",
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
