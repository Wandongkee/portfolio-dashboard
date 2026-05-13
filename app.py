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
DEFAULT_DATA_DIR = BASE_DIR / "web_data"
DEFAULT_SUMMARY_PATH = DEFAULT_DATA_DIR / "portfolio_summary.json"
DEFAULT_HOLDINGS_PATH = DEFAULT_DATA_DIR / "portfolio_holdings.csv"


st.set_page_config(
    page_title="포트폴리오 대시보드",
    page_icon="📊",
    layout="wide",
)


# -----------------------------
# Formatting helpers
# -----------------------------

def fmt_krw(value: Any) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):,.0f}원"
    except Exception:
        return "-"


def fmt_num(value: Any, digits: int = 2) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "-"


def fmt_pct(value: Any, digits: int = 2) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value) * 100:,.{digits}f}%"
    except Exception:
        return "-"


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def parse_generated_at(value: str | None) -> str:
    if not value:
        return "-"
    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return value


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in result.columns:
        if col in {
            "row_no",
            "quantity",
            "avg_price_foreign",
            "avg_price_krw",
            "purchase_amount_foreign",
            "purchase_amount_krw",
            "current_price_foreign",
            "current_price_krw",
            "market_value_foreign",
            "market_value_krw",
            "pnl_rate",
            "pnl_amount_krw",
            "pe",
            "forward_pe",
            "pb",
            "roe",
            "debt_ratio",
            "dividend_yield",
        }:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


# -----------------------------
# Data loading
# -----------------------------

@st.cache_data(ttl=300, show_spinner=False)
def load_summary_from_local(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=300, show_spinner=False)
def load_holdings_from_local(path_str: str) -> pd.DataFrame:
    return clean_numeric_columns(pd.read_csv(path_str))


@st.cache_data(ttl=300, show_spinner=False)
def load_summary_from_url(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=15) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


@st.cache_data(ttl=300, show_spinner=False)
def load_holdings_from_url(url: str) -> pd.DataFrame:
    return clean_numeric_columns(pd.read_csv(url))


def get_table(summary: dict[str, Any], key: str) -> pd.DataFrame:
    rows = summary.get("summary", {}).get(key, [])
    return pd.DataFrame(rows)


def make_allocation_chart(df: pd.DataFrame, label_col: str, pct_col: str = "pct_investment_base") -> None:
    if df.empty or label_col not in df.columns or pct_col not in df.columns:
        st.info("표시할 데이터가 없습니다.")
        return

    chart_df = df[[label_col, pct_col]].copy()
    chart_df[pct_col] = pd.to_numeric(chart_df[pct_col], errors="coerce") * 100
    chart_df = chart_df.dropna(subset=[pct_col]).sort_values(pct_col, ascending=False)
    chart_df = chart_df.set_index(label_col)
    st.bar_chart(chart_df, horizontal=True)


def show_percent_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return

    display = df.copy()
    for col in ["amount_krw_total", "amount_krw_investment_base"]:
        if col in display.columns:
            display[col] = display[col].map(fmt_krw)
    for col in ["pct_total_assets", "pct_investment_base"]:
        if col in display.columns:
            display[col] = display[col].map(fmt_pct)

    rename_map = {
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
    display = display.rename(columns=rename_map)
    st.dataframe(display, use_container_width=True, hide_index=True)


def download_buttons(holdings_df: pd.DataFrame, summary_json: dict[str, Any]) -> None:
    col1, col2 = st.columns(2)
    with col1:
        csv_data = holdings_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="보유내역 CSV 다운로드",
            data=csv_data,
            file_name="portfolio_holdings.csv",
            mime="text/csv",
        )
    with col2:
        json_data = json.dumps(summary_json, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            label="요약 JSON 다운로드",
            data=json_data,
            file_name="portfolio_summary.json",
            mime="application/json",
        )


def filter_holdings(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    with st.sidebar:
        st.header("필터")

        owner_values = sorted([x for x in result.get("owner", pd.Series(dtype=str)).dropna().unique()])
        selected_owner = st.multiselect("소유자", owner_values, default=owner_values)
        if selected_owner:
            result = result[result["owner"].isin(selected_owner)]

        asset_values = sorted([x for x in result.get("asset_class1", pd.Series(dtype=str)).dropna().unique()])
        selected_asset = st.multiselect("자산구분1", asset_values, default=asset_values)
        if selected_asset:
            result = result[result["asset_class1"].isin(selected_asset)]

        broker_values = sorted([x for x in result.get("broker", pd.Series(dtype=str)).dropna().unique()])
        selected_broker = st.multiselect("증권사/계좌기관", broker_values, default=broker_values)
        if selected_broker:
            result = result[result["broker"].isin(selected_broker)]

        search = st.text_input("종목명/티커 검색", "")
        if search.strip():
            pattern = re.escape(search.strip())
            mask = (
                result.get("name", pd.Series("", index=result.index)).fillna("").str.contains(pattern, case=False, regex=True)
                | result.get("ticker", pd.Series("", index=result.index)).fillna("").str.contains(pattern, case=False, regex=True)
            )
            result = result[mask]

    return result


# -----------------------------
# Sidebar data source settings
# -----------------------------

st.sidebar.title("데이터 소스")
data_source = st.sidebar.radio("읽기 방식", ["로컬 web_data", "URL 직접 입력"], index=0)

summary_json: dict[str, Any]
holdings_df: pd.DataFrame

try:
    if data_source == "URL 직접 입력":
        summary_url = st.sidebar.text_input("portfolio_summary.json URL", "")
        holdings_url = st.sidebar.text_input("portfolio_holdings.csv URL", "")
        if not summary_url or not holdings_url:
            st.warning("URL 방식은 JSON URL과 CSV URL을 모두 입력해야 합니다.")
            st.stop()
        summary_json = load_summary_from_url(summary_url)
        holdings_df = load_holdings_from_url(holdings_url)
    else:
        if not DEFAULT_SUMMARY_PATH.exists() or not DEFAULT_HOLDINGS_PATH.exists():
            st.error("web_data 폴더에 portfolio_summary.json / portfolio_holdings.csv 파일이 없습니다.")
            st.stop()
        summary_json = load_summary_from_local(str(DEFAULT_SUMMARY_PATH))
        holdings_df = load_holdings_from_local(str(DEFAULT_HOLDINGS_PATH))
except Exception as exc:
    st.error(f"데이터를 읽는 중 오류가 발생했습니다: {exc}")
    st.stop()

filtered_df = filter_holdings(holdings_df)


# -----------------------------
# Main page
# -----------------------------

st.title("포트폴리오 대시보드")
st.caption("포트폴리오.xlsx → Python 가격/환율 업데이트 → web_data CSV·JSON 생성 → Streamlit 표시 / dashboard fix v2")

updated_at = parse_generated_at(summary_json.get("generated_at"))
source_file = summary_json.get("source_file", "-")
st.write(f"기준 파일: `{source_file}` / 생성시각: `{updated_at}`")

if summary_json.get("method_note"):
    with st.expander("계산 기준"):
        st.write(summary_json["method_note"])


totals = summary_json.get("totals", {})
fx = summary_json.get("fx", {})

metric_cols = st.columns(4)
metric_cols[0].metric("총자산", fmt_krw(totals.get("total_asset_krw")))
metric_cols[1].metric("투자기준 총액", fmt_krw(totals.get("investment_base_krw")))
metric_cols[2].metric("투자기준 제외", fmt_krw(totals.get("excluded_from_investment_base_krw")))
metric_cols[3].metric("보유행 수", f"{int(totals.get('holding_count', 0)):,}개")

fx_cols = st.columns(3)
fx_cols[0].metric("USD/KRW", fmt_num(fx.get("usd_krw_sell_rate"), 2))
fx_cols[1].metric("JPY 100엔/KRW", fmt_num(fx.get("jpy_100_krw_sell_rate"), 2))
fx_cols[2].metric("JPY 1엔/KRW", fmt_num(fx.get("jpy_1_krw_sell_rate"), 4))

st.divider()

tab_summary, tab_holdings, tab_risk = st.tabs(["요약", "보유내역", "분석용 체크포인트"])

with tab_summary:
    c1, c2 = st.columns(2)
    by_asset1 = get_table(summary_json, "by_asset_class1")
    by_currency = get_table(summary_json, "by_investment_currency")
    by_listing_currency = get_table(summary_json, "by_listing_currency")
    by_hedge = get_table(summary_json, "by_hedge_flag")
    by_owner = get_table(summary_json, "by_owner")
    by_asset12 = get_table(summary_json, "by_asset_class1_2")

    with c1:
        st.subheader("자산구분1 비중")
        make_allocation_chart(by_asset1, "asset_class1")
        show_percent_table(by_asset1)

    with c2:
        st.subheader("투자환종 비중")
        make_allocation_chart(by_currency, "investment_currency")
        show_percent_table(by_currency)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("상장통화 비중")
        make_allocation_chart(by_listing_currency, "listing_currency")
        show_percent_table(by_listing_currency)

    with c4:
        st.subheader("환헤지 구분")
        make_allocation_chart(by_hedge, "hedge_flag")
        show_percent_table(by_hedge)

    st.subheader("자산구분1·2 상세")
    show_percent_table(by_asset12)

    st.subheader("소유자별")
    show_percent_table(by_owner)

with tab_holdings:
    st.subheader("보유내역")
    st.write(f"필터 적용 후 {len(filtered_df):,}개 행")

    display_cols = [
        "owner", "broker", "account", "asset_class1", "asset_class2",
        "investment_currency", "listing_currency", "hedge_flag", "ticker", "name",
        "quantity", "current_price_foreign", "current_price_krw",
        "market_value_krw", "pnl_rate", "pnl_amount_krw",
        "pe", "forward_pe", "pb", "roe", "debt_ratio", "dividend_yield",
        "include_in_investment_base",
    ]
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    display_df = filtered_df[display_cols].copy()
    if "pnl_rate" in display_df.columns:
        display_df["pnl_rate"] = pd.to_numeric(display_df["pnl_rate"], errors="coerce") * 100

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "owner": "소유자",
            "broker": "증권사",
            "account": "계좌",
            "asset_class1": "자산구분1",
            "asset_class2": "자산구분2",
            "investment_currency": "투자환종",
            "listing_currency": "상장통화",
            "hedge_flag": "환헤지",
            "ticker": "티커",
            "name": "종목명",
            "quantity": st.column_config.NumberColumn("수량", format="%.4f"),
            "current_price_foreign": st.column_config.NumberColumn("현재가 외화", format="%.4f"),
            "current_price_krw": st.column_config.NumberColumn("현재가 원화", format="%d원"),
            "market_value_krw": st.column_config.NumberColumn("원화평가액", format="%d원"),
            "pnl_rate": st.column_config.NumberColumn("수익률", format="%.2f%%"),
            "pnl_amount_krw": st.column_config.NumberColumn("평가손익", format="%d원"),
            "pe": st.column_config.NumberColumn("P/E", format="%.2f"),
            "forward_pe": st.column_config.NumberColumn("Forward P/E", format="%.2f"),
            "pb": st.column_config.NumberColumn("P/B", format="%.2f"),
            "roe": st.column_config.NumberColumn("ROE", format="%.4f"),
            "debt_ratio": st.column_config.NumberColumn("부채비율", format="%.2f"),
            "dividend_yield": st.column_config.NumberColumn("배당수익률", format="%.4f"),
            "include_in_investment_base": "투자기준 포함",
        },
    )

    download_buttons(filtered_df, summary_json)

with tab_risk:
    st.subheader("분석용 체크포인트")

    invest_base = float(totals.get("investment_base_krw") or 0)
    total_asset = float(totals.get("total_asset_krw") or 0)
    excluded = float(totals.get("excluded_from_investment_base_krw") or 0)

    by_asset1 = get_table(summary_json, "by_asset_class1")
    asset_pct = {}
    if not by_asset1.empty:
        for _, row in by_asset1.iterrows():
            asset_pct[normalize_text(row.get("asset_class1"))] = float(row.get("pct_investment_base") or 0)

    stock_pct = asset_pct.get("개별주식", 0) + asset_pct.get("지수", 0)
    bond_pct = asset_pct.get("채권", 0)
    cash_pct = asset_pct.get("현금", 0)
    gold_pct = asset_pct.get("헷지", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("주식+지수", fmt_pct(stock_pct))
    c2.metric("채권", fmt_pct(bond_pct))
    c3.metric("현금", fmt_pct(cash_pct))
    c4.metric("금/헷지", fmt_pct(gold_pct))

    st.write("아래 표는 투자 판단용 원자료 확인을 위한 보조 지표입니다. 매수·매도 신호로 자동 해석하지 않습니다.")

    top_df = filtered_df.copy()
    if "market_value_krw" in top_df.columns:
        top_df = top_df[top_df["market_value_krw"].fillna(0) > 0]
        top_df = top_df.sort_values("market_value_krw", ascending=False).head(15)
        st.subheader("상위 보유종목")
        top_display_df = top_df[[c for c in ["asset_class1", "asset_class2", "ticker", "name", "market_value_krw", "pnl_rate"] if c in top_df.columns]].copy()
        if "pnl_rate" in top_display_df.columns:
            top_display_df["pnl_rate"] = pd.to_numeric(top_display_df["pnl_rate"], errors="coerce") * 100

        st.dataframe(
            top_display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "asset_class1": "자산구분1",
                "asset_class2": "자산구분2",
                "ticker": "티커",
                "name": "종목명",
                "market_value_krw": st.column_config.NumberColumn("원화평가액", format="%d원"),
                "pnl_rate": st.column_config.NumberColumn("수익률", format="%.2f%%"),
            },
        )

    st.subheader("기준 금액 확인")
    st.write(
        {
            "총자산": fmt_krw(total_asset),
            "투자기준 총액": fmt_krw(invest_base),
            "투자기준 제외금액": fmt_krw(excluded),
        }
    )
