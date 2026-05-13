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


def safe_amount(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def get_group_pct(summary: dict[str, Any], table_key: str, label_col: str, label_value: str) -> float:
    rows = summary.get("summary", {}).get(table_key, [])
    for row in rows:
        if normalize_text(row.get(label_col)) == label_value:
            return safe_amount(row.get("pct_investment_base"))
    return 0.0


def get_group_amount(summary: dict[str, Any], table_key: str, label_col: str, label_value: str) -> float:
    rows = summary.get("summary", {}).get(table_key, [])
    for row in rows:
        if normalize_text(row.get(label_col)) == label_value:
            return safe_amount(row.get("amount_krw_investment_base"))
    return 0.0


def get_asset12_amount(summary: dict[str, Any], asset_class1: str, asset_class2: str) -> float:
    rows = summary.get("summary", {}).get("by_asset_class1_2", [])
    for row in rows:
        if normalize_text(row.get("asset_class1")) == asset_class1 and normalize_text(row.get("asset_class2")) == asset_class2:
            return safe_amount(row.get("amount_krw_investment_base"))
    return 0.0


def get_asset12_pct(summary: dict[str, Any], asset_class1: str, asset_class2: str) -> float:
    rows = summary.get("summary", {}).get("by_asset_class1_2", [])
    for row in rows:
        if normalize_text(row.get("asset_class1")) == asset_class1 and normalize_text(row.get("asset_class2")) == asset_class2:
            return safe_amount(row.get("pct_investment_base"))
    return 0.0


def target_gap_row(name: str, current_amount: float, current_pct: float, target_pct: float, invest_base: float) -> dict[str, Any]:
    target_amount = invest_base * target_pct
    additional = max(target_amount - current_amount, 0.0)
    return {
        "구분": name,
        "현재비중": current_pct,
        "목표비중": target_pct,
        "현재금액": current_amount,
        "목표금액": target_amount,
        "추가필요액": additional,
    }


def display_money_pct_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return

    display = df.copy()
    for col in ["현재비중", "목표비중", "비중", "상한", "하한"]:
        if col in display.columns:
            display[col] = display[col].map(fmt_pct)

    for col in ["현재금액", "목표금액", "추가필요액", "금액", "월DCA", "장기채", "주식·지수", "실행금액"]:
        if col in display.columns:
            display[col] = display[col].map(fmt_krw)

    st.dataframe(display, use_container_width=True, hide_index=True)


# -----------------------------
# Data loading
# -----------------------------

@st.cache_data(ttl=60, show_spinner=False)
def load_summary_from_local(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=60, show_spinner=False)
def load_holdings_from_local(path_str: str) -> pd.DataFrame:
    return clean_numeric_columns(pd.read_csv(path_str))


@st.cache_data(ttl=60, show_spinner=False)
def load_summary_from_url(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=15) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


@st.cache_data(ttl=60, show_spinner=False)
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
        if selected_owner and "owner" in result.columns:
            result = result[result["owner"].isin(selected_owner)]

        asset_values = sorted([x for x in result.get("asset_class1", pd.Series(dtype=str)).dropna().unique()])
        selected_asset = st.multiselect("자산구분1", asset_values, default=asset_values)
        if selected_asset and "asset_class1" in result.columns:
            result = result[result["asset_class1"].isin(selected_asset)]

        broker_values = sorted([x for x in result.get("broker", pd.Series(dtype=str)).dropna().unique()])
        selected_broker = st.multiselect("증권사/계좌기관", broker_values, default=broker_values)
        if selected_broker and "broker" in result.columns:
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


def show_rule_tab(summary_json: dict[str, Any]) -> None:
    totals = summary_json.get("totals", {})
    invest_base = safe_amount(totals.get("investment_base_krw"))

    stock_amount = get_group_amount(summary_json, "by_asset_class1", "asset_class1", "개별주식")
    stock_pct = get_group_pct(summary_json, "by_asset_class1", "asset_class1", "개별주식")
    index_amount = get_group_amount(summary_json, "by_asset_class1", "asset_class1", "지수")
    index_pct = get_group_pct(summary_json, "by_asset_class1", "asset_class1", "지수")
    stock_index_amount = stock_amount + index_amount
    stock_index_pct = stock_pct + index_pct

    bond_pct = get_group_pct(summary_json, "by_asset_class1", "asset_class1", "채권")
    cash_pct = get_group_pct(summary_json, "by_asset_class1", "asset_class1", "현금")
    gold_pct = get_group_pct(summary_json, "by_asset_class1", "asset_class1", "헷지")
    individual_stock_pct = stock_pct
    usd_pct = get_group_pct(summary_json, "by_investment_currency", "investment_currency", "달러")

    long_bond_amount = get_asset12_amount(summary_json, "채권", "장기채")
    long_bond_pct = get_asset12_pct(summary_json, "채권", "장기채")
    short_bond_pct = get_asset12_pct(summary_json, "채권", "초단기")

    st.subheader("신규 운용 룰")
    st.caption("기준: 장기예금 제외 후 투자기준 총액. 이 표는 자동 매수 주문이 아니라 운용 기준표입니다.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("주식+지수", fmt_pct(stock_index_pct))
    c2.metric("장기채", fmt_pct(long_bond_pct))
    c3.metric("현금", fmt_pct(cash_pct))
    c4.metric("달러 투자환종", fmt_pct(usd_pct))

    st.divider()

    st.markdown("#### 1. 목표비중 및 추가 필요금액")

    target_rows = [
        target_gap_row("주식+지수 1차 목표", stock_index_amount, stock_index_pct, 0.50, invest_base),
        target_gap_row("주식+지수 2차 목표", stock_index_amount, stock_index_pct, 0.55, invest_base),
        target_gap_row("주식+지수 최종 목표", stock_index_amount, stock_index_pct, 0.60, invest_base),
        target_gap_row("장기채 1차 목표", long_bond_amount, long_bond_pct, 0.05, invest_base),
        target_gap_row("장기채 2차 목표", long_bond_amount, long_bond_pct, 0.07, invest_base),
        target_gap_row("장기채 최종 목표", long_bond_amount, long_bond_pct, 0.10, invest_base),
    ]
    display_money_pct_table(pd.DataFrame(target_rows))

    st.markdown("#### 2. 기본 월 DCA 룰")

    raw_monthly_dca = invest_base * 0.005
    monthly_dca = round(raw_monthly_dca / 10000) * 10000

    if long_bond_pct < 0.07:
        phase = "장기채 7% 도달 전"
        bond_dca = min(500000, monthly_dca)
        stock_dca = max(monthly_dca - bond_dca, 0)
        phase_note = "장기채 비중을 우선 7%까지 끌어올리는 구간입니다."
    elif long_bond_pct < 0.10:
        phase = "장기채 7~10% 구간"
        bond_dca = round(monthly_dca * 0.2 / 10000) * 10000
        stock_dca = max(monthly_dca - bond_dca, 0)
        phase_note = "장기채는 속도를 줄이고 주식·지수 DCA 비중을 높이는 구간입니다."
    else:
        phase = "장기채 10% 도달 후"
        bond_dca = 0
        stock_dca = monthly_dca
        phase_note = "장기채 신규 DCA는 중단하고 주식·지수 중심으로 운용합니다."

    dca_df = pd.DataFrame(
        [
            {
                "구간": phase,
                "월DCA": monthly_dca,
                "장기채": bond_dca,
                "주식·지수": stock_dca,
                "비고": phase_note,
            }
        ]
    )
    display_money_pct_table(dca_df)
    st.caption(f"월 DCA 기준: 투자기준 총액의 0.5% = {fmt_krw(raw_monthly_dca)}, 실행 편의를 위해 1만 원 단위 반올림.")

    stock_dca_df = pd.DataFrame(
        [
            {"구분": "S&P500", "비중": 0.70, "비고": "기본 주식 DCA의 중심"},
            {"구분": "나스닥100/기술주 지수", "비중": 0.30, "비고": "AI·기술주 노출 유지"},
            {"구분": "한국/반도체 지수", "비중": 0.00, "비고": "급등 구간에서는 보류, 조정 시 재검토"},
        ]
    )
    display_money_pct_table(stock_dca_df)

    st.markdown("#### 3. 하락률 트리거 매수 룰")

    drawdown_pct_input = st.number_input(
        "S&P500 또는 기준지수의 최근 고점 대비 하락률을 입력하세요. 예: -7",
        min_value=-80.0,
        max_value=30.0,
        value=0.0,
        step=0.5,
    )
    drawdown = drawdown_pct_input / 100

    trigger_rows = [
        {"트리거": "최근 고점 대비 -5%", "기준": -0.05, "실행금액": invest_base * 0.01, "비고": "소규모 1차 매수"},
        {"트리거": "최근 고점 대비 -7%", "기준": -0.07, "실행금액": invest_base * 0.03, "비고": "본격 1차 하락 매수"},
        {"트리거": "최근 고점 대비 -10%", "기준": -0.10, "실행금액": invest_base * 0.04, "비고": "2차 하락 매수"},
        {"트리거": "최근 고점 대비 -15%", "기준": -0.15, "실행금액": invest_base * 0.05, "비고": "강한 하락장 대응 매수"},
    ]
    trigger_df = pd.DataFrame(trigger_rows)
    trigger_display = trigger_df.copy()
    trigger_display["기준"] = trigger_display["기준"].map(fmt_pct)
    trigger_display["실행금액"] = trigger_display["실행금액"].map(fmt_krw)
    st.dataframe(trigger_display[["트리거", "기준", "실행금액", "비고"]], use_container_width=True, hide_index=True)

    eligible = [row for row in trigger_rows if drawdown <= row["기준"]]
    if eligible:
        max_row = eligible[-1]
        st.success(f"현재 입력 하락률 기준 실행 가능 단계: {max_row['트리거']} / 기준 실행금액 {fmt_krw(max_row['실행금액'])}")
    else:
        st.info("현재 입력 하락률 기준으로는 하락률 트리거 매수 조건이 발동되지 않았습니다.")

    st.markdown("#### 4. 매수 중단·주의 조건")

    guardrails = [
        {"조건": "주식+지수 60% 초과", "현재": stock_index_pct, "판정": stock_index_pct > 0.60, "의미": "위험자산 추가매수 중단"},
        {"조건": "현금 10% 미만", "현재": cash_pct, "판정": cash_pct < 0.10, "의미": "현금 방어력 훼손"},
        {"조건": "달러 투자환종 60% 초과", "현재": usd_pct, "판정": usd_pct > 0.60, "의미": "환노출형 미국자산 매수 제한"},
        {"조건": "개별주식 30% 초과", "현재": individual_stock_pct, "판정": individual_stock_pct > 0.30, "의미": "개별주 신규매수 금지"},
        {"조건": "금 10% 초과", "현재": gold_pct, "판정": gold_pct > 0.10, "의미": "금 추가매수 보류"},
    ]
    guard_df = pd.DataFrame(guardrails)
    guard_df["현재"] = guard_df["현재"].map(fmt_pct)
    guard_df["판정"] = guard_df["판정"].map(lambda x: "주의" if x else "정상")
    st.dataframe(guard_df, use_container_width=True, hide_index=True)

    st.markdown("#### 5. 장기채 별도 분할매수 조건")
    bond_rule_df = pd.DataFrame(
        [
            {"단계": "2차 매수", "금액": "300만~400만 원", "조건": "미국 장기금리 재상승, 30년물 5% 부근 재접근, 또는 보유 장기채 추가 하락"},
            {"단계": "3차 매수", "금액": "300만~400만 원", "조건": "PCE·고용·유가 이벤트 이후 금리 급등 시"},
            {"단계": "4차 매수", "금액": "300만~400만 원", "조건": "금리 피크아웃 가능성 확인 시, 최종 10% 근접까지"},
            {"단계": "반등 시", "금액": "보류", "조건": "장기채가 먼저 반등하면 7~8%에서 일단 정지"},
        ]
    )
    st.dataframe(bond_rule_df, use_container_width=True, hide_index=True)

    st.markdown("#### 6. 현재 기준 요약 판단")
    notes = []
    if stock_index_pct < 0.50:
        notes.append("주식+지수는 1차 목표 50%까지 여지가 있습니다. 다만 급등 구간에서는 DCA와 하락률 트리거 중심으로 접근합니다.")
    elif stock_index_pct < 0.55:
        notes.append("주식+지수는 1차 목표를 충족한 상태입니다. 추가 확대는 조정 또는 금리 안정 확인 후 검토합니다.")
    elif stock_index_pct < 0.60:
        notes.append("주식+지수는 목표 상단에 접근 중입니다. 신규매수 속도를 낮춥니다.")
    else:
        notes.append("주식+지수가 60% 이상입니다. 신규 주식매수는 중단하는 기준입니다.")

    if long_bond_pct < 0.07:
        notes.append("장기채는 7%까지 우선 확대하는 구간입니다. 월 DCA에서 장기채 비중을 높게 유지합니다.")
    elif long_bond_pct < 0.10:
        notes.append("장기채는 7~10% 구간입니다. 장기채 DCA 속도를 낮추고 주식·지수 DCA를 늘립니다.")
    else:
        notes.append("장기채가 10%에 도달했습니다. 장기채 신규 DCA는 중단합니다.")

    if usd_pct > 0.50:
        notes.append("달러 투자환종 비중이 50%를 넘습니다. 신규 매수는 환헤지형 또는 원화상장 상품을 우선 검토합니다.")

    for note in notes:
        st.write(f"- {note}")


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
st.caption("포트폴리오.xlsx → Python 가격/환율 업데이트 → web_data CSV·JSON 생성 → Streamlit 표시")

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

tab_summary, tab_holdings, tab_risk, tab_rules = st.tabs(["요약", "보유내역", "분석용 체크포인트", "운용 룰"])

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
        display_df["pnl_rate_display"] = display_df["pnl_rate"] * 100
        display_df = display_df.drop(columns=["pnl_rate"])
        insert_at = display_cols.index("pnl_rate") if "pnl_rate" in display_cols else len(display_df.columns)
        cols = list(display_df.columns)
        if "pnl_rate_display" in cols:
            cols.remove("pnl_rate_display")
            cols.insert(min(insert_at, len(cols)), "pnl_rate_display")
            display_df = display_df[cols]

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
            "pnl_rate_display": st.column_config.NumberColumn("수익률", format="%.2f%%"),
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

    stock_pct_risk = asset_pct.get("개별주식", 0) + asset_pct.get("지수", 0)
    bond_pct_risk = asset_pct.get("채권", 0)
    cash_pct_risk = asset_pct.get("현금", 0)
    gold_pct_risk = asset_pct.get("헷지", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("주식+지수", fmt_pct(stock_pct_risk))
    c2.metric("채권", fmt_pct(bond_pct_risk))
    c3.metric("현금", fmt_pct(cash_pct_risk))
    c4.metric("금/헷지", fmt_pct(gold_pct_risk))

    st.write("아래 표는 투자 판단용 원자료 확인을 위한 보조 지표입니다. 매수·매도 신호로 자동 해석하지 않습니다.")

    top_df = filtered_df.copy()
    if "market_value_krw" in top_df.columns:
        top_df = top_df[top_df["market_value_krw"].fillna(0) > 0]
        top_df = top_df.sort_values("market_value_krw", ascending=False).head(15)
        if "pnl_rate" in top_df.columns:
            top_df["pnl_rate_display"] = top_df["pnl_rate"] * 100

        st.subheader("상위 보유종목")
        st.dataframe(
            top_df[[c for c in ["asset_class1", "asset_class2", "ticker", "name", "market_value_krw", "pnl_rate_display"] if c in top_df.columns]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "asset_class1": "자산구분1",
                "asset_class2": "자산구분2",
                "ticker": "티커",
                "name": "종목명",
                "market_value_krw": st.column_config.NumberColumn("원화평가액", format="%d원"),
                "pnl_rate_display": st.column_config.NumberColumn("수익률", format="%.2f%%"),
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

with tab_rules:
    show_rule_tab(summary_json)
