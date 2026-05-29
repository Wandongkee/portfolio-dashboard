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


st.set_page_config(page_title="포트폴리오 대시보드", page_icon="📊", layout="wide")


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
    numeric_cols = {
        "row_no", "quantity", "avg_price_foreign", "avg_price_krw",
        "purchase_amount_foreign", "purchase_amount_krw",
        "current_price_foreign", "current_price_krw",
        "market_value_foreign", "market_value_krw",
        "pnl_rate", "pnl_amount_krw", "pe", "forward_pe",
        "pb", "roe", "debt_ratio", "dividend_yield",
    }
    for col in result.columns:
        if col in numeric_cols:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def safe_amount(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def summary_rows(summary: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows = summary.get("summary", {}).get(key, [])
    return rows if isinstance(rows, list) else []


def get_group_value(summary: dict[str, Any], table_key: str, label_col: str, label_value: str, value_col: str) -> float:
    for row in summary_rows(summary, table_key):
        if normalize_text(row.get(label_col)) == label_value:
            return safe_amount(row.get(value_col))
    return 0.0


def get_group_amount(summary: dict[str, Any], table_key: str, label_col: str, label_value: str) -> float:
    return get_group_value(summary, table_key, label_col, label_value, "amount_krw_investment_base")


def get_group_pct(summary: dict[str, Any], table_key: str, label_col: str, label_value: str) -> float:
    return get_group_value(summary, table_key, label_col, label_value, "pct_investment_base")


def get_group_sum(summary: dict[str, Any], table_key: str, label_col: str, label_values: list[str], value_col: str) -> float:
    targets = {normalize_text(x) for x in label_values}
    total = 0.0
    for row in summary_rows(summary, table_key):
        if normalize_text(row.get(label_col)) in targets:
            total += safe_amount(row.get(value_col))
    return total


def get_asset12_sum(summary: dict[str, Any], asset_class1_values: list[str], asset_class2_values: list[str], value_col: str) -> float:
    class1_targets = {normalize_text(x) for x in asset_class1_values}
    class2_targets = {normalize_text(x) for x in asset_class2_values}
    total = 0.0
    for row in summary_rows(summary, "by_asset_class1_2"):
        if normalize_text(row.get("asset_class1")) in class1_targets and normalize_text(row.get("asset_class2")) in class2_targets:
            total += safe_amount(row.get(value_col))
    return total


def get_long_bond_amount(summary: dict[str, Any]) -> float:
    return get_group_sum(summary, "by_asset_class1", "asset_class1", ["장기채권"], "amount_krw_investment_base") + get_asset12_sum(
        summary, ["채권"], ["장기채", "장기채권"], "amount_krw_investment_base"
    )


def get_long_bond_pct(summary: dict[str, Any]) -> float:
    return get_group_sum(summary, "by_asset_class1", "asset_class1", ["장기채권"], "pct_investment_base") + get_asset12_sum(
        summary, ["채권"], ["장기채", "장기채권"], "pct_investment_base"
    )


def get_short_bond_amount(summary: dict[str, Any]) -> float:
    return get_group_sum(summary, "by_asset_class1", "asset_class1", ["단기채권"], "amount_krw_investment_base") + get_asset12_sum(
        summary, ["채권"], ["초단기", "단기채", "단기채권"], "amount_krw_investment_base"
    )


def get_short_bond_pct(summary: dict[str, Any]) -> float:
    return get_group_sum(summary, "by_asset_class1", "asset_class1", ["단기채권"], "pct_investment_base") + get_asset12_sum(
        summary, ["채권"], ["초단기", "단기채", "단기채권"], "pct_investment_base"
    )


def get_table(summary: dict[str, Any], key: str) -> pd.DataFrame:
    return pd.DataFrame(summary_rows(summary, key))


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for col in ["현재비중", "목표비중", "비중", "상한", "하한", "기준", "현재"]:
        if col in display.columns:
            display[col] = display[col].map(fmt_pct)
    money_cols = [
        "총자산 기준 금액", "투자기준 금액", "현재금액", "목표금액", "추가필요액",
        "초과금액", "금액", "월 유입액", "주식·지수", "단기채권·현금",
        "장기채권", "기존자산 추가투입", "S&P500", "나스닥100/기술주",
        "환헤지형", "환노출형",
    ]
    for col in money_cols:
        if col in display.columns:
            display[col] = display[col].map(fmt_krw)
    return display


def display_money_pct_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return
    st.dataframe(format_table(df), use_container_width=True, hide_index=True)


def target_gap_row(name: str, current_amount: float, current_pct: float, target_pct: float, invest_base: float) -> dict[str, Any]:
    target_amount = invest_base * target_pct
    return {
        "구분": name,
        "현재비중": current_pct,
        "목표비중": target_pct,
        "현재금액": current_amount,
        "목표금액": target_amount,
        "추가필요액": max(target_amount - current_amount, 0.0),
        "초과금액": max(current_amount - target_amount, 0.0),
    }


@st.cache_data(ttl=60, show_spinner=False)
def load_summary_from_local(path_str: str) -> dict[str, Any]:
    with Path(path_str).open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=60, show_spinner=False)
def load_holdings_from_local(path_str: str) -> pd.DataFrame:
    return clean_numeric_columns(pd.read_csv(path_str))


@st.cache_data(ttl=60, show_spinner=False)
def load_summary_from_url(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


@st.cache_data(ttl=60, show_spinner=False)
def load_holdings_from_url(url: str) -> pd.DataFrame:
    return clean_numeric_columns(pd.read_csv(url))


def make_allocation_chart(df: pd.DataFrame, label_col: str, pct_col: str = "pct_investment_base") -> None:
    if df.empty or label_col not in df.columns or pct_col not in df.columns:
        st.info("표시할 데이터가 없습니다.")
        return
    chart_df = df[[label_col, pct_col]].copy()
    chart_df[pct_col] = pd.to_numeric(chart_df[pct_col], errors="coerce") * 100
    chart_df = chart_df.dropna(subset=[pct_col]).sort_values(pct_col, ascending=False).set_index(label_col)
    st.bar_chart(chart_df, horizontal=True)


def show_percent_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return
    display = df.copy()
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
    for col in ["총자산 기준 금액", "투자기준 금액"]:
        if col in display.columns:
            display[col] = display[col].map(fmt_krw)
    for col in ["총자산 비중", "투자기준 비중"]:
        if col in display.columns:
            display[col] = display[col].map(fmt_pct)
    st.dataframe(display, use_container_width=True, hide_index=True)


def download_buttons(holdings_df: pd.DataFrame, summary_json: dict[str, Any]) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="보유내역 CSV 다운로드",
            data=holdings_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="portfolio_holdings.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            label="요약 JSON 다운로드",
            data=json.dumps(summary_json, ensure_ascii=False, indent=2).encode("utf-8"),
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


def get_monthly_allocation(
    monthly_inflow: float,
    stock_index_pct: float,
    long_bond_pct: float,
    usd_pct: float,
    drawdown: float,
    us_30y_yield: float,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    stock_buy = monthly_inflow * 0.5
    short_cash_buy = monthly_inflow * 0.5
    long_bond_buy = 0.0
    mode = "평시 모드"
    note = "주식·지수 50%, 단기채권·현금 50%, 장기채권은 금리 재상승 전까지 보류합니다."

    if drawdown <= -0.10:
        stock_buy = monthly_inflow
        short_cash_buy = 0.0
        mode = "위험자산 보강 모드: -10% 이상 조정"
        note = "월 유입자금 전액을 주식·지수에 배정합니다."
    elif drawdown <= -0.07:
        stock_buy = monthly_inflow * 0.8
        short_cash_buy = monthly_inflow * 0.2
        mode = "위험자산 보강 모드: -7% 이상 조정"
        note = "월 유입자금의 80%를 주식·지수에 배정합니다."
    elif drawdown <= -0.05:
        stock_buy = monthly_inflow * 0.7
        short_cash_buy = monthly_inflow * 0.3
        mode = "위험자산 보강 모드: -5% 이상 조정"
        note = "월 유입자금의 70%를 주식·지수에 배정합니다."

    if stock_index_pct >= 0.48:
        stock_buy = 0.0
        short_cash_buy = monthly_inflow
        mode = "위험자산 상단 초과"
        note = "주식+지수가 48% 이상이면 신규 주식·지수 DCA를 중단하고 유입자금을 대기자산에 적립합니다."
    elif stock_index_pct >= 0.45 and drawdown > -0.07:
        stock_buy = monthly_inflow * 0.3
        short_cash_buy = monthly_inflow * 0.7
        mode = "위험자산 상단 접근"
        note = "주식+지수가 45% 이상이면 평시 주식 DCA를 30%로 낮춥니다."

    if us_30y_yield >= 5.35 and long_bond_pct < 0.08:
        long_bond_buy = monthly_inflow * 0.75
        stock_buy = min(stock_buy, monthly_inflow * 0.25)
        short_cash_buy = max(monthly_inflow - long_bond_buy - stock_buy, 0.0)
        mode = "장기채 금리 급등 모드"
        note = "미국 30년물 5.35% 이상이고 장기채 8% 미만이면 월 유입자금의 75%까지 장기채에 배정합니다."
    elif us_30y_yield >= 5.20 and long_bond_pct < 0.08:
        long_bond_buy = monthly_inflow * 0.5
        stock_buy = min(stock_buy, monthly_inflow * 0.35)
        short_cash_buy = max(monthly_inflow - long_bond_buy - stock_buy, 0.0)
        mode = "장기채 금리 재상승 모드"
        note = "미국 30년물 5.20% 이상이고 장기채 8% 미만이면 월 유입자금의 50%를 장기채에 배정합니다."
    elif us_30y_yield >= 5.10 and long_bond_pct < 0.07:
        long_bond_buy = monthly_inflow * 0.25
        stock_buy = min(stock_buy, monthly_inflow * 0.5)
        short_cash_buy = max(monthly_inflow - long_bond_buy - stock_buy, 0.0)
        mode = "장기채 7% 보강 모드"
        note = "미국 30년물 5.10% 이상이고 장기채 7% 미만이면 월 유입자금의 25%를 장기채에 배정합니다."

    if usd_pct >= 0.60:
        note += " 달러 투자환종이 60% 이상이면 환노출형 신규매수는 중단합니다."
    elif usd_pct >= 0.55:
        note += " 달러 투자환종이 55% 이상이면 환헤지형·원화상장 상품을 우선합니다."

    stock_buy = round(stock_buy / 10000) * 10000
    short_cash_buy = round(short_cash_buy / 10000) * 10000
    long_bond_buy = round(long_bond_buy / 10000) * 10000
    total = stock_buy + short_cash_buy + long_bond_buy
    if total != monthly_inflow:
        short_cash_buy += monthly_inflow - total

    rows = [{
        "구간": mode,
        "월 유입액": monthly_inflow,
        "주식·지수": stock_buy,
        "단기채권·현금": short_cash_buy,
        "장기채권": long_bond_buy,
        "비고": note,
    }]
    summary = {
        "mode": mode,
        "stock_buy": stock_buy,
        "short_cash_buy": short_cash_buy,
        "long_bond_buy": long_bond_buy,
        "note": note,
    }
    return pd.DataFrame(rows), summary


def show_rule_tab(summary_json: dict[str, Any]) -> None:
    totals = summary_json.get("totals", {})
    invest_base = safe_amount(totals.get("investment_base_krw"))

    stock_amount = get_group_amount(summary_json, "by_asset_class1", "asset_class1", "개별주식")
    stock_pct = get_group_pct(summary_json, "by_asset_class1", "asset_class1", "개별주식")
    index_amount = get_group_amount(summary_json, "by_asset_class1", "asset_class1", "지수")
    index_pct = get_group_pct(summary_json, "by_asset_class1", "asset_class1", "지수")
    stock_index_amount = stock_amount + index_amount
    stock_index_pct = stock_pct + index_pct

    cash_amount = get_group_amount(summary_json, "by_asset_class1", "asset_class1", "현금")
    cash_pct = get_group_pct(summary_json, "by_asset_class1", "asset_class1", "현금")
    gold_amount = get_group_amount(summary_json, "by_asset_class1", "asset_class1", "헷지")
    gold_pct = get_group_pct(summary_json, "by_asset_class1", "asset_class1", "헷지")
    usd_pct = get_group_pct(summary_json, "by_investment_currency", "investment_currency", "달러")

    long_bond_amount = get_long_bond_amount(summary_json)
    long_bond_pct = get_long_bond_pct(summary_json)
    short_bond_amount = get_short_bond_amount(summary_json)
    short_bond_pct = get_short_bond_pct(summary_json)
    liquidity_amount = cash_amount + short_bond_amount
    liquidity_pct = cash_pct + short_bond_pct

    st.subheader("운용 룰")
    st.caption("기준: 장기예금 제외 후 투자기준 총액. 월 신규 현금유입 200만원을 전제로 한 운용 기준표이며, 자동 매수 주문이 아닙니다.")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("주식+지수", fmt_pct(stock_index_pct))
    c2.metric("장기채권", fmt_pct(long_bond_pct))
    c3.metric("단기채권", fmt_pct(short_bond_pct))
    c4.metric("현금+단기채", fmt_pct(liquidity_pct))
    c5.metric("금/헷지", fmt_pct(gold_pct))
    c6.metric("달러 투자환종", fmt_pct(usd_pct))

    st.divider()
    st.markdown("#### 1. 목표비중")
    target_rows = [
        target_gap_row("주식+지수 기준 목표", stock_index_amount, stock_index_pct, 0.45, invest_base),
        target_gap_row("주식+지수 상단", stock_index_amount, stock_index_pct, 0.48, invest_base),
        target_gap_row("장기채권 기준 목표", long_bond_amount, long_bond_pct, 0.07, invest_base),
        target_gap_row("장기채권 상단", long_bond_amount, long_bond_pct, 0.08, invest_base),
        target_gap_row("현금+단기채 하단", liquidity_amount, liquidity_pct, 0.30, invest_base),
        target_gap_row("현금+단기채 기준", liquidity_amount, liquidity_pct, 0.40, invest_base),
        target_gap_row("금/헷지 상단", gold_amount, gold_pct, 0.12, invest_base),
        target_gap_row("개별주식 상단", stock_amount, stock_pct, 0.20, invest_base),
    ]
    display_money_pct_table(pd.DataFrame(target_rows))

    st.markdown("#### 2. 월 신규 유입자금 배분 룰")
    col1, col2, col3 = st.columns(3)
    with col1:
        monthly_inflow = st.number_input("월 신규 현금유입", min_value=0, max_value=20_000_000, value=2_000_000, step=100_000, format="%d")
    with col2:
        drawdown_pct_input = st.number_input("S&P500 또는 기준지수 고점 대비 하락률(%)", min_value=-80.0, max_value=30.0, value=0.0, step=0.5)
    with col3:
        us_30y_yield_input = st.number_input("미국 30년물 국채금리(%)", min_value=0.0, max_value=10.0, value=4.98, step=0.01, format="%.2f")

    drawdown = drawdown_pct_input / 100
    allocation_df, allocation_summary = get_monthly_allocation(
        float(monthly_inflow), stock_index_pct, long_bond_pct, usd_pct, drawdown, float(us_30y_yield_input)
    )
    display_money_pct_table(allocation_df)

    stock_buy = float(allocation_summary["stock_buy"])
    if stock_buy > 0:
        st.markdown("##### 주식·지수 DCA 내부 배분")
        stock_dca_df = pd.DataFrame([
            {"구분": "S&P500", "비중": 0.60, "금액": round(stock_buy * 0.60 / 10000) * 10000, "비고": "기본 지수 DCA의 중심"},
            {"구분": "나스닥100/기술주", "비중": 0.40, "금액": round(stock_buy * 0.40 / 10000) * 10000, "비고": "AI·기술주 노출 유지"},
        ])
        display_money_pct_table(stock_dca_df)

        st.markdown("##### 환헤지·환노출 배분")
        hedge_df = pd.DataFrame([
            {"구분": "환헤지형", "비중": 0.70, "금액": round(stock_buy * 0.70 / 10000) * 10000, "비고": "달러 투자환종이 높은 구간의 기본 우선순위"},
            {"구분": "환노출형", "비중": 0.30, "금액": round(stock_buy * 0.30 / 10000) * 10000, "비고": "달러 투자환종 60% 이상이면 신규매수 중단"},
        ])
        display_money_pct_table(hedge_df)
    else:
        st.info("현재 조건에서는 주식·지수 월 DCA가 중단 또는 축소되는 구간입니다.")

    st.markdown("#### 3. 하락률 트리거 매수 룰")
    trigger_rows = [
        {"트리거": "최근 고점 대비 -5%", "기준": -0.05, "월 유입자금 조정": "주식·지수 140만원 / 단기채·현금 60만원", "기존자산 추가투입": 0.0, "비고": "기존 단기채는 유지하고 월 유입자금만 공격적으로 조정"},
        {"트리거": "최근 고점 대비 -7%", "기준": -0.07, "월 유입자금 조정": "주식·지수 160만원 / 단기채·현금 40만원", "기존자산 추가투입": invest_base * 0.01, "비고": "기존 단기채 일부를 지수 ETF로 전환"},
        {"트리거": "최근 고점 대비 -10%", "기준": -0.10, "월 유입자금 조정": "주식·지수 200만원", "기존자산 추가투입": invest_base * 0.02, "비고": "추가 조정 시 2차 매수"},
        {"트리거": "최근 고점 대비 -15%", "기준": -0.15, "월 유입자금 조정": "주식·지수 200만원", "기존자산 추가투입": invest_base * 0.03, "비고": "강한 하락장 대응 매수"},
    ]
    trigger_df = pd.DataFrame(trigger_rows)
    trigger_display = trigger_df.copy()
    trigger_display["기준"] = trigger_display["기준"].map(fmt_pct)
    trigger_display["기존자산 추가투입"] = trigger_display["기존자산 추가투입"].map(fmt_krw)
    st.dataframe(trigger_display, use_container_width=True, hide_index=True)

    eligible = [row for row in trigger_rows if drawdown <= row["기준"]]
    if eligible:
        max_row = eligible[-1]
        st.success(f"현재 입력 하락률 기준 실행 가능 단계: {max_row['트리거']} / 기존자산 추가투입 {fmt_krw(max_row['기존자산 추가투입'])}")
    else:
        st.info("현재 입력 하락률 기준으로는 기존자산 추가투입 트리거가 발동되지 않았습니다.")

    st.markdown("#### 4. 장기채권 금리 재상승 트리거")
    bond_trigger_rows = [
        {"미국 30년물 금리": "5.10% 미만", "월 유입자금 배분": "장기채 0원", "조건": "추격매수 금지, 보유 유지"},
        {"미국 30년물 금리": "5.10~5.20%", "월 유입자금 배분": "장기채 50만원", "조건": "장기채 7% 미만일 때만"},
        {"미국 30년물 금리": "5.20~5.35%", "월 유입자금 배분": "장기채 100만원", "조건": "장기채 8% 미만일 때만"},
        {"미국 30년물 금리": "5.35% 이상", "월 유입자금 배분": "장기채 150만원", "조건": "장기채 8% 미만일 때만"},
        {"미국 30년물 금리": "무관", "월 유입자금 배분": "장기채 0원", "조건": "장기채 8% 도달 시 신규매수 중단"},
    ]
    st.dataframe(pd.DataFrame(bond_trigger_rows), use_container_width=True, hide_index=True)

    st.markdown("#### 5. 매수 중단·주의 조건")
    guardrails = [
        {"조건": "주식+지수 45% 초과", "현재": stock_index_pct, "판정": stock_index_pct > 0.45, "의미": "평시 주식 DCA 축소"},
        {"조건": "주식+지수 48% 초과", "현재": stock_index_pct, "판정": stock_index_pct > 0.48, "의미": "평시 주식 DCA 중단"},
        {"조건": "개별주식 20% 초과", "현재": stock_pct, "판정": stock_pct > 0.20, "의미": "개별주 신규매수 금지"},
        {"조건": "장기채권 8% 초과", "현재": long_bond_pct, "판정": long_bond_pct > 0.08, "의미": "장기채 신규매수 중단"},
        {"조건": "현금+단기채권 30% 미만", "현재": liquidity_pct, "판정": liquidity_pct < 0.30, "의미": "유동성 방어자산 부족"},
        {"조건": "금/헷지 12% 초과", "현재": gold_pct, "판정": gold_pct > 0.12, "의미": "금 신규매수 중단"},
        {"조건": "달러 투자환종 55% 초과", "현재": usd_pct, "판정": usd_pct > 0.55, "의미": "환헤지형·원화상장 상품 우선"},
        {"조건": "달러 투자환종 60% 초과", "현재": usd_pct, "판정": usd_pct > 0.60, "의미": "환노출형 신규매수 중단"},
    ]
    guard_df = pd.DataFrame(guardrails)
    guard_df["현재"] = guard_df["현재"].map(fmt_pct)
    guard_df["판정"] = guard_df["판정"].map(lambda x: "주의" if x else "정상")
    st.dataframe(guard_df, use_container_width=True, hide_index=True)

    st.markdown("#### 6. 현재 기준 요약 판단")
    notes = []
    if stock_index_pct < 0.45:
        notes.append("주식+지수는 기준 목표 45% 아래입니다. 평시에는 월 유입자금의 50%를 지수 ETF에 배정합니다.")
    elif stock_index_pct < 0.48:
        notes.append("주식+지수는 기준 목표를 넘었지만 상단 48% 이내입니다. 평시 DCA는 축소하고 조정장에서만 확대합니다.")
    else:
        notes.append("주식+지수가 48% 이상입니다. 평시 주식·지수 신규매수는 중단합니다.")

    if long_bond_pct < 0.07:
        notes.append("장기채는 7% 미만입니다. 자동 DCA가 아니라 미국 30년물 5.10% 이상 재상승 시에만 보강합니다.")
    elif long_bond_pct < 0.08:
        notes.append("장기채는 7~8% 관리 구간입니다. 현재 금리가 낮아진 경우 추격매수는 보류합니다.")
    else:
        notes.append("장기채가 8% 이상입니다. 장기채 신규매수는 중단합니다.")

    if liquidity_pct >= 0.30:
        notes.append("현금+단기채권 기준 유동성 방어자산은 충분합니다. 현금 단독 비중이 낮아도 별도 경고하지 않습니다.")
    else:
        notes.append("현금+단기채권이 30% 미만입니다. 신규 유입자금 일부를 대기자산으로 우선 배정합니다.")

    if gold_pct > 0.10:
        notes.append("금/헷지 비중이 10%를 넘습니다. 금 신규매수는 보류합니다.")

    if usd_pct > 0.55:
        notes.append("달러 투자환종이 55%를 넘습니다. 신규 매수는 환헤지형 또는 원화상장 상품을 우선합니다.")
    elif usd_pct > 0.50:
        notes.append("달러 투자환종이 50%를 넘습니다. 환노출형 증액은 제한적으로 접근합니다.")

    for note in notes:
        st.write(f"- {note}")


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
    st.dataframe(display_df, use_container_width=True, hide_index=True)
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
    long_bond_pct_risk = get_long_bond_pct(summary_json)
    short_bond_pct_risk = get_short_bond_pct(summary_json)
    cash_pct_risk = asset_pct.get("현금", 0)
    gold_pct_risk = asset_pct.get("헷지", 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("주식+지수", fmt_pct(stock_pct_risk))
    c2.metric("장기채권", fmt_pct(long_bond_pct_risk))
    c3.metric("단기채권", fmt_pct(short_bond_pct_risk))
    c4.metric("현금", fmt_pct(cash_pct_risk))
    c5.metric("금/헷지", fmt_pct(gold_pct_risk))

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
        )

    st.subheader("기준 금액 확인")
    st.write({"총자산": fmt_krw(total_asset), "투자기준 총액": fmt_krw(invest_base), "투자기준 제외금액": fmt_krw(excluded)})

with tab_rules:
    show_rule_tab(summary_json)
