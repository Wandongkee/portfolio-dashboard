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


def group_amount(summary: dict[str, Any], key: str, label_col: str, label: str) -> float:
    return group_value(summary, key, label_col, label, "amount_krw_investment_base")


def group_pct(summary: dict[str, Any], key: str, label_col: str, label: str) -> float:
    return group_value(summary, key, label_col, label, "pct_investment_base")


def long_bond_pct(summary: dict[str, Any]) -> float:
    return group_pct(summary, "by_asset_class1", "asset_class1", "장기채권")


def short_bond_pct(summary: dict[str, Any]) -> float:
    return group_pct(summary, "by_asset_class1", "asset_class1", "단기채권")


def format_df(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for col in ["현재비중", "목표비중", "기준", "현재", "비중"]:
        if col in display.columns:
            display[col] = display[col].map(fmt_pct)
    for col in ["현재금액", "목표금액", "추가필요액", "초과금액", "월 유입액", "주식·지수", "단기채권·현금", "금액"]:
        if col in display.columns:
            display[col] = display[col].map(fmt_krw)
    return display


def show_money_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
    else:
        st.dataframe(format_df(df), use_container_width=True, hide_index=True)


def target_row(name: str, current_amount: float, current_pct: float, target_pct: float, base: float) -> dict[str, Any]:
    target_amount = base * target_pct
    return {
        "구분": name,
        "현재비중": current_pct,
        "목표비중": target_pct,
        "현재금액": current_amount,
        "목표금액": target_amount,
        "추가필요액": max(target_amount - current_amount, 0),
        "초과금액": max(current_amount - target_amount, 0),
    }


def clean_holdings(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    numeric_cols = ["quantity", "current_price_foreign", "current_price_krw", "market_value_krw", "pnl_rate", "pnl_amount_krw", "pe", "forward_pe", "pb", "roe", "debt_ratio", "dividend_yield"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


@st.cache_data(ttl=60, show_spinner=False)
def load_summary_local(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=60, show_spinner=False)
def load_holdings_local(path: str) -> pd.DataFrame:
    return clean_holdings(pd.read_csv(path))


@st.cache_data(ttl=60, show_spinner=False)
def load_summary_url(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


@st.cache_data(ttl=60, show_spinner=False)
def load_holdings_url(url: str) -> pd.DataFrame:
    return clean_holdings(pd.read_csv(url))


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
            mask = result.get("name", pd.Series("", index=result.index)).fillna("").str.contains(pattern, case=False, regex=True) | result.get("ticker", pd.Series("", index=result.index)).fillna("").str.contains(pattern, case=False, regex=True)
            result = result[mask]
    return result


def dca_allocation(monthly: float, stock_index_pct: float, drawdown: float) -> tuple[pd.DataFrame, float]:
    if stock_index_pct < 0.45:
        stock_ratio, mode = 0.60, "주식비중 회복 구간"
    elif stock_index_pct < 0.50:
        stock_ratio, mode = 0.50, "기본 DCA 구간"
    elif stock_index_pct < 0.52:
        stock_ratio, mode = 0.25, "감속 구간"
    else:
        stock_ratio, mode = 0.00, "DCA 중단 구간"

    if stock_index_pct < 0.52:
        if drawdown <= -0.10:
            stock_ratio, mode = 1.00, "위험자산 보강: -10% 이상 조정"
        elif drawdown <= -0.07:
            stock_ratio, mode = 0.80, "위험자산 보강: -7% 이상 조정"
        elif drawdown <= -0.05 and stock_index_pct < 0.50:
            stock_ratio, mode = 0.70, "위험자산 보강: -5% 이상 조정"

    stock_buy = round(monthly * stock_ratio / 10000) * 10000
    cash_buy = monthly - stock_buy
    note = {
        "주식비중 회복 구간": "주식+지수 45% 미만: 지수 DCA를 60%로 높입니다.",
        "기본 DCA 구간": "주식+지수 45~50%: 지수 50%, 단기채·현금 50%를 유지합니다.",
        "감속 구간": "주식+지수 50~52%: 지수 DCA를 25%로 감속합니다.",
        "DCA 중단 구간": "주식+지수 52% 이상: 신규 지수 DCA를 중단합니다.",
    }.get(mode, "하락률 트리거에 따라 월 유입자금의 주식·지수 비중을 높입니다.")
    df = pd.DataFrame([{"구간": mode, "월 유입액": monthly, "주식·지수": stock_buy, "단기채권·현금": cash_buy, "비고": note}])
    return df, stock_buy


def show_percent_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return
    rename = {"asset_class1": "자산구분1", "asset_class2": "자산구분2", "investment_currency": "투자환종", "listing_currency": "상장통화", "hedge_flag": "환헤지", "owner": "소유자", "amount_krw_total": "총자산 기준 금액", "amount_krw_investment_base": "투자기준 금액", "pct_total_assets": "총자산 비중", "pct_investment_base": "투자기준 비중"}
    display = df.rename(columns=rename)
    for col in ["총자산 기준 금액", "투자기준 금액"]:
        if col in display.columns:
            display[col] = display[col].map(fmt_krw)
    for col in ["총자산 비중", "투자기준 비중"]:
        if col in display.columns:
            display[col] = display[col].map(fmt_pct)
    st.dataframe(display, use_container_width=True, hide_index=True)


def show_rules(summary: dict[str, Any]) -> None:
    base = num(summary.get("totals", {}).get("investment_base_krw"))
    stock_amount = group_amount(summary, "by_asset_class1", "asset_class1", "개별주식")
    stock_pct = group_pct(summary, "by_asset_class1", "asset_class1", "개별주식")
    index_amount = group_amount(summary, "by_asset_class1", "asset_class1", "지수")
    index_pct = group_pct(summary, "by_asset_class1", "asset_class1", "지수")
    stock_index_amount = stock_amount + index_amount
    stock_index_pct = stock_pct + index_pct
    cash_pct = group_pct(summary, "by_asset_class1", "asset_class1", "현금")
    short_pct = short_bond_pct(summary)
    liquidity_pct = cash_pct + short_pct
    gold_pct = group_pct(summary, "by_asset_class1", "asset_class1", "헷지")
    usd_pct = group_pct(summary, "by_investment_currency", "investment_currency", "달러")
    lb_pct = long_bond_pct(summary)

    st.subheader("운용 룰")
    st.caption("기준: 장기예금 제외 후 투자기준 총액. 월 신규 현금유입 200만원 기준표이며 자동 주문이 아닙니다.")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("주식+지수", fmt_pct(stock_index_pct)); c2.metric("장기채권", fmt_pct(lb_pct)); c3.metric("단기채권", fmt_pct(short_pct))
    c4.metric("현금+단기채", fmt_pct(liquidity_pct)); c5.metric("금/헷지", fmt_pct(gold_pct)); c6.metric("달러 투자환종", fmt_pct(usd_pct))

    st.markdown("#### 1. 목표비중")
    targets = [
        target_row("주식+지수 목표 하단", stock_index_amount, stock_index_pct, 0.45, base),
        target_row("주식+지수 기본 상단", stock_index_amount, stock_index_pct, 0.50, base),
        target_row("주식+지수 감속 상단", stock_index_amount, stock_index_pct, 0.52, base),
        target_row("주식+지수 리밸런싱 검토선", stock_index_amount, stock_index_pct, 0.55, base),
        target_row("장기채권 기준 목표", group_amount(summary, "by_asset_class1", "asset_class1", "장기채권"), lb_pct, 0.07, base),
        target_row("장기채권 상단", group_amount(summary, "by_asset_class1", "asset_class1", "장기채권"), lb_pct, 0.08, base),
        target_row("현금+단기채 하단", 0, liquidity_pct, 0.30, base),
        target_row("현금+단기채 기준", 0, liquidity_pct, 0.40, base),
        target_row("금/헷지 상단", 0, gold_pct, 0.12, base),
        target_row("개별주식 상단", stock_amount, stock_pct, 0.20, base),
    ]
    show_money_table(pd.DataFrame(targets))

    st.markdown("#### 2. 월 신규 유입자금 배분 룰")
    rules = pd.DataFrame([
        {"주식+지수 구간": "45% 미만", "배분": "주식·지수 120만원 / 단기채·현금 80만원", "의미": "주식비중 회복"},
        {"주식+지수 구간": "45~50%", "배분": "주식·지수 100만원 / 단기채·현금 100만원", "의미": "기본 DCA"},
        {"주식+지수 구간": "50~52%", "배분": "주식·지수 50만원 / 단기채·현금 150만원", "의미": "과열 추격 방지 감속"},
        {"주식+지수 구간": "52% 초과", "배분": "주식·지수 0원 / 단기채·현금 200만원", "의미": "신규 DCA 중단"},
        {"주식+지수 구간": "55% 초과", "배분": "신규 DCA 중단 + 개별주식 비중 점검", "의미": "리밸런싱 검토"},
    ])
    st.dataframe(rules, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        monthly = st.number_input("월 신규 현금유입", min_value=0, max_value=20_000_000, value=2_000_000, step=100_000, format="%d")
    with col2:
        drawdown = st.number_input("S&P500 또는 기준지수 고점 대비 하락률(%)", min_value=-80.0, max_value=30.0, value=0.0, step=0.5) / 100
    allocation, stock_buy = dca_allocation(float(monthly), stock_index_pct, drawdown)
    show_money_table(allocation)

    if stock_buy > 0:
        st.markdown("##### 주식·지수 DCA 내부 배분")
        show_money_table(pd.DataFrame([
            {"구분": "S&P500", "비중": 0.60, "금액": round(stock_buy * 0.60 / 10000) * 10000, "비고": "기본 지수 DCA의 중심"},
            {"구분": "나스닥100/기술주", "비중": 0.40, "금액": round(stock_buy * 0.40 / 10000) * 10000, "비고": "AI·기술주 노출 유지"},
        ]))
        st.markdown("##### 환헤지·환노출 배분")
        show_money_table(pd.DataFrame([
            {"구분": "환헤지형", "비중": 0.70, "금액": round(stock_buy * 0.70 / 10000) * 10000, "비고": "달러 투자환종이 높은 구간의 기본 우선순위"},
            {"구분": "환노출형", "비중": 0.30, "금액": round(stock_buy * 0.30 / 10000) * 10000, "비고": "달러 투자환종 60% 이상이면 신규매수 중단"},
        ]))
    else:
        st.info("현재 조건에서는 주식·지수 월 DCA가 중단되는 구간입니다.")

    st.markdown("#### 3. 하락률 트리거 매수 룰")
    trigger = pd.DataFrame([
        {"트리거": "최근 고점 대비 -5%", "기준": -0.05, "월 유입자금 조정": "주식·지수 140만원 / 단기채·현금 60만원", "기존자산 추가투입": base * 0.00},
        {"트리거": "최근 고점 대비 -7%", "기준": -0.07, "월 유입자금 조정": "주식·지수 160만원 / 단기채·현금 40만원", "기존자산 추가투입": base * 0.01},
        {"트리거": "최근 고점 대비 -10%", "기준": -0.10, "월 유입자금 조정": "주식·지수 200만원", "기존자산 추가투입": base * 0.02},
        {"트리거": "최근 고점 대비 -15%", "기준": -0.15, "월 유입자금 조정": "주식·지수 200만원", "기존자산 추가투입": base * 0.03},
    ])
    trigger_display = trigger.copy(); trigger_display["기준"] = trigger_display["기준"].map(fmt_pct); trigger_display["기존자산 추가투입"] = trigger_display["기존자산 추가투입"].map(fmt_krw)
    st.dataframe(trigger_display, use_container_width=True, hide_index=True)

    st.markdown("#### 4. 매수 중단·주의 조건")
    guards = pd.DataFrame([
        {"조건": "주식+지수 50% 초과", "현재": stock_index_pct, "판정": stock_index_pct > 0.50, "의미": "평시 주식 DCA 감속"},
        {"조건": "주식+지수 52% 초과", "현재": stock_index_pct, "판정": stock_index_pct > 0.52, "의미": "평시 주식 DCA 중단"},
        {"조건": "주식+지수 55% 초과", "현재": stock_index_pct, "판정": stock_index_pct > 0.55, "의미": "개별주식 비중 점검 및 리밸런싱 검토"},
        {"조건": "개별주식 20% 초과", "현재": stock_pct, "판정": stock_pct > 0.20, "의미": "개별주 신규매수 금지"},
        {"조건": "장기채권 8% 초과", "현재": lb_pct, "판정": lb_pct > 0.08, "의미": "장기채 신규매수 중단"},
        {"조건": "현금+단기채권 30% 미만", "현재": liquidity_pct, "판정": liquidity_pct < 0.30, "의미": "유동성 방어자산 부족"},
        {"조건": "금/헷지 12% 초과", "현재": gold_pct, "판정": gold_pct > 0.12, "의미": "금 신규매수 중단"},
        {"조건": "달러 투자환종 55% 초과", "현재": usd_pct, "판정": usd_pct > 0.55, "의미": "환헤지형·원화상장 상품 우선"},
        {"조건": "달러 투자환종 60% 초과", "현재": usd_pct, "판정": usd_pct > 0.60, "의미": "환노출형 신규매수 중단"},
    ])
    guards["현재"] = guards["현재"].map(fmt_pct); guards["판정"] = guards["판정"].map(lambda x: "주의" if x else "정상")
    st.dataframe(guards, use_container_width=True, hide_index=True)

    st.markdown("#### 5. 현재 기준 요약 판단")
    if stock_index_pct < 0.45:
        st.write("- 주식+지수는 목표 하단 45% 아래입니다. 월 유입자금의 60%를 지수 ETF에 배정합니다.")
    elif stock_index_pct < 0.50:
        st.write("- 주식+지수는 45~50% 기본 구간입니다. 월 유입자금의 50%를 지수 ETF에 배정합니다.")
    elif stock_index_pct < 0.52:
        st.write("- 주식+지수는 50~52% 감속 구간입니다. 평시 DCA는 25%로 낮춥니다.")
    elif stock_index_pct < 0.55:
        st.write("- 주식+지수가 52% 이상입니다. 평시 주식·지수 신규매수는 중단합니다.")
    else:
        st.write("- 주식+지수가 55%를 초과했습니다. 개별주식 중심의 리밸런싱을 검토합니다.")


st.sidebar.title("데이터 소스")
source = st.sidebar.radio("읽기 방식", ["로컬 web_data", "URL 직접 입력"], index=0)

try:
    if source == "URL 직접 입력":
        summary_url = st.sidebar.text_input("portfolio_summary.json URL", "")
        holdings_url = st.sidebar.text_input("portfolio_holdings.csv URL", "")
        if not summary_url or not holdings_url:
            st.warning("URL 방식은 JSON URL과 CSV URL을 모두 입력해야 합니다.")
            st.stop()
        summary_json = load_summary_url(summary_url)
        holdings_df = load_holdings_url(holdings_url)
    else:
        if not SUMMARY_PATH.exists() or not HOLDINGS_PATH.exists():
            st.error("web_data 폴더에 portfolio_summary.json / portfolio_holdings.csv 파일이 없습니다.")
            st.stop()
        summary_json = load_summary_local(str(SUMMARY_PATH))
        holdings_df = load_holdings_local(str(HOLDINGS_PATH))
except Exception as exc:
    st.error(f"데이터를 읽는 중 오류가 발생했습니다: {exc}")
    st.stop()

filtered_df = filtered_holdings(holdings_df)

st.title("포트폴리오 대시보드")
st.caption("포트폴리오.xlsx → Python 가격/환율 업데이트 → web_data CSV·JSON 생성 → Streamlit 표시")
st.write(f"기준 파일: `{summary_json.get('source_file', '-')}` / 생성시각: `{generated_at(summary_json.get('generated_at'))}`")

if summary_json.get("method_note"):
    with st.expander("계산 기준"):
        st.write(summary_json["method_note"])

totals = summary_json.get("totals", {})
fx = summary_json.get("fx", {})

m1, m2, m3, m4 = st.columns(4)
m1.metric("총자산", fmt_krw(totals.get("total_asset_krw")))
m2.metric("투자기준 총액", fmt_krw(totals.get("investment_base_krw")))
m3.metric("투자기준 제외", fmt_krw(totals.get("excluded_from_investment_base_krw")))
m4.metric("보유행 수", f"{int(num(totals.get('holding_count'))):,}개")

f1, f2, f3 = st.columns(3)
f1.metric("USD/KRW", fmt_num(fx.get("usd_krw_sell_rate"), 2))
f2.metric("JPY 100엔/KRW", fmt_num(fx.get("jpy_100_krw_sell_rate"), 2))
f3.metric("JPY 1엔/KRW", fmt_num(fx.get("jpy_1_krw_sell_rate"), 4))

st.divider()

tab_summary, tab_holdings, tab_risk, tab_rules = st.tabs(["요약", "보유내역", "분석용 체크포인트", "운용 룰"])

with tab_summary:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("자산구분1 비중")
        show_percent_table(table(summary_json, "by_asset_class1"))
    with c2:
        st.subheader("투자환종 비중")
        show_percent_table(table(summary_json, "by_investment_currency"))
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("상장통화 비중")
        show_percent_table(table(summary_json, "by_listing_currency"))
    with c4:
        st.subheader("환헤지 구분")
        show_percent_table(table(summary_json, "by_hedge_flag"))
    st.subheader("자산구분1·2 상세")
    show_percent_table(table(summary_json, "by_asset_class1_2"))
    st.subheader("소유자별")
    show_percent_table(table(summary_json, "by_owner"))

with tab_holdings:
    st.subheader("보유내역")
    st.write(f"필터 적용 후 {len(filtered_df):,}개 행")
    cols = ["owner", "broker", "account", "asset_class1", "asset_class2", "investment_currency", "listing_currency", "hedge_flag", "ticker", "name", "quantity", "current_price_foreign", "current_price_krw", "market_value_krw", "pnl_rate", "pnl_amount_krw", "pe", "forward_pe", "pb", "roe", "debt_ratio", "dividend_yield", "include_in_investment_base"]
    cols = [c for c in cols if c in filtered_df.columns]
    display_df = filtered_df[cols].copy()
    if "pnl_rate" in display_df.columns:
        display_df["pnl_rate_display"] = display_df["pnl_rate"] * 100
        display_df = display_df.drop(columns=["pnl_rate"])
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.download_button("보유내역 CSV 다운로드", filtered_df.to_csv(index=False).encode("utf-8-sig"), "portfolio_holdings.csv", "text/csv")
    st.download_button("요약 JSON 다운로드", json.dumps(summary_json, ensure_ascii=False, indent=2).encode("utf-8"), "portfolio_summary.json", "application/json")

with tab_risk:
    st.subheader("분석용 체크포인트")
    asset_pct = {}
    asset1_df = table(summary_json, "by_asset_class1")
    if not asset1_df.empty:
        for _, row in asset1_df.iterrows():
            asset_pct[text(row.get("asset_class1"))] = num(row.get("pct_investment_base"))
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("주식+지수", fmt_pct(asset_pct.get("개별주식", 0) + asset_pct.get("지수", 0)))
    r2.metric("장기채권", fmt_pct(long_bond_pct(summary_json)))
    r3.metric("단기채권", fmt_pct(short_bond_pct(summary_json)))
    r4.metric("현금", fmt_pct(asset_pct.get("현금", 0)))
    r5.metric("금/헷지", fmt_pct(asset_pct.get("헷지", 0)))
    st.write("아래 표는 투자 판단용 원자료 확인을 위한 보조 지표입니다. 매수·매도 신호로 자동 해석하지 않습니다.")
    if "market_value_krw" in filtered_df.columns:
        top_df = filtered_df[filtered_df["market_value_krw"].fillna(0) > 0].sort_values("market_value_krw", ascending=False).head(15).copy()
        if "pnl_rate" in top_df.columns:
            top_df["pnl_rate_display"] = top_df["pnl_rate"] * 100
        st.subheader("상위 보유종목")
        st.dataframe(top_df[[c for c in ["asset_class1", "asset_class2", "ticker", "name", "market_value_krw", "pnl_rate_display"] if c in top_df.columns]], use_container_width=True, hide_index=True)

with tab_rules:
    show_rules(summary_json)
