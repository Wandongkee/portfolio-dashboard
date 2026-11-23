"""주택가격 대비 성과 비교 계산 (2026-11-23, investment_policy.md '주택가격 대비 성과 비교').

- 추이와 비교지표 분리: 비교는 월간 지수 대상월의 '마지막 history 기록일' 스냅샷 기준.
- 월 수익률: Modified Dietz 근사 (입출금은 cash_flows.csv의 counted=Y, 반영일 기준).
  r = (V1 - V0 - F) / (V0 + Σ w_i·F_i),  w_i = (d1 - t_i) / (d1 - d0)
  반영일 t_i의 평가액에는 이미 그 입금이 들어 있다고 보므로 t_i == d1이면 w=0, t_i == d0은 이전 구간.
- ① 상대성과 지수 = (1 + 누적 TWR) / (비교월 지수 / 시작월 지수) × 100  (적립금 효과 제외)
- ② 주택가격 환산 자산 = 투자기준 총액 / (비교월 지수 / 시작월 지수)  (적립금 포함, 시작월 원화 기준)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

START_MONTH = "2026-11"          # 시작 스냅샷 = 2026-11 마지막 기록일 (사용자 결정 2026-11-23)
FLOW_START_DATE = "2026-12-01"   # 입출금 기록 시작일
VALUE_COL = "investment_base_krw"


@dataclass
class CompareResult:
    table: pd.DataFrame
    start_snapshot: pd.Timestamp | None
    start_value: float | None
    start_index: float | None
    pending_months: list[str] = field(default_factory=list)  # 스냅샷은 있으나 지수 미공표
    missing_snapshot_months: list[str] = field(default_factory=list)  # 지수는 있으나 기록 없음
    warnings: list[str] = field(default_factory=list)
    status: str = "ok"


def _month(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m")


def month_snapshots(history: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """완료된 달(today가 속한 달 이전)의 마지막 기록일 스냅샷."""
    h = history.dropna(subset=["date", VALUE_COL]).copy()
    h["date"] = pd.to_datetime(h["date"]).dt.normalize()
    h["month"] = h["date"].dt.strftime("%Y-%m")
    h = h[h["month"] < today.strftime("%Y-%m")]
    snap = h.sort_values("date").groupby("month", as_index=False).last()
    return snap[["month", "date", VALUE_COL]].rename(columns={VALUE_COL: "value"})


def _counted_flows(flows: pd.DataFrame | None) -> pd.DataFrame:
    if flows is None or flows.empty:
        return pd.DataFrame(columns=["date", "net_flow_krw"])
    f = flows.copy()
    f["date"] = pd.to_datetime(f["date"], errors="coerce").dt.normalize()
    f["net_flow_krw"] = pd.to_numeric(f["net_flow_krw"], errors="coerce").fillna(0.0)
    if "counted" in f.columns:
        f = f[f["counted"].astype(str).str.upper() == "Y"]
    f = f.dropna(subset=["date"])
    return f[f["date"] >= pd.Timestamp(FLOW_START_DATE)][["date", "net_flow_krw"]]


def modified_dietz(v0: float, v1: float, d0: pd.Timestamp, d1: pd.Timestamp, flows: pd.DataFrame) -> tuple[float, float]:
    period = flows[(flows["date"] > d0) & (flows["date"] <= d1)]
    total = float(period["net_flow_krw"].sum())
    days = (d1 - d0).days
    if days <= 0:
        raise ValueError("스냅샷 날짜 순서 오류")
    weighted = float(sum(((d1 - t).days / days) * a for t, a in zip(period["date"], period["net_flow_krw"])))
    denom = v0 + weighted
    if denom <= 0:
        raise ValueError("Modified Dietz 분모가 0 이하")
    return (v1 - v0 - total) / denom, total


def compute(history: pd.DataFrame, housing: pd.DataFrame, flows: pd.DataFrame | None,
            today: pd.Timestamp | None = None) -> CompareResult:
    today = (today or pd.Timestamp.now()).normalize()
    warnings: list[str] = []
    empty = pd.DataFrame()

    if history is None or history.empty or VALUE_COL not in history.columns:
        return CompareResult(empty, None, None, None, status="no_history")
    snaps = month_snapshots(history, today)
    snaps = snaps[snaps["month"] >= START_MONTH].reset_index(drop=True)
    if snaps.empty or snaps.iloc[0]["month"] != START_MONTH:
        return CompareResult(empty, None, None, None, status="start_pending")

    if housing is None or housing.empty:
        return CompareResult(empty, snaps.iloc[0]["date"], float(snaps.iloc[0]["value"]), None, status="no_index")
    hx = housing.copy()
    hx["index_value"] = pd.to_numeric(hx["index_value"], errors="coerce")
    hx = hx.dropna(subset=["index_value"])
    if "base_period" in hx.columns and hx["base_period"].nunique() > 1:
        warnings.append("지수 기준시점이 섞여 있습니다. 동일 기준시점 시계열로 다시 수집해야 합니다.")
    idx = dict(zip(hx["target_month"].astype(str), hx["index_value"]))

    start = snaps.iloc[0]
    if START_MONTH not in idx:
        return CompareResult(empty, start["date"], float(start["value"]), None,
                             pending_months=[START_MONTH], status="start_index_pending")
    i0 = float(idx[START_MONTH])
    fl = _counted_flows(flows)
    early = fl[fl["date"] <= start["date"]]
    if not early.empty:
        warnings.append("시작 스냅샷 이전 날짜의 입출금은 계산에서 제외했습니다.")

    rows = []
    growth = 1.0
    cum_flow = 0.0
    prev = start
    pending, missing = [], []
    snap_months = set(snaps["month"])
    for m in sorted(k for k in idx if k > START_MONTH):
        if m not in snap_months and m < today.strftime("%Y-%m"):
            missing.append(m)
    for _, s in snaps.iloc[1:].iterrows():
        r, f = modified_dietz(float(prev["value"]), float(s["value"]), prev["date"], s["date"], fl)
        growth *= 1 + r
        cum_flow += f
        prev = s
        if s["month"] not in idx:
            pending.append(s["month"])
            continue
        ratio = float(idx[s["month"]]) / i0
        y, mo = map(int, s["month"].split("-"))
        y0, mo0 = map(int, START_MONTH.split("-"))
        months = (y - y0) * 12 + (mo - mo0)  # 경과 개월 수 (기록이 빠진 달이 있어도 달력 기준)
        twr = growth - 1
        row = {
            "비교월": s["month"],
            "스냅샷일": s["date"].strftime("%Y-%m-%d"),
            "투자기준 총액": float(s["value"]),
            "누적 순입금": cum_flow,
            "월 수익률(근사)": r,
            "누적 TWR": twr,
            "서울 아파트 지수": float(idx[s["month"]]),
            "지수 누적 상승률": ratio - 1,
            "누적 초과성과(%p)": (twr - (ratio - 1)) * 100,
            "① 상대성과 지수": (1 + twr) / ratio * 100,
            "② 주택가격 환산 자산": float(s["value"]) / ratio,
        }
        if months >= 12:
            row["TWR 연환산"] = (1 + twr) ** (12 / months) - 1
            row["지수 연환산"] = ratio ** (12 / months) - 1
        rows.append(row)
    return CompareResult(pd.DataFrame(rows), start["date"], float(start["value"]), i0,
                         pending, missing, warnings, "ok")
