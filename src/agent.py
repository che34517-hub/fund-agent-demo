"""核心逻辑：读取数据 -> 结合 RAG 规则 -> 调用大模型 -> 生成晨间日报与结构化策略建议。

当未配置 LLM_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY 时，自动回退到内置规则引擎，
保证 MVP 开箱即用。
"""
from __future__ import annotations

import csv
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import rag
from .schemas import ActionType, PLATFORM_NAMES, StrategyAction, StrategyPlan

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "mock_ads_7d.csv"
APPROVAL_PATH = DATA_DIR / "approvals.csv"

# ---- LLM 配置（OpenAI 兼容协议，默认指向 DeepSeek，可用环境变量覆盖）----
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")


# --------------------------------------------------------------------------- #
# 数据读取与聚合
# --------------------------------------------------------------------------- #
def _safe_ratio(num, den) -> np.ndarray:
    """安全除法：分母为 0 时返回 NaN（避免 inf）。"""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num / den
    return np.where(den > 0, ratio, np.nan)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def platform_summary(df: pd.DataFrame) -> pd.DataFrame:
    """分平台聚合近 N 天核心指标。"""
    g = (
        df.groupby(["platform", "platform_name"], as_index=False)
        .agg(
            plan_count=("plan_id", "nunique"),
            spend=("spend", "sum"),
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
        )
    )
    g["ctr"] = _safe_ratio(g["clicks"], g["impressions"])
    g["cvr"] = _safe_ratio(g["conversions"], g["clicks"])
    g["cpa"] = _safe_ratio(g["spend"], g["conversions"])
    return g


def plan_summary(df: pd.DataFrame) -> pd.DataFrame:
    """计划级聚合近 N 天核心指标。"""
    g = (
        df.groupby(["platform", "platform_name", "plan_id", "plan_name", "audience_id"], as_index=False)
        .agg(
            days=("date", "nunique"),
            spend=("spend", "sum"),
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
        )
    )
    g["ctr"] = _safe_ratio(g["clicks"], g["impressions"])
    g["cvr"] = _safe_ratio(g["conversions"], g["clicks"])
    g["cpa"] = _safe_ratio(g["spend"], g["conversions"])
    return g


# --------------------------------------------------------------------------- #
# 诊断（规则引擎）
# --------------------------------------------------------------------------- #
def detect_anomalies(df: pd.DataFrame) -> list[dict]:
    """识别异常，供诊断预警卡片与 LLM 提示词使用。"""
    anomalies: list[dict] = []
    plan_df = plan_summary(df)
    median = plan_df.groupby("platform")["cpa"].median()

    # 1) 计划级：无转化高消耗 / CPA 超标
    for _, row in plan_df.iterrows():
        cpa = row["cpa"]
        med = median.get(row["platform"])
        base = {"platform": row["platform"], "platform_name": row["platform_name"], "plan_ids": [row["plan_id"]]}

        if row["conversions"] == 0 and row["spend"] >= 500:
            anomalies.append({
                **base,
                "level": "critical",
                "title": f"无转化高消耗：{row['plan_name']}",
                "detail": f"近 7 天消耗 ¥{row['spend']:,.0f}，转化 0 次，建议暂停止损。",
            })
            continue
        if pd.isna(cpa) or pd.isna(med):
            continue
        ratio = cpa / med if med > 0 else float("inf")
        if ratio >= 2.0 and row["spend"] >= 500:
            anomalies.append({
                **base,
                "level": "critical",
                "title": f"CPA 异常：{row['plan_name']}",
                "detail": f"CPA ¥{cpa:.1f}，为平台中位数 ¥{med:.1f} 的 {ratio:.1f} 倍，建议降预算 30%。",
            })
        elif ratio >= 1.5 and row["spend"] >= 500:
            anomalies.append({
                **base,
                "level": "warning",
                "title": f"CPA 偏高：{row['plan_name']}",
                "detail": f"CPA ¥{cpa:.1f}，为平台中位数 ¥{med:.1f} 的 {ratio:.1f} 倍，建议降预算 15%。",
            })

    # 2) 转化骤降：近 3 天日均转化较前段下滑 > 50%
    df_sorted = df.sort_values("date")
    for (platform, plan_id), g in df_sorted.groupby(["platform", "plan_id"]):
        if g["spend"].sum() < 500:
            continue
        g = g.sort_values("date")
        recent = float(g.tail(3)["conversions"].mean())
        prior = float(g.head(max(0, len(g) - 3))["conversions"].mean())
        if prior > 0 and recent < prior * 0.5:
            drop = 1 - recent / prior
            plan_name = g["plan_name"].iloc[0]
            plat_name = g["platform_name"].iloc[0]
            anomalies.append({
                "platform": platform,
                "platform_name": plat_name,
                "plan_ids": [plan_id],
                "level": "warning",
                "title": f"转化骤降：{plan_name}",
                "detail": f"近 3 天日均转化 {recent:.1f}，较前段 {prior:.1f} 下滑 {drop:.0%}，建议排查素材/人群衰退。",
            })

    # 3) 受众重叠：同一平台多个活跃计划共用同一人群包
    grouped = plan_df[plan_df["audience_id"].ne("")].groupby(["platform", "audience_id"])
    for (platform, audience_id), group in grouped:
        if len(group) < 2:
            continue
        plat_name = group["platform_name"].iloc[0]
        plan_names = "、".join(group["plan_name"].tolist())
        anomalies.append({
            "platform": platform,
            "platform_name": plat_name,
            "plan_ids": group["plan_id"].tolist(),
            "level": "info",
            "title": f"受众重叠：{len(group)} 个计划共用人群包 {audience_id}",
            "detail": f"涉及 {plan_names}，互相抢量，建议错开定向或差异化出价。",
        })

    order = {"critical": 0, "warning": 1, "info": 2}
    anomalies.sort(key=lambda a: order[a["level"]])
    return anomalies


# --------------------------------------------------------------------------- #
# LLM 调用
# --------------------------------------------------------------------------- #
def _get_client():
    if not LLM_API_KEY:
        return None, None
    try:
        from openai import OpenAI
    except ImportError:
        return None, None
    kwargs = {"api_key": LLM_API_KEY}
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    return OpenAI(**kwargs), LLM_MODEL


def call_llm(system: str, user: str) -> dict | None:
    """调用 OpenAI 兼容接口，要求返回 JSON 对象；失败返回 None 走规则回退。"""
    client, model = _get_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as exc:  # noqa: BLE001
        print(f"[LLM] 调用/解析失败，回退规则引擎：{exc}")
        return None


def _new_plan_id() -> str:
    return f"SP-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"


def _build_messages(plan_df, platform_df, anomalies, rules_text, report_date) -> tuple[str, str]:
    system = (
        "你是 AdPilot 的多平台广告投放优化助手，精通淘宝直通车、抖音千川、小红书聚光。\n"
        "请基于投放数据、诊断异常与平台官方规则，输出晨间日报摘要与结构化策略建议。\n"
        "硬性要求：\n"
        "1. 只输出一个合法 JSON 对象，不要输出 markdown 代码块或解释文字。\n"
        "2. JSON 必须严格符合给定的 JSON Schema。\n"
        "3. action_type 只能使用 Schema 中 enum 列出的英文枚举值。\n"
        "4. adjustment_ratio 在 [-1,1] 之间：-1=暂停/全量下调，0=不变，正数=上调。\n"
        "5. 每条策略的 adjustment_reason 必须具体、可执行。\n"
    )
    schema = StrategyPlan.model_json_schema()
    user = "\n\n".join([
        "# 报告日期",
        report_date,
        "# 分平台概览（近 7 天）",
        platform_df.to_json(orient="records", force_ascii=False),
        "# 计划明细（近 7 天汇总）",
        plan_df.to_json(orient="records", force_ascii=False),
        "# 诊断异常",
        json.dumps(anomalies, ensure_ascii=False, default=str),
        "# 平台官方规则（RAG 检索结果）",
        rules_text,
        "# 输出 JSON Schema（严格遵循）",
        json.dumps(schema, ensure_ascii=False),
    ])
    return system, user


# --------------------------------------------------------------------------- #
# 规则引擎回退
# --------------------------------------------------------------------------- #
def rule_based_strategy(plan_df, platform_df, anomalies, report_date) -> StrategyPlan:
    """确定性规则引擎：当无 LLM 或 LLM 输出非法时兜底。"""
    actions: list[StrategyAction] = []
    median = plan_df.groupby("platform")["cpa"].median()
    paused: set[str] = set()

    for _, row in plan_df.iterrows():
        cpa = row["cpa"]
        med = median.get(row["platform"])
        plan_id = row["plan_id"]

        if row["conversions"] == 0 and row["spend"] >= 500:
            paused.add(plan_id)
            actions.append(StrategyAction(
                plan_id=plan_id, plan_name=row["plan_name"], platform=row["platform"],
                action_type=ActionType.PAUSE_PLAN, adjustment_ratio=-1.0,
                adjustment_reason=f"近 7 天消耗 ¥{row['spend']:,.0f} 但转化 0 次，建议暂停止损。",
                confidence=0.9,
            ))
            continue
        if pd.isna(cpa) or pd.isna(med):
            continue
        ratio = cpa / med if med > 0 else 0.0
        if ratio >= 2.0:
            actions.append(StrategyAction(
                plan_id=plan_id, plan_name=row["plan_name"], platform=row["platform"],
                action_type=ActionType.REDUCE_BUDGET, adjustment_ratio=-0.3,
                adjustment_reason=f"CPA ¥{cpa:.1f} 为平台中位数 ¥{med:.1f} 的 {ratio:.1f} 倍，建议降预算 30%。",
                confidence=0.85,
            ))
        elif ratio >= 1.5:
            actions.append(StrategyAction(
                plan_id=plan_id, plan_name=row["plan_name"], platform=row["platform"],
                action_type=ActionType.REDUCE_BUDGET, adjustment_ratio=-0.15,
                adjustment_reason=f"CPA ¥{cpa:.1f} 为平台中位数 ¥{med:.1f} 的 {ratio:.1f} 倍，建议降预算 15%。",
                confidence=0.8,
            ))
        elif ratio <= 0.6 and row["spend"] >= 500:
            actions.append(StrategyAction(
                plan_id=plan_id, plan_name=row["plan_name"], platform=row["platform"],
                action_type=ActionType.INCREASE_BUDGET, adjustment_ratio=0.2,
                adjustment_reason=f"CPA ¥{cpa:.1f} 低于平台中位数 ¥{med:.1f}，预算花满，建议提量 20%。",
                confidence=0.75,
            ))

    # 受众重叠 -> 调整定向（跳过已建议暂停的计划）
    grouped = plan_df[plan_df["audience_id"].ne("")].groupby(["platform", "audience_id"])
    for (platform, audience_id), group in grouped:
        if len(group) < 2:
            continue
        for _, row in group.iterrows():
            if row["plan_id"] in paused:
                continue
            actions.append(StrategyAction(
                plan_id=row["plan_id"], plan_name=row["plan_name"], platform=platform,
                action_type=ActionType.ADJUST_AUDIENCE, adjustment_ratio=0.0,
                adjustment_reason=f"人群包 {audience_id} 被 {len(group)} 个计划共用，互相抢量，建议错开定向或差异化出价。",
                confidence=0.8,
            ))

    summary = _build_summary(platform_df, anomalies, actions)
    return StrategyPlan(
        plan_id=_new_plan_id(),
        report_date=report_date,
        summary=summary,
        actions=actions,
    )


def _build_summary(platform_df, anomalies, actions) -> str:
    total_spend = float(platform_df["spend"].sum())
    total_conv = int(platform_df["conversions"].sum())
    total_cpa = total_spend / total_conv if total_conv else 0.0
    n_critical = sum(1 for a in anomalies if a["level"] == "critical")
    n_warning = sum(1 for a in anomalies if a["level"] == "warning")
    n_info = sum(1 for a in anomalies if a["level"] == "info")

    parts = [f"近 7 天三平台合计消耗 ¥{total_spend:,.0f}，总转化 {total_conv:,} 次，整体 CPA ¥{total_cpa:.1f}。"]
    if n_critical or n_warning or n_info:
        parts.append(f"诊断发现 {n_critical} 条严重预警、{n_warning} 条提醒、{n_info} 条受众重叠提示。")
    if total_conv:
        best = platform_df.loc[platform_df["cpa"].idxmin()]
        if pd.notna(best["cpa"]):
            parts.append(f"表现最佳平台为 {best['platform_name']}（CPA ¥{best['cpa']:.1f}）。")
    parts.append(f"共生成 {len(actions)} 条策略建议，等待审批。")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# 策略生成入口
# --------------------------------------------------------------------------- #
def generate_strategy(df: pd.DataFrame | None = None, force_rule_based: bool = False) -> StrategyPlan:
    """生成晨间日报与结构化策略；优先 LLM，失败/未配置时回退规则引擎。"""
    if df is None:
        df = load_data()

    plan_df = plan_summary(df)
    platform_df = platform_summary(df)
    anomalies = detect_anomalies(df)
    report_date = str(df["date"].max().date())

    query = " ".join(a["title"] for a in anomalies) + " " + " ".join(platform_df["platform_name"].tolist())
    rules = rag.retrieve(query, top_k=2)
    rules_text = rag.format_context(rules)

    plan = None
    if not force_rule_based:
        system, user = _build_messages(plan_df, platform_df, anomalies, rules_text, report_date)
        payload = call_llm(system, user)
        if payload:
            try:
                payload["report_date"] = report_date
                if not payload.get("plan_id"):
                    payload["plan_id"] = _new_plan_id()
                plan = StrategyPlan.model_validate(payload)
            except Exception as exc:  # noqa: BLE001
                print(f"[策略] Schema 校验失败，回退规则引擎：{exc}")
                plan = None

    if plan is None:
        plan = rule_based_strategy(plan_df, platform_df, anomalies, report_date)

    return plan


# --------------------------------------------------------------------------- #
# 审批回写
# --------------------------------------------------------------------------- #
_APPROVAL_COLUMNS = [
    "approved_at", "strategy_plan_id", "report_date", "platform", "platform_name",
    "plan_id", "plan_name", "action_type", "adjustment_ratio", "adjustment_reason", "status",
]


def write_approvals(plan: StrategyPlan, statuses: dict[int, str]) -> Path | None:
    """把已审批（通过/拒绝）的策略动作追加回写到 data/approvals.csv。"""
    rows = []
    for i, a in enumerate(plan.actions):
        sts = statuses.get(i, "pending")
        if sts == "pending":
            continue
        rows.append({
            "approved_at": datetime.now().isoformat(timespec="seconds"),
            "strategy_plan_id": plan.plan_id,
            "report_date": plan.report_date,
            "platform": a.platform,
            "platform_name": PLATFORM_NAMES.get(a.platform, a.platform),
            "plan_id": a.plan_id,
            "plan_name": a.plan_name,
            "action_type": a.action_type.value,
            "adjustment_ratio": a.adjustment_ratio,
            "adjustment_reason": a.adjustment_reason,
            "status": sts,
        })
    if not rows:
        return None
    need_header = not APPROVAL_PATH.exists()
    with APPROVAL_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_APPROVAL_COLUMNS)
        if need_header:
            writer.writeheader()
        writer.writerows(rows)
    return APPROVAL_PATH


def load_approvals() -> pd.DataFrame:
    if not APPROVAL_PATH.exists():
        return pd.DataFrame(columns=_APPROVAL_COLUMNS)
    return pd.read_csv(APPROVAL_PATH)
