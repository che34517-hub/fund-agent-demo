"""AdPilot 多平台广告投放 AI 助手 —— Streamlit 前端仪表盘。

功能：
- 多平台核心指标总览（KPI + 图表）；
- 智能诊断预警卡片；
- 策略建议审批台（逐条审批 + 一键审批回写）。
"""
from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.generate_mock_data import OUTPUT_PATH, generate_data, write_csv  # noqa: E402
from src import importer  # noqa: E402
from src.agent import (  # noqa: E402
    LLM_API_KEY,
    LLM_MODEL,
    detect_anomalies,
    generate_strategy,
    load_approvals,
    load_data,
    platform_summary,
    plan_summary,
    write_approvals,
)
from src.schemas import ACTION_TYPE_NAMES, PLATFORM_NAMES  # noqa: E402

st.set_page_config(page_title="AdPilot 多平台广告投放 AI 助手", page_icon="📈", layout="wide")

# 统一视觉：平台固定分类色（Okabe-Ito 色盲友好，顺序固定，不随筛选变化）
PLATFORM_ORDER = ["淘宝直通车", "抖音千川", "小红书聚光", "京东"]
PALETTE = {
    "淘宝直通车": "#E69F00",
    "抖音千川": "#56B4E9",
    "小红书聚光": "#009E73",
    "京东": "#CC79A7",
}
INK = "#3B3B3B"
GRID = "#E4E4E4"


def _chart_layout(fig: go.Figure, showlegend: bool = False) -> go.Figure:
    fig.update_layout(
        height=340,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=13),
        xaxis=dict(showgrid=False, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zeroline=False),
        showlegend=showlegend,
        legend=dict(orientation="h", y=1.15, x=0),
    )
    return fig


def _format_ratio(action) -> str:
    if action.action_type.value in {"adjust_audience", "optimize_creative"}:
        return "—"
    return f"{action.adjustment_ratio:+.0%}"


def _actions_frame(actions, statuses) -> pd.DataFrame:
    cn_status = {"pending": "待审批", "approved": "已通过", "rejected": "已拒绝"}
    rows = []
    for i, a in enumerate(actions):
        rows.append({
            "计划": a.plan_name,
            "平台": PLATFORM_NAMES.get(a.platform, a.platform),
            "操作": ACTION_TYPE_NAMES.get(a.action_type.value, a.action_type.value),
            "调整比例": _format_ratio(a),
            "调整理由": a.adjustment_reason,
            "置信度": f"{a.confidence:.0%}",
            "状态": cn_status[statuses.get(i, "pending")],
        })
    return pd.DataFrame(rows)


def ensure_data() -> None:
    if not OUTPUT_PATH.exists():
        write_csv(generate_data())


@st.cache_data(ttl=60, show_spinner=False)
def get_df() -> pd.DataFrame:
    return load_data()


ensure_data()

# 数据源：优先使用已导入的各平台数据（多平台联合合并），否则回退到内置模拟数据
uploaded_data = st.session_state.get("uploaded_data") or {}
imported_frames = [
    importer.ensure_writable(r.df) for r in list(uploaded_data.values())
    if isinstance(r, importer.ImportResult) and r.ok and r.df is not None
]
if imported_frames:
    df = importer.ensure_writable(pd.concat(imported_frames, ignore_index=True))
    data_source = "imported"
else:
    df = importer.ensure_writable(get_df())
    data_source = "mock"

# 页面导航标签
NAV_OVERVIEW = "📊 广告投放总览"
NAV_AI = "🤖 AI 智能诊断"
NAV_SETTINGS = "⚙️ 系统设置"

# --------------------------------------------------------------------------- #
# 侧边栏
# --------------------------------------------------------------------------- #
with st.sidebar:
    nav = st.sidebar.radio(
        "🧭 页面导航", [NAV_OVERVIEW, NAV_AI, NAV_SETTINGS], key="nav_page"
    )

    st.header("📊 数据")
    platforms_all = sorted(df["platform_name"].unique().tolist())
    selected = st.multiselect("平台筛选", platforms_all, default=platforms_all)

    st.header("📥 导入广告数据")
    import_platform = st.selectbox(
        "平台", list(importer.PLATFORM_CODE.keys()), key="import_platform"
    )
    # 每个平台独立的上传控件：切换平台不会复用上一个平台的文件
    uploaded = st.file_uploader(
        "上传报表（CSV / Excel）",
        type=["csv", "xlsx"],
        key=f"import_file_{import_platform}_{st.session_state.get('import_reset', 0)}",
    )
    if uploaded is not None:
        digest = hashlib.md5(uploaded.getvalue()).hexdigest()
        digests = st.session_state.setdefault("_upload_digests", {})
        if digests.get(import_platform) != digest:
            try:
                raw = importer.read_uploaded(uploaded)
                result = importer.check_and_normalize(raw, import_platform)
                result.source_name = getattr(uploaded, "name", "")
            except Exception as exc:  # noqa: BLE001
                result = importer.ImportResult()
                result.errors.append(f"读取文件失败：{exc}")
            ud = st.session_state.setdefault("uploaded_data", {})
            errs = st.session_state.setdefault("_import_errors", {})
            if result.ok:
                ud[import_platform] = result
                errs.pop(import_platform, None)
            else:
                ud.pop(import_platform, None)
                errs[import_platform] = result.errors
            digests[import_platform] = digest
            st.rerun()

    # 导入状态与错误提示：按平台展示，互不覆盖
    uploaded_data = st.session_state.get("uploaded_data") or {}
    for pname, r in uploaded_data.items():
        if isinstance(r, importer.ImportResult) and r.ok:
            st.success(f"✅ 已导入「{pname}」：{r.source_name} · {r.n_rows} 行")
    for pname, errs in (st.session_state.get("_import_errors") or {}).items():
        st.error(f"❌ 「{pname}」导入失败，请修正文件后重试")
        for e in errs:
            st.error(e)
    if uploaded_data:
        with st.expander("⚠️ 导入提示", expanded=False):
            for pname, r in uploaded_data.items():
                for w in r.warnings:
                    st.caption(f"· [{pname}] " + w)
        if st.button("🗑 清空已导入数据", use_container_width=True):
            st.session_state["import_reset"] = st.session_state.get("import_reset", 0) + 1
            st.session_state["uploaded_data"] = {}
            st.session_state["_upload_digests"] = {}
            st.session_state["_import_errors"] = {}
            for key in ("strategy_plan", "action_statuses", "_fp"):
                st.session_state.pop(key, None)
            st.rerun()

view = df[df["platform_name"].isin(selected)] if selected else df
if view.empty:
    st.warning("请至少选择一个平台。")
    st.stop()

# --------------------------------------------------------------------------- #
# 标题（全局）
# --------------------------------------------------------------------------- #
st.title("📈 AdPilot · 多平台广告投放 AI 助手")
st.caption("淘宝直通车 · 抖音千川 · 小红书聚光 —— 晨间日报 / 智能诊断 / 策略审批")

# --------------------------------------------------------------------------- #
# 页面 1：📊 广告投放总览
# --------------------------------------------------------------------------- #
if nav == NAV_OVERVIEW:
    if data_source == "imported":
        platform_list = "、".join(uploaded_data.keys())
        st.caption(
            f"📥 当前数据源：已导入 {len(uploaded_data)} 个平台（{platform_list}）· 共 {len(df)} 行"
        )
        with st.expander("📥 导入数据预览与字段映射"):
            for pname, r in uploaded_data.items():
                st.markdown(
                    f"**平台**：{pname}　|　**文件**：`{r.source_name}`　|　**行数**：{r.n_rows}"
                )
                if r.field_map:
                    st.markdown("字段映射（目标字段 ← 上传文件列）")
                    st.dataframe(
                        pd.DataFrame(
                            [{"目标字段": k, "上传文件列": v} for k, v in r.field_map.items()]
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )
                if r.unrecognized:
                    st.caption("未被使用的列：" + "、".join(r.unrecognized))
            st.markdown("**合并数据预览（前 10 行）**")
            st.dataframe(df.head(10), hide_index=True, use_container_width=True)

    total_spend = view["spend"].sum()
    total_conv = view["conversions"].sum()
    total_clicks = view["clicks"].sum()
    total_imp = view["impressions"].sum()
    overall_cpa = total_spend / total_conv if total_conv else float("nan")
    n_plans = view["plan_id"].nunique()

    has_conv = bool(view["conversions"].notna().any())
    has_imp = bool(view["impressions"].notna().any())
    has_gmv = "gmv" in view.columns and bool(view["gmv"].notna().any())
    total_gmv = float(view["gmv"].sum()) if has_gmv else float("nan")
    total_roi = total_gmv / total_spend if (has_gmv and total_spend) else float("nan")

    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric("近 7 天消耗", f"¥{total_spend:,.0f}")
    k2.metric("转化量", f"{total_conv:,}" if has_conv else "—")
    k3.metric("总成交额 GMV", f"¥{total_gmv:,.0f}" if pd.notna(total_gmv) else "—")
    k4.metric("整体 ROI", f"{total_roi:.2f}" if pd.notna(total_roi) else "—")
    k5.metric("整体 CPA", f"¥{overall_cpa:,.1f}" if (has_conv and total_conv) else "—")
    k6.metric("点击率 CTR", f"{total_clicks / total_imp:.2%}" if (has_imp and total_imp) else "—")
    k7.metric("活跃计划", f"{n_plans}")

    st.subheader("📊 多平台核心指标")
    pf = platform_summary(view).sort_values("spend", ascending=False)

    left, right = st.columns(2)
    with left:
        st.markdown("**近 7 天消耗（按平台）**")
        fig = go.Figure(go.Bar(
            x=pf["platform_name"],
            y=pf["spend"],
            marker_color=[PALETTE.get(n, "#888888") for n in pf["platform_name"]],
            text=[f"¥{v:,.0f}" for v in pf["spend"]],
            textposition="outside",
            hovertemplate="%{x}<br>消耗 ¥%{y:,.0f}<extra></extra>",
        ))
        st.plotly_chart(_chart_layout(fig), use_container_width=True)

    with right:
        st.markdown("**整体 CPA（按平台）**")
        fig = go.Figure(go.Bar(
            x=pf["platform_name"],
            y=pf["cpa"],
            marker_color=[PALETTE.get(n, "#888888") for n in pf["platform_name"]],
            text=[f"¥{v:,.1f}" if pd.notna(v) else "—" for v in pf["cpa"]],
            textposition="outside",
            hovertemplate="%{x}<br>CPA ¥%{y:,.1f}<extra></extra>",
        ))
        st.plotly_chart(_chart_layout(fig), use_container_width=True)

    st.markdown("**近 7 天 CPA 趋势**")
    daily = view.groupby(["date", "platform_name"], as_index=False).agg(
        spend=("spend", "sum"), conversions=("conversions", "sum")
    )
    daily["cpa"] = np.where(daily["conversions"] > 0, daily["spend"] / daily["conversions"], np.nan)

    fig = go.Figure()
    for name in [n for n in PLATFORM_ORDER if n in selected]:
        d = daily[daily["platform_name"] == name].sort_values("date")
        if d.empty:
            continue
        color = PALETTE[name]
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["cpa"], name=name, mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color),
            hovertemplate=f"{name}<br>%{{x|%m-%d}} CPA ¥%{{y:,.1f}}<extra></extra>",
        ))
        last = d.iloc[-1]
        if pd.notna(last["cpa"]):
            fig.add_annotation(
                x=last["date"], y=last["cpa"], text=f"¥{last['cpa']:.0f}",
                showarrow=False, xanchor="left", xshift=6, font=dict(color=color, size=12),
            )
    st.plotly_chart(_chart_layout(fig, showlegend=True), use_container_width=True)

    with st.expander("📋 计划明细数据"):
        col_cn = {
            "platform": "平台", "platform_name": "平台名称", "plan_id": "计划ID",
            "plan_name": "计划名称", "audience_id": "人群包ID", "days": "投放天数",
            "spend": "消耗", "impressions": "展示", "clicks": "点击",
            "conversions": "转化", "ctr": "CTR", "cvr": "CVR", "cpa": "CPA",
        }
        st.dataframe(plan_summary(view).rename(columns=col_cn), use_container_width=True, hide_index=True)

    if st.button("🔄 重新生成模拟数据", use_container_width=True):
        write_csv(generate_data(seed=random.randint(1, 1_000_000)))
        get_df.clear()
        for key in ("strategy_plan", "action_statuses", "_fp"):
            st.session_state.pop(key, None)
        st.rerun()

# --------------------------------------------------------------------------- #
# 页面 2：🤖 AI 智能诊断
# --------------------------------------------------------------------------- #
elif nav == NAV_AI:
    st.subheader("🩺 智能诊断预警")
    anomalies = detect_anomalies(view)
    if not anomalies:
        st.success("未发现异常，所有计划表现健康。")
    for a in anomalies:
        level = a["level"]
        icon = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}[level]
        body = f"{icon} **{a['title']}**  \n{a['detail']}  \n*平台：{a['platform_name']}*"
        if level == "critical":
            st.error(body)
        elif level == "warning":
            st.warning(body)
        else:
            st.info(body)

    fingerprint = tuple(sorted(selected))
    if "strategy_plan" not in st.session_state or st.session_state.get("_fp") != fingerprint:
        with st.spinner("正在生成策略建议…"):
            st.session_state.strategy_plan = generate_strategy(view)
            st.session_state._fp = fingerprint
    plan = st.session_state.strategy_plan

    st.subheader("🌅 晨间日报")
    st.info(plan.summary)

    st.subheader("✅ 策略建议审批台")
    if st.session_state.get("flash"):
        st.success(st.session_state.flash)
        st.session_state.flash = ""

    if "action_statuses" not in st.session_state:
        st.session_state.action_statuses = {}
    statuses = st.session_state.action_statuses

    if not plan.actions:
        st.info("暂无策略建议。")
    else:
        st.dataframe(_actions_frame(plan.actions, statuses), use_container_width=True, hide_index=True)

        st.markdown("**逐条审批**")
        for i, a in enumerate(plan.actions):
            sts = statuses.get(i, "pending")
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([2.2, 1.2, 1.0, 3.4, 1.4])
                c1.markdown(
                    f"**{a.plan_name}**  \n<small>{PLATFORM_NAMES.get(a.platform, a.platform)} · `{a.plan_id}`</small>",
                    unsafe_allow_html=True,
                )
                c2.markdown(ACTION_TYPE_NAMES.get(a.action_type.value, a.action_type.value))
                c3.markdown(_format_ratio(a))
                c4.caption(a.adjustment_reason)
                if sts == "approved":
                    c5.success("已通过 ✅")
                elif sts == "rejected":
                    c5.error("已拒绝 ❌")
                else:
                    b1, b2 = c5.columns(2)
                    if b1.button("通过", key=f"appr_{plan.plan_id}_{i}"):
                        statuses[i] = "approved"
                        st.rerun()
                    if b2.button("拒绝", key=f"rej_{plan.plan_id}_{i}"):
                        statuses[i] = "rejected"
                        st.rerun()

        col_a, col_b, col_c = st.columns(3)
        if col_a.button("✅ 一键审批全部通过", use_container_width=True):
            for i in range(len(plan.actions)):
                statuses[i] = "approved"
            st.rerun()
        if col_b.button("❌ 一键全部拒绝", use_container_width=True):
            for i in range(len(plan.actions)):
                statuses[i] = "rejected"
            st.rerun()
        if col_c.button("💾 回写审批结果", use_container_width=True):
            path = write_approvals(plan, statuses)
            if path is None:
                st.warning("没有待回写的审批结果，请先通过/拒绝。")
            else:
                n = sum(1 for v in statuses.values() if v != "pending")
                st.session_state.flash = f"✅ 已回写 {n} 条审批结果到 `{path.name}`"
                st.session_state.action_statuses = {}
                st.rerun()

    if st.button("🧠 重新生成策略", use_container_width=True):
        st.session_state.pop("strategy_plan", None)
        st.session_state.pop("action_statuses", None)
        st.rerun()

    with st.expander("🗂️ 已回写的审批记录"):
        approvals = load_approvals()
        if approvals.empty:
            st.caption("暂无回写记录。")
        else:
            st.dataframe(approvals, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# 页面 3：⚙️ 系统设置
# --------------------------------------------------------------------------- #
else:
    st.subheader("🔑 模型 / API Key 配置")
    if LLM_API_KEY:
        st.success(f"✅ LLM 已启用：`{LLM_MODEL}`")
    else:
        st.info("未配置 API Key，当前使用内置规则引擎。")

    st.markdown(
        "AdPilot 的策略建议优先调用大模型（OpenAI 兼容协议，兼容 DeepSeek 等），"
        "未配置 API Key 时自动回退到内置规则引擎。\n\n"
        "**环境变量**：\n"
        "- `LLM_API_KEY`（或 `OPENAI_API_KEY` / `DEEPSEEK_API_KEY`）：API Key\n"
        "- `LLM_BASE_URL`（或 `OPENAI_BASE_URL` / `DEEPSEEK_BASE_URL`）：接口地址（可选）\n"
        "- `LLM_MODEL`：模型名（默认 `deepseek-chat`）\n\n"
        "在 Streamlit Cloud 中可在「Settings → Secrets」里配置上述变量。"
    )
