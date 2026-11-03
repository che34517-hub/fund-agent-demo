"""统一的广告指标 Schema 与 LLM 策略输出模型（JSON Schema 约束）。

包含：
- AdMetric：跨平台统一的核心指标 Schema；
- StrategyAction：单条策略建议（计划ID、操作类型、调整比例、调整理由）；
- StrategyPlan：LLM 输出的一整套策略（含日报摘要与动作列表）。
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Platform(str, Enum):
    TAOBAO_ZTC = "taobao_ztc"   # 淘宝直通车
    DOUYIN_QC = "douyin_qc"     # 抖音千川
    XHS_JG = "xhs_jg"           # 小红书聚光
    JD = "jd"                   # 京东


PLATFORM_NAMES = {
    Platform.TAOBAO_ZTC.value: "淘宝直通车",
    Platform.DOUYIN_QC.value: "抖音千川",
    Platform.XHS_JG.value: "小红书聚光",
    Platform.JD.value: "京东",
}


class ActionType(str, Enum):
    REDUCE_BUDGET = "reduce_budget"          # 降低预算/出价
    INCREASE_BUDGET = "increase_budget"      # 提高预算
    PAUSE_PLAN = "pause_plan"                # 暂停计划
    ADJUST_AUDIENCE = "adjust_audience"      # 调整人群定向
    OPTIMIZE_CREATIVE = "optimize_creative"  # 优化素材
    NO_ACTION = "no_action"                  # 保持现状


ACTION_TYPE_NAMES = {
    ActionType.REDUCE_BUDGET.value: "降低预算/出价",
    ActionType.INCREASE_BUDGET.value: "提高预算",
    ActionType.PAUSE_PLAN.value: "暂停计划",
    ActionType.ADJUST_AUDIENCE.value: "调整人群定向",
    ActionType.OPTIMIZE_CREATIVE.value: "优化素材",
    ActionType.NO_ACTION.value: "保持现状",
}


class AdMetric(BaseModel):
    """单条广告计划单日的核心指标（跨平台统一 Schema）。"""

    date: str = Field(..., description="日期，格式 YYYY-MM-DD")
    platform: str = Field(..., description="平台标识，如 taobao_ztc / douyin_qc / xhs_jg")
    platform_name: str = Field("", description="平台中文名")
    plan_id: str = Field(..., description="计划 ID")
    plan_name: str = Field("", description="计划名称")
    objective: str = Field("", description="投放目标 / 营销目的")
    audience_id: str = Field("", description="人群定向包 ID")
    audience_tags: list[str] = Field(default_factory=list, description="人群标签")
    spend: float = Field(0.0, ge=0.0, description="消耗（元）")
    impressions: int = Field(0, ge=0, description="展示量")
    clicks: int = Field(0, ge=0, description="点击量")
    conversions: int = Field(0, ge=0, description="转化量")
    ctr: float = Field(0.0, ge=0.0, description="点击率")
    cvr: float = Field(0.0, ge=0.0, description="转化率")
    cpa: float = Field(0.0, ge=0.0, description="单次转化成本（元）")
    gmv: float = Field(0.0, ge=0.0, description="成交金额 / GMV（元）")
    roi: float = Field(0.0, ge=0.0, description="投产比 / ROI（成交金额/消耗）")


class StrategyAction(BaseModel):
    """单条结构化策略建议：计划ID + 操作类型 + 调整比例 + 调整理由。"""

    plan_id: str = Field(..., description="目标计划 ID")
    plan_name: str = Field("", description="计划名称")
    platform: str = Field("", description="平台标识")
    action_type: ActionType = Field(..., description="操作类型")
    adjustment_ratio: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="调整比例：-1=暂停/全量下调，0=不变，正数=上调（0.2 表示 +20%）",
    )
    adjustment_reason: str = Field(..., description="调整理由，需具体可执行")
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="置信度")
    status: Literal["pending", "approved", "rejected"] = Field("pending", description="审批状态")


class StrategyPlan(BaseModel):
    """LLM 输出的一整套结构化策略（含批次 ID、日报摘要与动作列表）。"""

    plan_id: str = Field(..., description="策略批次 ID（唯一）")
    report_date: str = Field(..., description="报告日期 YYYY-MM-DD")
    summary: str = Field("", description="晨间日报摘要")
    actions: list[StrategyAction] = Field(default_factory=list, description="策略动作列表")
