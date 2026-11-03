"""导入真实广告数据：读取 CSV / Excel，做字段检查与归一化，映射到 AdMetric 统一结构。

设计原则：
- 只做「识别 + 转换 + 校验」，绝不静默填充虚假数值；
- 缺字段 / 格式错误时返回明确错误信息；
- 输出 DataFrame 的列与现有 mock_ads_7d.csv 一致（另增 gmv / roi），从而直接复用
  src/agent.py 里的 platform_summary / plan_summary / detect_anomalies 等计算逻辑。

对真实平台报表的兼容要点（完全通用）：
- 只有「日期 + 消耗」为必要字段，其余（转化 / 展示 / 点击 / GMV / ROI / 计划）全部可选；
- 自动剥除表头聚合前缀：整体 / 总 / 汇总 / 合计 / 总计 / 累计 / 求和项: / 平均值项: 等；
- 自动识别 Excel 数据透视表导出的错位表头（前若干行为空 / 标题）；
- 日期兼容 2026-08-01、2026/8/17 以及仅月份「8月」等写法，并丢弃「全部 / 合计 / 总计」汇总行；
- 无「计划 / 商品 / 场景」维度的报表（按日期汇总），自动按单一计划处理，不报错。
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

# 平台中文名 -> 内部平台标识（与 src/schemas.py 的 Platform 枚举一致）
PLATFORM_CODE = {
    "抖音千川": "douyin_qc",
    "小红书聚光": "xhs_jg",
    "淘宝直通车": "taobao_ztc",
    "京东": "jd",
}

# 统一输出列顺序（在 mock_ads_7d.csv 基础上追加 gmv / roi）
CANONICAL_COLUMNS = [
    "date", "platform", "platform_name", "plan_id", "plan_name", "objective",
    "audience_id", "audience_tags", "spend", "impressions", "clicks",
    "conversions", "ctr", "cvr", "cpa", "gmv", "roi",
]

# 归一化目标字段 -> 常见源列名（中英文、含/不含单位括号等）
COLUMN_ALIASES = {
    "date": ["日期", "时间", "月份", "统计日期", "投放日期", "date", "day", "天"],
    "spend": ["消耗", "花费", "消费", "cost", "spend", "charge", "消耗金额", "花费金额"],
    "impressions": ["展示", "展示量", "展示次数", "曝光", "曝光量", "曝光次数", "展现", "展现量", "展现次数", "impressions", "impression", "show_cnt"],
    "clicks": ["点击", "点击量", "点击次数", "clicks", "click", "click_cnt"],
    "conversions": [
        "转化", "转化量", "转化数", "转化次数",
        "成交笔数", "成交数", "成交订单数", "成交订单量",
        "订单数", "订单量", "下单量", "支付订单数", "支付订单量", "支付订单",
        "conversions", "conversion", "convert_cnt",
    ],
    "gmv": [
        "成交金额", "成交额", "交易额", "全站交易额", "支付金额",
        "交易金额", "成交总额", "gmv", "总成交金额",
    ],
    "roi": ["支付roi", "投入产出比", "投产比", "全站投产比", "roi", "投资回报率", "含预售投产比"],
    "plan_id": ["计划id", "计划编号", "广告计划id", "推广计划id", "商品id", "商品编号", "场景id", "plan_id", "campaign_id", "ad_id", "product_id"],
    "plan_name": ["计划名称", "计划名", "广告计划名称", "推广计划名称", "商品名称", "商品名", "场景名字", "场景名称", "场景名", "plan_name", "campaign_name", "ad_name", "product_name"],
    "objective": ["投放目标", "营销目标", "营销目的", "推广目标", "objective"],
    "audience_id": ["人群包id", "人群id", "受众id", "定向包id", "audience_id", "dmp_id", "定向人群id", "人群包"],
    "audience_tags": ["人群标签", "受众标签", "定向标签", "兴趣标签", "audience_tags"],
}

# 用于报错信息的中文显示名
FIELD_LABELS = {
    "date": "日期", "spend": "消耗", "impressions": "展示", "clicks": "点击",
    "conversions": "转化", "gmv": "成交金额(GMV)", "roi": "ROI",
    "plan_id": "计划ID", "plan_name": "计划名称",
    "objective": "投放目标", "audience_id": "人群包ID", "audience_tags": "人群标签",
}

# 表头聚合前缀（出现在最前面时剥除），如「整体消耗」「总成交笔数」「求和项:花费」
_AGG_PREFIX_RE = re.compile(
    r"^(?:整体|全部|汇总|合计|总计|累计|总|求和项|平均值项|计数项|最大值项|最小值项|sum\s+of|average\s+of)[:：]?"
)

# 仅月份写法，如「10月」「8月」-> 当年该月第一天
_MONTH_ONLY_RE = re.compile(r"^(\d{1,2})月$")


def _norm(s: str) -> str:
    """规范化表头：去空白、转小写、去掉括号内单位、剥除聚合前缀。"""
    s = str(s).strip().lower()
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\([^)]*\)", "", s)          # 去掉 (元) / (次) 等单位后缀
    s = re.sub(r"\s+", "", s)                 # 去掉所有空白
    s = _AGG_PREFIX_RE.sub("", s)             # 剥除 整体/总/求和项: … 前缀
    return s


def _to_numeric(series: pd.Series) -> pd.Series:
    """把带货币符号/千分位/单位的中文数字列转为数值。"""
    s = (
        series.astype(str)
        .str.replace("¥", "", regex=False)
        .str.replace("￥", "", regex=False)
        .str.replace("元", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("，", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(s, errors="coerce")


def _coerce_dates(series: pd.Series) -> pd.Series:
    """解析日期：兼容 2026-08-01 / 2026/8/17 / 10月 等写法，失败为 NaT。"""
    def _fmt(v) -> str:
        t = str(v).strip()
        m = _MONTH_ONLY_RE.fullmatch(t)
        if m:
            return f"{date.today().year}-{int(m.group(1)):02d}-01"
        return t
    return pd.to_datetime(series.map(_fmt), errors="coerce", format="mixed")


def ensure_writable(df: pd.DataFrame) -> pd.DataFrame:
    """深拷贝并确保所有底层 numpy 数组可写，杜绝只读数组进入后续计算。

    openpyxl / Streamlit Cloud 在某些版本组合下会返回 writeable=False 的数组，
    后续任何原地运算（如 ``valid &= mask``）都会抛 "output array is read-only"。
    """
    if df is None:
        return df
    df = df.copy(deep=True)
    for col in df.columns:
        arr = df[col].to_numpy()
        if not arr.flags.writeable:
            df[col] = arr.copy()
    return df


@dataclass
class ImportResult:
    """导入结果：成功时携带归一化后的 DataFrame，失败时携带错误信息。"""
    ok: bool = False
    df: pd.DataFrame | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    field_map: dict[str, str] = field(default_factory=dict)   # 目标字段 -> 源列名
    recognized: list[str] = field(default_factory=list)
    unrecognized: list[str] = field(default_factory=list)
    source_name: str = ""
    platform_code: str = ""
    platform_name: str = ""
    n_rows: int = 0


def _detect_header_row(raw: pd.DataFrame) -> int:
    """定位真实表头行：返回首个含 >=2 个非空字符串单元格的行号（跳过标题/空行）。"""
    for idx in range(len(raw)):
        n_str = sum(1 for c in raw.iloc[idx] if isinstance(c, str) and c.strip())
        if n_str >= 2:
            return idx
    return 0


def read_uploaded(uploaded) -> pd.DataFrame:
    """读取上传文件为 DataFrame；自动识别错位表头；CSV 自动尝试 utf-8 / gb18030 编码。"""
    name = str(getattr(uploaded, "name", "") or "")
    raw = uploaded.getvalue()

    if name.lower().endswith((".xlsx", ".xlsm")):
        df0 = pd.read_excel(io.BytesIO(raw), engine="openpyxl", header=None)
        hdr = _detect_header_row(df0)
        df = df0.iloc[hdr + 1:].reset_index(drop=True)
        df.columns = [str(c).strip() for c in df0.iloc[hdr].tolist()]
        return ensure_writable(df)

    for enc in ("utf-8-sig", "gb18030"):
        try:
            df0 = pd.read_csv(io.BytesIO(raw), encoding=enc, header=None)
            hdr = _detect_header_row(df0)
            df = df0.iloc[hdr + 1:].reset_index(drop=True)
            df.columns = [str(c).strip() for c in df0.iloc[hdr].tolist()]
            return ensure_writable(df)
        except UnicodeDecodeError:
            continue
    df0 = pd.read_csv(io.BytesIO(raw), encoding="utf-8", errors="replace", header=None)
    hdr = _detect_header_row(df0)
    df = df0.iloc[hdr + 1:].reset_index(drop=True)
    df.columns = [str(c).strip() for c in df0.iloc[hdr].tolist()]
    return ensure_writable(df)


def _build_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            lookup.setdefault(_norm(a), canonical)
    return lookup


def check_and_normalize(df: pd.DataFrame, platform_label: str) -> ImportResult:
    """对原始 DataFrame 做字段检查与格式转换，输出可复用的统一结构。"""
    res = ImportResult(
        platform_code=PLATFORM_CODE.get(platform_label, ""),
        platform_name=platform_label,
    )

    if df is None or df.empty or df.shape[1] == 0:
        res.errors.append("文件为空，或没有可识别的列。")
        return res

    # 深拷贝 + 强制可写：解除 openpyxl / 云端环境可能返回的只读 numpy 数组限制
    # （否则后续原地运算会抛出 "output array is read-only"）。
    df = ensure_writable(df)

    lookup = _build_lookup()
    col_map: dict[str, str] = {}
    for col in df.columns:
        canon = lookup.get(_norm(col))
        if canon and canon not in col_map:
            col_map[canon] = str(col)
            res.recognized.append(str(col))
        elif canon is None:
            res.unrecognized.append(str(col))
    res.field_map = col_map

    # ---- 必要字段检查：仅「日期 + 消耗」为必填，其余全部可选 ----
    missing = [FIELD_LABELS[f] for f in ("date", "spend") if f not in col_map]
    if missing:
        res.errors.append("缺少必要字段：" + "、".join(missing))
        res.errors.append("已识别字段：" + (", ".join(res.recognized) if res.recognized else "无"))
        if res.unrecognized:
            res.errors.append("未识别字段（未被使用）：" + ", ".join(res.unrecognized))
        return res

    # ---- 解析日期与核心数值，构造有效行掩码 ----
    dates = _coerce_dates(df[col_map["date"]])
    valid = np.asarray(dates.notna(), dtype=bool).copy()  # 显式可写副本

    parsed: dict[str, pd.Series] = {}
    parsed["spend"] = _to_numeric(df[col_map["spend"]])
    valid = valid & np.asarray(parsed["spend"].notna(), dtype=bool)  # 非原地，避免只读写入

    for f in ("conversions", "impressions", "clicks", "gmv", "roi"):
        if f in col_map:
            parsed[f] = _to_numeric(df[col_map[f]])
            valid = valid & np.asarray(parsed[f].notna(), dtype=bool)

    if int(valid.sum()) == 0:
        res.errors.append("所有数据行都无法通过「日期」或数字格式校验，请检查文件内容。")
        return res

    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        res.warnings.append(
            f"有 {n_dropped} 行因日期无法解析（如汇总行「全部/合计/总计」）或数字格式错误被丢弃（未填充任何值）。"
        )

    dfv = df[valid].reset_index(drop=True)

    # ---- 组装统一结构 ----
    out = pd.DataFrame()
    out["date"] = np.asarray(dates.dt.normalize())[valid]
    out["spend"] = np.asarray(parsed["spend"], dtype=float)[valid]
    for f in ("conversions", "impressions", "clicks"):
        if f in parsed:
            out[f] = np.asarray(parsed[f], dtype=float)[valid].round().astype("int64")
        else:
            out[f] = np.nan
    for f in ("gmv", "roi"):
        out[f] = np.asarray(parsed[f], dtype=float)[valid] if f in parsed else np.nan

    def _s(col: str) -> pd.Series:
        return dfv[col].astype(str).str.strip()

    # 计划维度：四种情况分别处理，缺失时绝不报错
    has_pid = "plan_id" in col_map
    has_pname = "plan_name" in col_map
    if has_pid and has_pname:
        out["plan_id"] = _s(col_map["plan_id"])
        out["plan_name"] = _s(col_map["plan_name"])
    elif has_pid:
        out["plan_id"] = _s(col_map["plan_id"])
        out["plan_name"] = out["plan_id"]
        res.warnings.append("未找到「计划名称」列，已用计划ID作为显示名称。")
    elif has_pname:
        out["plan_id"] = _s(col_map["plan_name"])
        out["plan_name"] = _s(col_map["plan_name"])
        res.warnings.append("未找到「计划ID」列，已用「计划名称」作为计划标识。")
    else:
        account_plan = res.platform_name + "·账户汇总"
        out["plan_id"] = (res.platform_code or "account") + "_account"
        out["plan_name"] = account_plan
        res.warnings.append(
            f"该报表按日期/时间汇总、无计划/商品/场景维度，已按「{account_plan}」作为单一计划处理。"
        )

    # 可选字段：缺失则留空（不影响核心计算），并明确告知
    missing_optional = []
    for f in ("objective", "audience_id", "audience_tags"):
        if f in col_map:
            vals = dfv[col_map[f]]
            if f == "audience_tags":
                out[f] = vals.apply(
                    lambda x: "|".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x)
                )
            else:
                out[f] = vals.astype(str).str.strip()
        else:
            out[f] = ""
            missing_optional.append(FIELD_LABELS[f])

    # 核心指标缺失提示（不影响导入，仅提示哪些指标不可用）
    missing_metrics = []
    if "conversions" not in parsed:
        missing_metrics.append("转化")
    if "impressions" not in parsed or "clicks" not in parsed:
        missing_metrics.append("展示/点击(CTR/CVR)")
    if "gmv" not in parsed:
        missing_metrics.append("GMV")
    if "roi" not in parsed:
        missing_metrics.append("ROI")
    if missing_metrics:
        res.warnings.append("未找到以下指标字段（已留空，相关指标不可用）：" + "、".join(missing_metrics))
    if missing_optional:
        res.warnings.append("未找到可选字段（已留空，不影响核心计算）：" + "、".join(missing_optional))

    # 平台字段：由侧边栏选择的平台统一设定
    out["platform"] = res.platform_code
    out["platform_name"] = res.platform_name

    # 派生指标：按核心指标真实计算（分母为 0 或缺失时为 NaN，不填充虚假值）
    out["ctr"] = out["clicks"] / out["impressions"].replace(0, np.nan)
    out["cvr"] = out["conversions"] / out["clicks"].replace(0, np.nan)
    out["cpa"] = out["spend"] / out["conversions"].replace(0, np.nan)

    out = out[CANONICAL_COLUMNS]
    res.df = ensure_writable(out)
    res.ok = True
    res.n_rows = len(out)
    return res
