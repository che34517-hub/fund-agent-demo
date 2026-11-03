"""生成过去 7 天三平台（淘宝直通车 / 抖音千川 / 小红书聚光）模拟广告计划数据。

故意构造的问题（供 AdPilot 诊断与策略演示）：
1. 受众重叠：抖音千川多个计划共用同一人群包，互相抢量、CPA 逐日抬升；
2. CPA 异常：部分计划转化率持续偏低或转化骤降，CPA 显著高于平台中位数；
3. 无转化空耗：个别计划持续高消耗但零转化。

用法：
    python data/generate_mock_data.py          # 直接运行生成 CSV
    from data.generate_mock_data import generate_data, write_csv
"""
from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = DATA_DIR / "mock_ads_7d.csv"

# 平台基线：cpm(千次展示成本/元)、ctr(点击率)、cvr(转化率)
PLATFORMS = {
    "taobao_ztc": {"name": "淘宝直通车", "cpm": 25.0, "ctr": 0.035, "cvr": 0.08},
    "douyin_qc": {"name": "抖音千川", "cpm": 18.0, "ctr": 0.025, "cvr": 0.05},
    "xhs_jg": {"name": "小红书聚光", "cpm": 30.0, "ctr": 0.015, "cvr": 0.045},
}

# (platform, plan_id, plan_name, objective, audience_id, audience_tags, base_spend, anomaly)
# anomaly: None | "audience_overlap" | "cpa_high" | "cpa_slump" | "no_conversion" | "high_performer"
PLANS = [
    # 淘宝直通车
    ("taobao_ztc", "ZTC-1001", "美妆-品牌词-精准", "品牌词卡位", "AUD-ZTC-BEAUTY-BRAND", ["美妆", "品牌词", "高意向"], 1200.0, None),
    ("taobao_ztc", "ZTC-1002", "美妆-长尾词-控本", "长尾词控本", "AUD-ZTC-BEAUTY-LONGTAIL", ["美妆", "长尾词"], 900.0, None),
    ("taobao_ztc", "ZTC-1003", "家居-大词-出价偏高", "大词抢量", "AUD-ZTC-HOME-BIGWORD", ["家居", "大词", "泛流量"], 1500.0, "cpa_high"),
    ("taobao_ztc", "ZTC-1004", "服饰-季节词-转化骤降", "季节词拉新", "AUD-ZTC-CLOTH-SEASON", ["服饰", "季节词"], 1100.0, "cpa_slump"),
    ("taobao_ztc", "ZTC-1005", "家居-大词-无转化空耗", "大词测试", "AUD-ZTC-HOME-TEST", ["家居", "大词", "测试"], 800.0, "no_conversion"),
    # 抖音千川（前三个共用同一人群包 -> 受众重叠）
    ("douyin_qc", "QC-A-101", "美妆-25-35女性-达人种草", "直播间引流", "AUD-FEMALE-25-35", ["女性", "25-35", "美妆兴趣"], 2000.0, "audience_overlap"),
    ("douyin_qc", "QC-A-102", "美妆-25-35女性-商品卡", "商品购买", "AUD-FEMALE-25-35", ["女性", "25-35", "美妆兴趣"], 1800.0, "audience_overlap"),
    ("douyin_qc", "QC-A-103", "美妆-25-35女性-直播间", "直播间成交", "AUD-FEMALE-25-35", ["女性", "25-35", "美妆兴趣"], 1600.0, "audience_overlap"),
    ("douyin_qc", "QC-B-201", "服饰-18-24女性-信息流", "短视频种草", "AUD-FEMALE-18-24", ["女性", "18-24", "服饰兴趣"], 1200.0, None),
    # 小红书聚光
    ("xhs_jg", "XHS-5001", "美妆-搜索广告-核心词", "搜索卡位", "AUD-XHS-BEAUTY-SEARCH", ["美妆", "搜索", "高意向"], 800.0, None),
    ("xhs_jg", "XHS-5002", "家居-信息流-种草", "笔记种草", "AUD-XHS-HOME-FEED", ["家居", "信息流"], 700.0, None),
    ("xhs_jg", "XHS-5003", "母婴-笔记加热-高转化", "笔记加热", "AUD-XHS-MOM-NOTE", ["母婴", "笔记加热"], 600.0, "high_performer"),
]


def _apply_anomaly(anomaly: str | None, day_idx: int, days: int, cpm: float, cvr: float) -> tuple[float, float]:
    """按异常类型对当日 CPM / CVR 施加干扰，制造可被诊断的异常信号。"""
    if anomaly == "audience_overlap":
        # 重叠计划互相抢量：CPM 逐日抬升、CVR 逐日下滑（最低保留 30%）
        cpm *= 1.0 + 0.12 * day_idx
        cvr *= max(0.3, 1.0 - 0.10 * day_idx)
    elif anomaly == "cpa_high":
        # 持续低效：转化率长期偏低
        cvr *= 0.2
    elif anomaly == "cpa_slump":
        # 转化骤降：最后三天转化率崩坏
        if day_idx >= days - 3:
            cvr *= 0.2
    elif anomaly == "no_conversion":
        cvr = 0.0
    elif anomaly == "high_performer":
        cvr *= 2.5
    return cpm, cvr


def generate_data(days: int = 7, seed: int = 42) -> list[dict]:
    """生成 `days` 天的计划级每日指标明细。"""
    rng = random.Random(seed)
    end = date.today()
    start = end - timedelta(days=days - 1)
    rows: list[dict] = []

    for plan in PLANS:
        (platform, plan_id, plan_name, objective, audience_id,
         audience_tags, base_spend, anomaly) = plan
        cfg = PLATFORMS[platform]

        for day_idx in range(days):
            d = start + timedelta(days=day_idx)
            weekday_factor = 1.0 if d.weekday() < 5 else 0.8  # 周末消耗略低
            spend = base_spend * weekday_factor * rng.uniform(0.9, 1.1)

            cpm, cvr = _apply_anomaly(anomaly, day_idx, days, cfg["cpm"], cfg["cvr"])

            impressions = int(spend / cpm * 1000)
            clicks = int(impressions * cfg["ctr"] * rng.uniform(0.9, 1.1))
            conversions = int(clicks * cvr * rng.uniform(0.85, 1.15))

            ctr_val = clicks / impressions if impressions else 0.0
            cvr_val = conversions / clicks if clicks else 0.0
            cpa_val = spend / conversions if conversions else 0.0

            rows.append({
                "date": d.isoformat(),
                "platform": platform,
                "platform_name": cfg["name"],
                "plan_id": plan_id,
                "plan_name": plan_name,
                "objective": objective,
                "audience_id": audience_id,
                "audience_tags": "|".join(audience_tags),
                "spend": round(spend, 2),
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "ctr": round(ctr_val, 5),
                "cvr": round(cvr_val, 5),
                "cpa": round(cpa_val, 2),
            })

    return rows


def write_csv(rows: list[dict], path: Path = OUTPUT_PATH) -> Path:
    """将明细行写入 CSV，返回写入路径。"""
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    data = generate_data()
    out = write_csv(data)
    print(f"已生成 {len(data)} 行模拟数据 -> {out}")
