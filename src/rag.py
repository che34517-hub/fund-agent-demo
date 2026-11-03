"""从 knowledge_base/ 读取平台官方投放规则并进行简单检索（关键词加权）。"""
from __future__ import annotations

from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"

# 领域词典：把业务上下文映射到对应平台的规则文档
DOMAIN_TERMS = {
    "抖音千川": ["千川", "抖音", "巨量", "人群包", "重叠", "抢量"],
    "淘宝直通车": ["直通车", "淘宝", "万相台", "关键词", "出价", "低效", "空耗"],
    "小红书聚光": ["聚光", "小红书", "种草", "笔记", "搜索"],
}

# 通用诊断信号词，用于跨文档匹配
GENERIC_TERMS = ["重叠", "抢量", "CPA", "低效", "出价", "人群", "转化", "预算", "暂停", "无转化", "骤降"]


def list_documents() -> list[Path]:
    return sorted(KB_DIR.glob("*.md"))


def load_documents() -> list[dict]:
    """加载 knowledge_base 下所有 markdown 文档。"""
    docs = []
    for p in list_documents():
        docs.append({
            "name": p.stem,
            "path": str(p),
            "content": p.read_text(encoding="utf-8"),
        })
    return docs


def _extract_terms(query: str) -> set[str]:
    """从查询中提取命中领域词典的关键词。"""
    terms: set[str] = set()
    for kw_group in DOMAIN_TERMS.values():
        for kw in kw_group:
            if kw in query:
                terms.add(kw)
    for kw in GENERIC_TERMS:
        if kw in query:
            terms.add(kw)
    return terms


def _score(doc: dict, terms: set[str]) -> float:
    """关键词加权评分：标题命中权重更高。"""
    name = doc["name"]
    content = doc["content"]
    score = 0.0
    for t in terms:
        if t in name:
            score += 3.0
        score += content.count(t) * 1.0
    return score


def retrieve(query: str, top_k: int = 2) -> list[dict]:
    """根据查询检索最相关的规则文档（返回 top_k 篇）。"""
    docs = load_documents()
    terms = _extract_terms(query)
    if not terms:
        return docs[:top_k]
    scored = sorted(((_score(d, terms), d) for d in docs), key=lambda x: x[0], reverse=True)
    top = [d for s, d in scored if s > 0][:top_k]
    return top or docs[:1]


def format_context(docs: list[dict]) -> str:
    """把检索到的规则拼成提示词片段。"""
    if not docs:
        return "（未检索到相关规则）"
    return "\n\n".join(f"### {d['name']}\n{d['content']}" for d in docs)
