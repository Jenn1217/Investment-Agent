# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 17:00:51 2026

@author: HUAWEI
"""
import os
import json
import pandas as pd
import requests
from typing import List, Dict, Any, Tuple

# ========== 你的 Dify 配置 ==========
API_KEY = os.getenv("DIFY_API_KEY", "app-sfr1VIzn7gHNXbNE9Mvq3jNp")
BASE_URL = os.getenv("DIFY_BASE_URL", "https://agent.test.investoday.net/v1")
CHAT_ENDPOINT = f"{BASE_URL}/chat-messages"

# ========== 你需要替换成“宏观/微观”各自的文件路径 ==========
MICRO_EXCEL_PATH = r"D:\current\今日投资\news_extraction\117_micro_bertopic_output_with_main_event.xlsx"
MICRO_NEWS_JSON_PATH =r"D:\current\今日投资\news_extraction\117_micro_extract.json"

# 下面两个路径请替换成你宏观文档实际上传的文件名/路径
MACRO_EXCEL_PATH = r"D:\current\今日投资\news_extraction\117_macro_bertopic_output_with_main_event.xlsx"
MACRO_NEWS_JSON_PATH = r"D:\current\今日投资\news_extraction\117_macro_extract.json"

REPORT_PROMPT = r"""
# Role Definition
你是一名服务于高净值客户的首席投资策略师。你的工作不是罗列新闻，而是从信息中提炼主线并撰写可执行的《高频投资情报》。

# Input Data
你将接收到两组 Topics：
A) micro_topics：偏行业/公司/产业链的事件主线
B) macro_topics：偏政策/监管/利率/汇率/财政/国际局势的事件主线
每个 Topic 包含：
- topic_name
- count（仅供你心中判断，报告中禁止出现“热度/计数/Topic/聚类”等字样）
- keywords
- original_texts（来自新闻原文提取的事件句，可用作证据引用）
- news_ids（仅供定位，报告中禁止输出）

# Task: Report Generation
请严格按以下结构输出，标题与顺序不能变：

## Section 1: 市场高价值信息综述
要求：允许综合 micro_topics + macro_topics，写 100-150 字，风格克制但要能给交易者方向。

## Section 2: 投资机会捕捉
要求：只允许基于 micro_topics 提炼 TOP3 正向催化主线（必要时合并同类项）。
个股推荐规则：必须在 original_texts 中能找到公司名称；代码也必须来自 original_texts；找不到则只能给 ETF/行业方向。

## Section 3: 宏观信息分析
要求：只允许基于 macro_topics 识别最重要的一条宏观主线；若 macro_topics 不足以支撑，就写“宏观面整体平稳”并给出仓位/风格建议。

## Section 4: 风控雷达
要求：允许使用 micro_topics + macro_topics 中的负面主线；每条给出明确动作（回避/卖出），并引用 original_texts 中一句话作为证据。

# Critical Constraints
1) 去技术化：严禁出现“聚类/Topic/Count/模型/算法/ID”等字样，也不要输出任何字段名。
2) 事实核查：不得编造股票/代码/事件。
3) 商业口吻：专业术语可以用，但避免空话。

# Evidence & Source Requirement（重要）
在以下 Section 中，请在每一段分析或每一条投资结论后，附上对应的新闻来源证据：

- 证据格式统一为：
  【来源】
  - NewsID: <新闻ID>
  - 摘要: <对应的 original_texts 中的事件总结句>

具体要求：
1) Section 1（市场综述）：  
   在段落结尾，用【来源】列出 2–3 条最关键的支撑新闻。

2) Section 2（投资机会捕捉）：  
   - 每一条“主线”分析后，必须附【来源】  
   - 若涉及个股，优先选择直接提及该公司的新闻作为来源

3) Section 3（宏观信息分析）：  
   - 必须至少给出 1 条宏观政策或

""".strip()


def load_news_map(news_json_path: str) -> Dict[str, str]:
    """NewsID -> event_summary（来自新闻原文抽取，可作为证据句）"""
    with open(news_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for item in data:
        nid = str(item.get("NewsID", "")).strip()
        if not nid:
            continue
        out[nid] = (item.get("event_summary") or "").strip()
    return out


def split_ids(id_str: Any) -> List[str]:
    return [x.strip() for x in str(id_str).split(",") if x.strip() and x.strip().lower() != "nan"]


def split_keywords(keyword_str: Any, max_k: int = 12) -> List[str]:
    parts = [p.strip() for p in str(keyword_str).split("|")]
    parts = [p for p in parts if p and p.lower() != "nan"]
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:max_k]


def build_topics(excel_path: str, news_map: Dict[str, str], max_texts: int = 25) -> List[Dict[str, Any]]:
    """
    从事件簇 Excel 构造 topics 列表。
    依赖列名：主要事件名称 / topic_size / 重点事件 / NewsID集合
    """
    df = pd.read_excel(excel_path)

    topics: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        topic_name = str(row.get("主要事件名称", "")).strip()
        if not topic_name or topic_name.lower() == "nan":
            continue

        news_ids = split_ids(row.get("NewsID集合", ""))
        original_texts = [news_map.get(nid, "") for nid in news_ids]
        original_texts = [t for t in original_texts if t]

        # topic_size 仅用于“心中判断”，不给报告输出
        try:
            count = int(row.get("topic_size", 0))
        except Exception:
            count = 0

        topics.append({
            "topic_name": topic_name,
            "count": count,
            "keywords": split_keywords(row.get("重点事件", "")),
            "original_texts": original_texts[:max_texts],
            "news_ids": news_ids[:50],
        })

    # 简单排序：热度高在前（不输出数字，只用于模型理解）
    topics.sort(key=lambda x: -x.get("count", 0))
    return topics


def call_dify_chat(prompt: str,
                  micro_topics: List[Dict[str, Any]],
                  macro_topics: List[Dict[str, Any]],
                  user_id: str = "invest_report_bot") -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # 关键修改：把 inputs 变成“可读文本”直接塞进 query，让大模型必定看到
    full_query = f"""
以下是用于生成投资报告的完整输入信息。请直接生成报告正文，不要向用户索取任何额外 inputs，不要输出任何字段名。

【主提示 Prompt】
{prompt}

【微观事件主题 micro_topics】
{json.dumps(micro_topics, ensure_ascii=False, indent=2)}

【宏观事件主题 macro_topics】
{json.dumps(macro_topics, ensure_ascii=False, indent=2)}
""".strip()

    payload = {
        # 关键修改：inputs 可留空，避免 Dify 编排变量不匹配导致模型“要 inputs”
        "inputs": {},
        "query": full_query,
        "response_mode": "blocking",
        "user": user_id,
    }

    resp = requests.post(CHAT_ENDPOINT, headers=headers, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    # 兼容不同 Dify 返回结构
    for key_path in [
        ("answer",),
        ("data", "answer"),
        ("message", "content"),
        ("data", "message", "content"),
    ]:
        cur = data
        ok = True
        for k in key_path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, str) and cur.strip():
            return cur

    return json.dumps(data, ensure_ascii=False, indent=2)



def main():
    # 读取微观
    micro_news_map = load_news_map(MICRO_NEWS_JSON_PATH)
    micro_topics = build_topics(MICRO_EXCEL_PATH, micro_news_map)

    # 读取宏观
    macro_news_map = load_news_map(MACRO_NEWS_JSON_PATH)
    macro_topics = build_topics(MACRO_EXCEL_PATH, macro_news_map)

    # 调用大模型生成报告
    report = call_dify_chat(REPORT_PROMPT, micro_topics, macro_topics, user_id="hnw_client_001")
    print(report)


if __name__ == "__main__":
    main()
