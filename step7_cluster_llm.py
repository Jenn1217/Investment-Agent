# -*- coding: utf-8 -*-
"""
Excel -> 事件簇主事件命名（Dify Chat App）
约束：API 每次最多处理 10 条（按 batch=10 + 批间 sleep 的保守策略）
输入列：NewsID集合 / event_summary集合
输出列：主要事件名称 / main_event_confidence / main_event_why
并带：缓存（可断点续跑）
"""

import os
import re
import json
import time
import hashlib
from typing import Optional, Dict, Any, List

import pandas as pd
import requests


# =========================
# 0) 基础配置
# =========================
API_KEY = os.getenv("DIFY_API_KEY", "app-sfr1VIzn7gHNXbNE9Mvq3jNp")
BASE_URL = os.getenv("DIFY_BASE_URL", "https://agent.test.investoday.net/v1")

DEFAULT_USER = os.getenv("DIFY_USER", "antigravity_tester")


# =========================
# 1) Dify API 封装（Chat App /chat-messages）
# =========================
def call_dify_api(
    query: str,
    inputs: Optional[Dict[str, Any]] = None,
    user: str = DEFAULT_USER,
    conversation_id: Optional[str] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    url = f"{BASE_URL}/chat-messages"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": inputs or {},
        "query": query,
        "response_mode": "blocking",
        "user": user,
        "conversation_id": conversation_id,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# =========================
# 2) 解析“集合”单元格
#    兼容：list / json / 逗号分隔 / 顿号 / 换行
# =========================
_SPLIT_RE = re.compile(r"[,\n\r\t;；、|]+")


def parse_set_cell(x: Any) -> List[str]:
    """把单元格内容解析成字符串列表（去空、去重、保序）"""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, (list, tuple)):
        raw = [str(i).strip() for i in x if str(i).strip()]
    else:
        s = str(x).strip()
        if not s:
            return []
        # 尝试 JSON list / dict
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    raw = [str(i).strip() for i in obj if str(i).strip()]
                else:
                    raw = [s]  # dict 等：当整体字符串处理
            except Exception:
                raw = [t.strip() for t in _SPLIT_RE.split(s) if t.strip()]
        else:
            raw = [t.strip() for t in _SPLIT_RE.split(s) if t.strip()]

    seen = set()
    out: List[str] = []
    for item in raw:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# =========================
# 3) Prompt（专业版：严格不补外部事实）
# =========================
def build_prompt(news_ids: List[str], summaries: List[str]) -> str:
    # 控制长度，避免输入爆炸
    summaries_trim = summaries[:20]
    if len(summaries) > 20:
        summaries_trim.append("...(仅展示前20条，其余已截断)")

    return f"""
你是一名金融新闻研究员，负责对“同一事件簇”的多来源新闻句子/摘要片段进行事件级归纳与标准化命名。

⚠️ 重要约束（必须严格遵守）：
- 你只能使用我提供的 event_summary 集合中的信息
- 不得引入、补充、推断任何外部事实或背景（包括常识性背景）
- 不得假设主体、时间、因果关系；若材料未明确出现，则视为“未知/不确定”
- event_summary 可能来自不同文章的句子拼接，存在转述、改写、噪声、并列事件

【输入材料】
- event_summary 集合（同一事件簇内多条句子/摘要片段）：
{summaries_trim}

【任务目标】
为该事件簇生成一个【主要事件名称】（中文），用于事件库主键与后续量化分析（需稳定、可复用、可审计）。

【工作流程（在心中完成，不要输出过程）】
A) 先从材料中提取“高频且语义一致”的事件核（谁/什么对象 + 做了什么动作/发生了什么变化 + 可见结果/性质）
B) 识别并排除噪声：
   - 纯观点/评论/情绪性表述
   - 与事件核弱相关的背景句、泛化描述
   - 与事件核不一致的支线事件（若占比低）
C) 若存在多个并列事件核：
   - 选择“出现频次最高 / 语义最集中 / 对其他句子具有解释力”的一个作为主事件
   - 其余作为从属，不写进主事件名称

【命名规则】
1) 事件名称必须是“事件级标题”，不是句子、不是评论、不是观点
2) 字数 10–18 个汉字；信息密度高、结构清晰
3) 主体可能不止一个：
   - 若材料中主体高度集中：写 1 个核心主体
   - 若材料中经常出现两个主体且关系清晰：最多写 2 个主体（用“/”或“与”连接）
   - 若主体分散且不稳定：不强行写具体主体，改用“行业/领域/政策项/事项”类中性对象（必须来自材料用词）
4) 必须体现“核心动作/变化/结果”，优先使用材料中的动词与名词短语进行抽象重组
5) 禁止空泛词与弱描述词，例如：
   - “相关动态”“持续发酵”“引发关注”“市场反应”“最新进展”“多方表态”等
6) 禁止直接复述任意一条 summary 的原句（允许复用关键词，但不得整句照搬）

【输出要求】
仅输出严格 JSON，不得包含任何多余文本，格式如下：

{{
  "main_event_name": "事件级标准化名称",
  "confidence": 0.0,
  "why": "不超过30字，说明基于哪些摘要共性形成命名"
}}

【质量自检（在心中完成，不要输出）】
- 名称放入事件库：是否稳定可复用、不会因个别句子噪声而偏移？
- 名称是否清楚表达“发生了什么”，而不是“有人怎么说”？

""".strip()


# =========================
# 4) 从 Dify 返回提取文本 + JSON 解析
# =========================
def extract_answer_text(dify_resp: Dict[str, Any]) -> str:
    if "error" in dify_resp:
        return ""

    if isinstance(dify_resp.get("answer"), str) and dify_resp["answer"].strip():
        return dify_resp["answer"].strip()

    for k in ["output_text", "message", "text"]:
        v = dify_resp.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return json.dumps(dify_resp, ensure_ascii=False)


def safe_parse_json(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    s = s.strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


# =========================
# 5) 缓存 key + 重试
# =========================
PROMPT_VERSION = "v2_multi_actor_20260208"  # 你每次改 prompt 就改这个

def make_key(news_ids: List[str], summaries: List[str]) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "news_ids": news_ids,
        "summaries": summaries
    }
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def call_with_retry(prompt: str, max_retry: int = 3, sleep_base: float = 1.2) -> Dict[str, Any]:
    last = None
    for i in range(max_retry):
        resp = call_dify_api(prompt)
        last = resp
        if "error" not in resp:
            return resp
        time.sleep(sleep_base * (2 ** i))
    return last or {"error": "unknown"}


# =========================
# 6) Batch 工具：严格每批 ≤ 10
# =========================
def iter_batches(items: List[int], batch_size: int = 10):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# =========================
# 7) 主流程：读 Excel → 分 batch 调用 → 写回
# =========================
def generate_main_event_name_for_excel(
    excel_path: str,
    out_path: str,
    col_newsids: str = "NewsID集合",
    col_summaries: str = "event_summary集合",
    out_col_name: str = "主要事件名称",
    out_col_conf: str = "main_event_confidence",
    out_col_why: str = "main_event_why",
    cache_path: str = "dify_cache_main_event.json",
    batch_size: int = 10,
    batch_sleep: float = 5.0,
):
    df = pd.read_excel(excel_path)

    # ---- load cache ----
    cache: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    # ---- results container ----
    results: Dict[int, Dict[str, Any]] = {
        int(idx): {"name": "", "conf": None, "why": ""} for idx in df.index
    }

    pending_indices: List[int] = []

    # ---- pre-scan (cache hit first) ----
    for idx, row in df.iterrows():
        idx_int = int(idx)
        news_ids = parse_set_cell(row.get(col_newsids))
        summaries = parse_set_cell(row.get(col_summaries))

        # 空行：保留空输出
        if not news_ids and not summaries:
            continue

        key = make_key(news_ids, summaries)

        if key in cache:
            obj = cache[key]
            results[idx_int]["name"] = str(obj.get("main_event_name", "")).strip()
            results[idx_int]["conf"] = obj.get("confidence", None)
            results[idx_int]["why"] = str(obj.get("why", "")).strip()
        else:
            pending_indices.append(idx_int)

    print(f"总行数：{len(df)}")
    print(f"缓存命中：{len(df) - len(pending_indices)}")
    print(f"需要调用 API：{len(pending_indices)}")

    total_batches = (len(pending_indices) - 1) // batch_size + 1 if pending_indices else 0

    # ---- batch calls (<=10 per batch) ----
    for batch_no, batch in enumerate(iter_batches(pending_indices, batch_size), start=1):
        print(f"\n▶ Batch {batch_no}/{total_batches}（{len(batch)} 条）")

        for idx_int in batch:
            row = df.loc[idx_int]
            news_ids = parse_set_cell(row.get(col_newsids))
            summaries = parse_set_cell(row.get(col_summaries))

            prompt = build_prompt(news_ids, summaries)
            resp = call_with_retry(prompt, max_retry=3)
            text = extract_answer_text(resp)

            obj = safe_parse_json(text) or {
                "main_event_name": "",
                "confidence": 0.0,
                "why": "JSON解析失败",
            }

            # 写入 cache + results
            key = make_key(news_ids, summaries)
            cache[key] = obj

            results[idx_int]["name"] = str(obj.get("main_event_name", "")).strip()
            results[idx_int]["conf"] = obj.get("confidence", None)
            results[idx_int]["why"] = str(obj.get("why", "")).strip()

        # 批内完成就落盘（可断点续跑）
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

        if batch_no < total_batches:
            print(f"Batch {batch_no} 完成，sleep {batch_sleep}s")
            time.sleep(batch_sleep)

    # ---- write back ----
    df[out_col_name] = [results[int(i)]["name"] for i in df.index]
    df[out_col_conf] = [results[int(i)]["conf"] for i in df.index]
    df[out_col_why] = [results[int(i)]["why"] for i in df.index]

    df.to_excel(out_path, index=False)
    print(f"\n✅ 完成：{out_path}")
    print(f"✅ 缓存：{cache_path}")


# =========================
# 8) 入口
# =========================
if __name__ == "__main__":
    # 你只需要改这里两个路径
    generate_main_event_name_for_excel(
        excel_path=r"D:\current\今日投资\news_extraction\117_macro_bertopic_event_agg2.xlsx",
        out_path=r"D:\current\今日投资\news_extraction\117_macro_bertopic_output_with_main_event.xlsx",
        col_newsids="NewsID集合",
        col_summaries="event_summary集合",
        batch_size=10,     # <= 10
        batch_sleep=5.0,   # 视你的限流强度调大/调小
    )
