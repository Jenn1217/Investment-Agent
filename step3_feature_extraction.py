# -*- coding: utf-8 -*-
"""
Integrated Event-Centric Pipeline (V4 Modified - Split Surface & Top-Level Metrics & Summary)
功能：
1. 实体提取：基于本地字典与LLM提取。
2. 拆分显示：解决"一句话多主体"问题，将同事件主体拆分为多条独立的事件记录。
3. 层级调整：Scope, Depth, Novelty 移动到新闻最外层。
4. 新增功能：增加全篇新闻的 Summary (摘要)。
5. 优化控制：严格限制 Trigger 字数，防止长句；优化兜底 Trigger 文案。
"""

import os
import re
import json
import time
import csv
import pandas as pd
from typing import Optional, Dict, Any, List
import requests

# -----------------------------
# 0) 配置与常量
# -----------------------------
FORBIDDEN_WORDS = [
    "利好", "利空", "影响", "预期", "推动", "改善", "承压", "提振", "走强", "走弱",
    "大涨", "大跌", "显著", "重磅", "有望", "可能", "或将", "预计", "值得关注",
    "建议", "投资", "买入", "卖出"
]

# -----------------------------
# 1) Dify 客户端与通用工具
# -----------------------------
def call_dify_api(query: str, api_key: str, base_url: str) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat-messages"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "inputs": {},
        "query": query,
        "response_mode": "blocking",
        "user": "integrated_event_analyst"
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    if not s: return None
    try:
        return json.loads(s)
    except:
        match = re.search(r"```json\s*(\{.*?\})\s*```", s, re.S)
        if match:
            try: return json.loads(match.group(1))
            except: pass
        
        match_raw = re.search(r"\{.*\}", s, re.S)
        if match_raw:
            try: return json.loads(match_raw.group(0))
            except: pass
    return None

def normalize_surface(s: str) -> str:
    s = re.sub(r"\s+", "", (s or "").strip())
    return s.replace("（", "(").replace("）", ")").replace("【", "[").replace("】", "]")

# -----------------------------
# 2) 字典索引逻辑
# -----------------------------
class CompanyAliasIndex:
    def __init__(self):
        self.alias_to_codes = {}

    @classmethod
    def from_csv(cls, path: str):
        idx = cls()
        if not os.path.exists(path): return idx
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                code = (r.get("stockCode") or "").strip()
                if not code: continue
                for fld in ["FullName", "stockName", "stockName2"]:
                    val = normalize_surface(r.get(fld, ""))
                    if val:
                        idx.alias_to_codes.setdefault(val, [])
                        if code not in idx.alias_to_codes[val]: idx.alias_to_codes[val].append(code)
        return idx

    def lookup(self, s: str):
        a = normalize_surface(s)
        codes = self.alias_to_codes.get(a, [])
        return [{"stockCode": c, "matchedAlias": a} for c in codes]

class SimpleNameIndex:
    def __init__(self, name_field: str, code_field: str):
        self.name_to_code = {}
        self.nf, self.cf = name_field, code_field

    @classmethod
    def from_csv(cls, path: str, name_f: str, code_f: str):
        obj = cls(name_f, code_f)
        if not os.path.exists(path): return obj
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                n, c = normalize_surface(r.get(name_f, "")), (r.get(code_f, "") or "").strip()
                if n and c: obj.name_to_code[n] = c
        return obj

    def lookup(self, s: str):
        return self.name_to_code.get(normalize_surface(s))

# -----------------------------
# 3) 提示词工程 (Prompts)
# -----------------------------
def build_extraction_prompt(title: str, content: str) -> str:
    """第一步：实体提取提示词"""
    return f"""你是金融新闻实体抽取器。
严格规则：
1) 只能抽取文中出现的字符串作为 surface。
2) 实体类型只允许：company / concept / industry。
3) 输出必须是严格 JSON。

请返回如下格式：
{{
  "companies": [{{"surface": "...", "confidence": 0.9}}],
  "concepts": [{{"surface": "...", "confidence": 0.9}}],
  "industries": [{{"surface": "...", "confidence": 0.9}}]
}}

标题：{title}
正文：{content}"""

def build_event_analysis_prompt(content: str, news_time: str, subjects_info: List[Dict]) -> str:
    """
    第二步：事件深度分析提示词
    修改点：
    1. 增加 article_summary 字段要求。
    2. 严格限制 Trigger 的字数和格式，解决长句问题。
    """
    subjects_json = json.dumps(subjects_info, ensure_ascii=False)
    forbidden = "、".join(FORBIDDEN_WORDS)
    
    return f"""# Role
你是一位量化交易系统的信息预处理专家。

# Task
阅读新闻，识别文中发生的关键事件。
**特别注意**：如果一个事件同时涉及名单中的多个主体（例如“A公司、B公司和C公司均发布了业绩预告”），请务必将它们聚合在同一个事件对象中。

[参考新闻发布时间]: {news_time}
[指定关注主体名单]: {subjects_json}
[新闻正文]: {content}

# Extraction Rules
1. **聚合原则**：如果多个主体在同一句话中被提及，且发生的事情相同，必须放入 `subjects` 列表中。
2. **Trigger (核心触发词)**：
   - 目标：输出**唯一一个**最核心的“触发短语”，作为算法可读取的“原子化信号”。
   - **严格限制**：**字数必须在 4-12 个字之间**。
   - **格式**：[主体]+[动作/指标]。**严禁**包含逗号、顿号或完整句子。
   - 错误示例：“中金公司发布观点，认为市场向好”（太长、有逗号）。
   - 正确示例：“中金发布研报”、“光伏板块反弹”、“宁德时代业绩预增”。
   - 禁止使用词汇：{forbidden}。
3. **Core Action**: 提取核心动词（如：涨停、注资、减持）。严禁提取“提及”、“表示”等中性词。
4. **Time**: 推导具体日期 YYYY-MM-DD。

# Analysis Dimensions
请对该事件进行多维度评级：
- **scope (范围)**: [个股事件 / 板块共振 / 宏观政策 / 市场情绪]。
- **depth (深度)**: [高 / 中 / 低]。
    * 高：有确凿数据(涨跌幅/金额)、实质性公告或重大政策变动。
    * 中：有观点分析或普通业务动态。
    * 低：单纯提及、传闻或无实质内容的复盘。
- **novelty (新颖度)**: [突发新事 / 持续发酵 / 旧闻重提]。

# Output Format (JSON)
必须严格输出以下 JSON 结构。**新增 article_summary 字段**：
{{
    "article_summary": "新闻的一句话摘要，概括全文核心内容（30字以内）",
    "event_analyses": [
        {{
            "subjects": ["主体A", "主体B"],
            "event_time": "YYYY-MM-DD",
            "trigger": "核心短语(限12字)",
            "core_action": "...",
            "event_summary": "...",
            "scope": "...",
            "depth": "...",
            "novelty": "...",
            "sentiment": "强利好/弱利好/中性/弱利空/强利空",
            "trend_one_line": "...",
            "investment_advice": "买入/卖出/观望"
        }}
    ]
}}"""

# -----------------------------
# 4) 主流水线逻辑
# -----------------------------
def process_pipeline(input_file, api_key, base_url, indices, base_number = 118, start_index = 1500 ):
    c_idx, ct_idx, i_idx = indices

    
    if input_file.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)
        
    results = []

    for idx, row in df.iterrows():
        news_id = f"{base_number}_{start_index + idx + 1}"
        title = str(row.get('F4001', '')).strip()
        content = str(row.get('F4020', '')).strip()
        news_time = str(row.get('F4003', '')).strip()
        
        raw_source = str(row.get('F4024', ''))
        link_url = str(row.get('F4026', ''))
        final_source = 'eastmoney' if '123456' in raw_source and 'eastmoney' in link_url else (raw_source if raw_source != 'nan' else '')

        print(f"Processing News ID: {news_id}...")

        # --- 步骤 1: 实体提取 ---
        extract_query = build_extraction_prompt(title, content)
        extract_resp = call_dify_api(extract_query, api_key, base_url)
        
        keep_entities_map = {}
        keep_entities_list = []
        
        if "answer" in extract_resp:
            ext_data = safe_json_load(extract_resp["answer"])
            if ext_data:
                for etype, key in [("company", "companies"), ("concept", "concepts"), ("industry", "industries")]:
                    for item in ext_data.get(key, []):
                        surf = item.get("surface")
                        if not surf: continue
                        
                        ent_obj = {"etype": etype, "surface": surf, "stockCode": None, "conceptCode": None, "industryCode": None}
                        
                        matched = False
                        if etype == "company":
                            hits = c_idx.lookup(surf)
                            if len(hits) == 1:
                                ent_obj["stockCode"] = hits[0]["stockCode"]
                                matched = True
                        elif etype == "concept":
                            code = ct_idx.lookup(surf)
                            if code: 
                                ent_obj["conceptCode"] = code
                                matched = True
                        elif etype == "industry":
                            code = i_idx.lookup(surf)
                            if code: 
                                ent_obj["industryCode"] = code
                                matched = True
                        
                        if matched:
                            keep_entities_list.append(ent_obj)
                            keep_entities_map[surf] = ent_obj

        # --- 步骤 2: 事件深度分析 ---
        final_events = []
        
        # 初始化最外层指标
        top_scope = "暂无"
        top_depth = "暂无"
        top_novelty = "暂无"
        top_summary = "暂无摘要" # 初始化 Summary

        if keep_entities_list:
            analysis_query = build_event_analysis_prompt(content, news_time, keep_entities_list)
            analysis_resp = call_dify_api(analysis_query, api_key, base_url)
            
            ai_data = safe_json_load(analysis_resp.get("answer", "")) if "answer" in analysis_resp else {}
            
            # --- 新增：提取 Article Summary ---
            if ai_data and "article_summary" in ai_data:
                top_summary = ai_data["article_summary"]
            
            ai_events = ai_data.get("event_analyses", [])
            
            # 提取 Top-Level 指标
            if ai_events:
                first_evt = ai_events[0]
                top_scope = first_evt.get("scope", "暂无")
                top_depth = first_evt.get("depth", "暂无")
                top_novelty = first_evt.get("novelty", "暂无")
            
            processed_surfaces = set()
            event_counter = 1

            for event in ai_events:
                subjects = event.get("subjects", [])
                if isinstance(subjects, str): subjects = [subjects]
                
                valid_subjects = [s for s in subjects if s in keep_entities_map]
                subjects_to_process = valid_subjects if valid_subjects else subjects

                # 拆分模式
                for subj in subjects_to_process:
                    current_type = "unknown"
                    current_code = None
                    if subj in keep_entities_map:
                        meta = keep_entities_map[subj]
                        current_type = meta["etype"]
                        current_code = meta.get("stockCode") or meta.get("conceptCode") or meta.get("industryCode")
                        processed_surfaces.add(subj) 
                    
                    captured_time = event.get("event_time")
                    if not captured_time or "YYYY" in captured_time:
                        captured_time = news_time

                    single_event = {
                        "event_ID": f"E_auto_{news_id}_{event_counter}",
                        "etype": current_type,
                        "surface": subj,
                        "stockCode": current_code,
                        "event_time": captured_time,
                        
                        "trigger": event.get("trigger", "暂无"),
                        "core_action": event.get("core_action", "提及"),
                        "event_summary": event.get("event_summary", "文中提及该主体"),
                        "sentiment": event.get("sentiment", "中性"),
                        "trend_one_line": event.get("trend_one_line", "暂无"),
                        "investment_advice": event.get("investment_advice", "观望")
                    }
                    final_events.append(single_event)
                    event_counter += 1

            # --- 兜底逻辑 ---
            # 您提到的 "trigger: 文中提及" 就是这里生成的
            # 我将其修改为 "相关动态提及"，以区别于 AI 生成的 Trigger
            for ent in keep_entities_list:
                if ent["surface"] not in processed_surfaces:
                    final_events.append({
                        "event_ID": f"E_auto_{news_id}_fallback_{event_counter}",
                        "etype": ent["etype"],
                        "surface": ent["surface"],
                        "stockCode": ent.get("stockCode") or ent.get("conceptCode") or ent.get("industryCode"),
                        "event_time": news_time,
                        "trigger": "相关动态提及", # 修改点：替换"文中提及"
                        "event_summary": "文中提及该主体，未识别到独立事件",
                        "sentiment": "中性",
                    })
                    event_counter += 1

        target_subjects_str = ", ".join([e['surface'] for e in final_events])
        
        results.append({
            "NewsID": news_id,
            "timestamp": news_time,
            "title": title,
            "source": final_source,
            
            # --- 最外层指标 ---
            "scope": top_scope,
            "depth": top_depth,
            "novelty": top_novelty,
            "summary": top_summary, # 新增 summary 字段
            # ----------------
            
            "target_subject": target_subjects_str,
            "events": final_events
        })
        time.sleep(0.5)

    return results

if __name__ == "__main__":
    # 配置信息
    URL = "https://agent.test.investoday.net/v1"
    KEY = "app-sfr1VIzn7gHNXbNE9Mvq3jNp"
    
    C_PATH = "曾用名.csv"
    CT_PATH = "概念列表.csv"
    I_PATH = "行业列表.csv"
    
    
    if os.path.exists(C_PATH):
        c_index = CompanyAliasIndex.from_csv(C_PATH)
        ct_index = SimpleNameIndex.from_csv(CT_PATH, "conceptName", "conceptCode")
        i_index = SimpleNameIndex.from_csv(I_PATH, "industryName", "industryCode")
        
        if os.path.exists("去重后118_2000.csv"):
            final_output = process_pipeline("去重后118_2000.csv", KEY, URL, (c_index, ct_index, i_index))
            
            with open('去重后118_2000.json', 'w', encoding='utf-8') as f:
                json.dump(final_output, f, ensure_ascii=False, indent=4)
            print("处理完成，结果已保存")
        else:
            print("未找到 df_5.csv")
    else:
        print("未找到字典文件，请检查路径。")