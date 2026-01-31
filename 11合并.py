# -*- coding: utf-8 -*-
"""
Integrated Trader Analysis Pipeline
Combines entity linking (11zncode.py) with Wall Street Senior Trader analysis (11xycode.py).

Logic:
1. Extract and normalize entities (Companies, Concepts, Industries).
2. Generate a news summary and brief.
3. Perform a "Senior Trader" analysis on the identified subjects.
4. Output a combined JSON result following the style of 11xycode.py.
"""

import os
import re
import json
import time
import argparse
import random
import csv
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple, Iterable
import requests

# -----------------------------
# 0) Constants & Config
# -----------------------------

FORBIDDEN_WORDS = [
    "利好", "利空", "影响", "预期", "推动", "改善", "承压", "提振", "走强", "走弱",
    "大涨", "大跌", "显著", "重磅", "有望", "可能", "或将", "预计", "值得关注",
    "建议", "投资", "买入", "卖出"
]

# -----------------------------
# 1) AI / Dify Client
# -----------------------------

def call_dify_api(
    query: str,
    inputs: Optional[Dict[str, Any]] = None,
    user: str = "integrated_analyzer",
    conversation_id: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 60,
    max_retries: int = 3,
    retry_backoff: float = 1.6,
) -> Dict[str, Any]:
    """
    Encapsulate Dify Chat App call with retries.
    """
    api_key = api_key or os.getenv("DIFY_API_KEY", "app-sfr1VIzn7gHNXbNE9Mvq3jNp")
    base_url = base_url or os.getenv("DIFY_BASE_URL", "https://agent.test.investoday.net/v1").rstrip("/")
    
    if not api_key or not base_url:
        raise RuntimeError("Missing DIFY_API_KEY or DIFY_BASE_URL.")

    url = f"{base_url}/chat-messages"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": inputs or {},
        "query": query,
        "response_mode": "blocking",
        "user": user,
        "conversation_id": conversation_id
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            sleep_s = (retry_backoff ** (attempt - 1)) + random.random() * 0.25
            print(f"[WARN] Dify call failed (attempt {attempt}/{max_retries}): {e}. Sleep {sleep_s:.2f}s")
            time.sleep(sleep_s)

    return {"error": str(last_err)}

# -----------------------------
# 2) Dictionaries & Indices (from 11zncode.py)
# -----------------------------

def read_csv_simple(path: str, encoding: str = "utf-8-sig") -> List[Dict[str, str]]:
    rows = []
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return []
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip().lstrip('\ufeff') for name in (reader.fieldnames or [])]
        for r in reader:
            if not r: continue
            rows.append({k.strip(): (v or "").strip() for k, v in r.items() if k})
    return rows

def normalize_surface(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("【", "[").replace("】", "]")
    return s

@dataclass
class CompanyHit:
    stockCode: str
    FullName: str
    matchedAlias: str

class CompanyAliasIndex:
    def __init__(self):
        self.alias_to_codes: Dict[str, List[str]] = {}
        self.code_to_fullname: Dict[str, str] = {}
        self.code_to_aliases: Dict[str, List[str]] = {}

    def add_alias(self, alias: str, code: str):
        a = normalize_surface(alias)
        if not a: return
        self.alias_to_codes.setdefault(a, [])
        if code not in self.alias_to_codes[a]:
            self.alias_to_codes[a].append(code)
        self.code_to_aliases.setdefault(code, [])
        if a not in self.code_to_aliases[code]:
            self.code_to_aliases[code].append(a)

    @classmethod
    def from_company_old_names_csv(cls, path: str, encoding: str = "utf-8-sig") -> "CompanyAliasIndex":
        rows = read_csv_simple(path, encoding=encoding)
        idx = cls()
        for r in rows:
            full = r.get("FullName", "").strip()
            code = r.get("stockCode", "").strip()
            name = r.get("stockName", "").strip()
            name2 = r.get("stockName2", "").strip()
            if not code: continue
            if full: idx.code_to_fullname[code] = full
            if full: idx.add_alias(full, code)
            if name: idx.add_alias(name, code)
            if name2: idx.add_alias(name2, code)
        return idx

    def lookup(self, surface: str) -> List[CompanyHit]:
        a = normalize_surface(surface)
        codes = self.alias_to_codes.get(a, [])
        return [CompanyHit(stockCode=c, FullName=self.code_to_fullname.get(c, ""), matchedAlias=a) for c in codes]

    def is_ambiguous(self, surface: str) -> bool:
        return len(self.alias_to_codes.get(normalize_surface(surface), [])) > 1

class SimpleNameIndex:
    def __init__(self, name_field: str, code_field: str):
        self.name_field = name_field
        self.code_field = code_field
        self.name_to_code: Dict[str, str] = {}

    @classmethod
    def from_csv(cls, path: str, name_field: str, code_field: str, encoding: str = "utf-8-sig") -> "SimpleNameIndex":
        rows = read_csv_simple(path, encoding=encoding)
        obj = cls(name_field=name_field, code_field=code_field)
        for r in rows:
            name = normalize_surface(r.get(name_field, ""))
            code = (r.get(code_field, "") or "").strip()
            if name and code:
                obj.name_to_code[name] = code
        return obj

    def lookup(self, surface: str) -> Optional[str]:
        return self.name_to_code.get(normalize_surface(surface))

# -----------------------------
# 3) Entity Extraction Logic (from 11zncode.py)
# -----------------------------

@dataclass
class ExtractedEntity:
    etype: str
    surface: str
    evidence: str
    confidence: float

@dataclass
class NormalizedEntity:
    etype: str
    surface: str
    evidence: str
    llm_confidence: float
    stockCode: Optional[str] = None
    fullName: Optional[str] = None
    conceptCode: Optional[str] = None
    industryCode: Optional[str] = None
    decision: str = "REVIEW"
    reason: str = ""

def build_llm_query_extraction(title: str, text: str) -> str:
    instructions = """你是金融新闻实体抽取器。你的目标是从给定文本中抽取实体表面形式（surface），并提供证据（evidence）。
严格规则：
1) 只能抽取文本中“真实出现过”的字符串作为 surface（逐字出现），禁止凭空补全公司全称/代码。
2) 每个实体必须给出 evidence：从原文复制一小段能证明该实体存在的片段（不要超过40字）。
3) 不要输出公司全称、股票代码、概念/行业代码（这些由下游表格回填）。
4) 输出必须是严格 JSON（不要 Markdown，不要解释文字）。
5) 实体类型只允许：company / concept / industry / person
6) 不确定时可以不输出该实体；也可以输出 confidence 较低（0~1）。
"""
    schema = """{
  "companies":[{"surface":str,"evidence":str,"confidence":float}],
  "concepts":[{"surface":str,"evidence":str,"confidence":float}],
  "industries":[{"surface":str,"evidence":str,"confidence":float}],
  "persons":[{"surface":str,"evidence":str,"confidence":float}]
}"""
    return f"{instructions}\n\n请返回JSON，schema如下：\n{schema}\n\n标题：{title}\n正文：\n{text}"

def safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    if not s: return None
    s = s.strip()
    try: return json.loads(s)
    except: pass
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except: return None

def flatten_llm_result(obj: Dict[str, Any]) -> List[ExtractedEntity]:
    res = []
    def take(key: str, etype: str):
        arr = obj.get(key, [])
        if not isinstance(arr, list): return
        for item in arr:
            if not isinstance(item, dict): continue
            surf = (item.get("surface") or "").strip()
            ev = (item.get("evidence") or "").strip()
            try: conf = float(item.get("confidence", 0.0))
            except: conf = 0.0
            if surf: res.append(ExtractedEntity(etype=etype, surface=surf, evidence=ev, confidence=conf))
    take("companies", "company")
    take("concepts", "concept")
    take("industries", "industry")
    take("persons", "person")
    return res

def decide_entity(ent: NormalizedEntity, company_idx: CompanyAliasIndex) -> NormalizedEntity:
    ev_len = len((ent.evidence or "").strip())
    evs = "strong" if ev_len >= 18 else ("medium" if ev_len >= 8 else "weak")
    surf = normalize_surface(ent.surface)
    generic_bad = {"机器人", "新能源", "芯片", "医药", "白酒", "银行", "券商", "保险", "地产", "板块", "概念", "行业", "公司", "集团"}
    
    mapped = (ent.etype == "company" and ent.stockCode) or \
             (ent.etype == "concept" and ent.conceptCode) or \
             (ent.etype == "industry" and ent.industryCode)

    if ent.etype == "person":
        ent.decision = "KEEP" if evs in ("medium", "strong") and ent.llm_confidence >= 0.5 else "REVIEW"
        return ent

    if mapped:
        if evs == "weak" and ent.llm_confidence < 0.35:
            ent.decision = "REVIEW"
            ent.reason = "mapped but low conf/weak evidence"
        else:
            ent.decision = "KEEP"
            ent.reason = "mapped to dict"
        if ent.etype == "company" and company_idx.is_ambiguous(ent.surface):
            ent.decision = "REVIEW"
            ent.reason = "ambiguous alias"
        return ent

    if surf in generic_bad and evs != "strong":
        ent.decision = "DROP"
        return ent

    if evs == "strong" and ent.llm_confidence >= 0.55:
        ent.decision = "REVIEW"
        ent.reason = "strong evidence but OOV"
    elif evs == "weak" and ent.llm_confidence < 0.45:
        ent.decision = "DROP"
    else:
        ent.decision = "REVIEW"
    return ent

def normalize_entities(extracted: List[ExtractedEntity], c_idx, ct_idx, i_idx) -> List[NormalizedEntity]:
    out = []
    for e in extracted:
        ne = NormalizedEntity(etype=e.etype, surface=e.surface, evidence=e.evidence, llm_confidence=e.confidence)
        if e.etype == "company":
            hits = c_idx.lookup(e.surface)
            if len(hits) == 1:
                ne.stockCode = hits[0].stockCode
                ne.fullName = hits[0].FullName or None
        elif e.etype == "concept":
            code = ct_idx.lookup(e.surface)
            if code: ne.conceptCode = code
        elif e.etype == "industry":
            code = i_idx.lookup(e.surface)
            if code: ne.industryCode = code
        out.append(decide_entity(ne, c_idx))
    return out

# -----------------------------
# 4) Analysis Dimensions & Trader Logic (from 11xycode.py)
# -----------------------------

def build_trader_prompt(content: str, subject: str) -> str:
    # prompt exact copy from 11xycode.py with adjusted formatting for subject
    return f"""
        # Role
        你是一名拥有20年经验的华尔街高级交易员。你的核心任务是提取新闻中的趋势逻辑，识别正负要素的博弈，并将其浓缩为一行带箭头的推演链。同时，你需要综合评估事件的影响力维度。

        # Task
        分析[Input News]，针对[Target Subject]（及相关方）生成分析结果。

        # Analysis Dimensions (综合影响力评估)
        请综合考虑以下三个维度，并将其浓缩为一个“综合影响分析”字段：
        1. **影响范围 (Scope)**: 个股 / 板块 / 全市场。
        2. **影响深度 (Depth)**: 短期情绪 / 中期业绩 / 长期逻辑重构。
        3. **新颖度 (Novelty)**: 旧闻炒作 / 预期内进展 / 超预期突发。

        # Example (One-Shot Learning)
        ---------------------------------------------------
        [输入新闻]: 
        受美联储加息预期影响，国际金价今日大跌2%，但四川黄金表示其矿山开采成本极低，且产量稳步提升。
        
        [输出 JSON]:
        {{
            "analyses": [
                {{
                    "subject_name": "国际金价",
                    "summary": "受加息预期压制，金价承压下跌。",
                    "event_type": "宏观数据 - 价格波动",
                    "sentiment": "利空",
                    "trend_one_line": "趋势承压: 加息预期升温 -> 资金流出贵金属 -> 金价短期回调",
                    "impact_summary": "全市场/短期/预期内: 加息预期主要影响短期资金面，属于市场已知风险的释放。",
                    "investment_advice": "观望"
                }},
                {{
                    "subject_name": "四川黄金",
                    "summary": "成本优势对冲金价下跌影响。",
                    "event_type": "经营数据 - 成本披露",
                    "sentiment": "弱利好",
                    "trend_one_line": "以量补价: 金价下跌(负) + 产量提升(正) -> 低成本护园河 -> 利润韧性强",
                    "impact_summary": "个股/中期/预期内: 成本优势构建中期业绩护城河，非突发利好。",
                    "investment_advice": "逢低吸纳"
                }}
            ]
        }}
        ---------------------------------------------------

        # Requirements
        1. **趋势分析 (trend_one_line)**: 必须是**一行字符串**。格式严格为：`趋势形态: 因素A -> 传导B -> 结果C`。
        2. **综合影响 (impact_summary)**: 必须包含三个维度的综合评估。格式建议为：`范围/深度/新颖度: 一句话简评`。
        3. **情感**: 仅输出 强利好/弱利好/中性/弱利空/强利空。
        4. **投资建议**: 必须明确给出 操作指令（买入/卖出/观望）。

        # Output Schema (JSON Array)
        请务必返回 JSON 格式：
        {{
            "analyses": [
                {{
                    "subject_name": "主体名",
                    "summary": "一句话总结",
                    "event_type": "事件类型",
                    "sentiment": "利好/利空", 
                    "trend_one_line": "趋势形态: A -> B -> C",
                    "impact_summary": "范围/深度/新颖度: 简评",
                    "investment_advice": "买入/卖出/观望"
                }}
            ]
        }}

        ## 当前任务输入
        [Input News]:
        {content}

        [Target Subject Hints]:
        {subject}
        """

# -----------------------------
# 5) Summary Generation (from 11zncode.py)
# -----------------------------

def build_summary_query(news_text: str, anchors: Optional[Dict[str, List[str]]] = None) -> str:
    anchor_block = ""
    if anchors:
        parts = []
        if anchors.get("companies"): parts.append(f"公司/机构：{'、'.join(anchors['companies'][:6])}")
        if anchors.get("concepts"): parts.append(f"概念：{'、'.join(anchors['concepts'][:6])}")
        if anchors.get("industries"): parts.append(f"行业：{'、'.join(anchors['industries'][:6])}")
        if parts: anchor_block = "\n【实体锚点（仅供参考）】\n" + "\n".join(parts) + "\n"
    
    forbidden = "、".join(FORBIDDEN_WORDS)
    return f"你是一个金融新闻助手。请基于【新闻正文】生成摘要。\n要求：\n1. 禁止使用：{forbidden}\n2. 严格 JSON 格式：{{ \"summary\": {{ \"news_summary\": \"...\", \"one_sentence_brief\": \"...\" }} }}\n3. brief 不超过30字。\n{anchor_block}\n【新闻正文】\n{news_text}"

# -----------------------------
# 6) Main Pipeline
# -----------------------------

def process_file_integrated(input_file, api_key, base_url, company_idx, concept_idx, industry_idx):
    if input_file.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(input_file)
    else:
        try: df = pd.read_csv(input_file, encoding='utf-8')
        except: df = pd.read_csv(input_file, encoding='gbk')

    results = []
    print(f"开始处理 {len(df)} 条新闻...")

    for idx, row in df.iterrows():
        news_id = idx + 1
        title = str(row.get('F4001', '')).strip()
        content = str(row.get('F4020', '')).strip()
        timestamp = str(row.get('F4003', '')).strip()
        raw_source = str(row.get('F4024', ''))
        link_url = str(row.get('F4026', ''))
        final_source = 'eastmoney' if '123456' in raw_source and 'eastmoney' in link_url else (raw_source if raw_source != 'nan' else '')

        print(f"  [{news_id}] 分析中: {title[:30]}...")

        # Step 1: Entity Extraction
        extract_query = build_llm_query_extraction(title, content)
        extract_res = call_dify_api(extract_query, api_key=api_key, base_url=base_url)
        
        entities = []
        keep_names = []
        if "answer" in extract_res:
            obj = safe_json_load(extract_res["answer"])
            if obj:
                extracted = flatten_llm_result(obj)
                normalized = normalize_entities(extracted, company_idx, concept_idx, industry_idx)
                entities = [asdict(ne) for ne in normalized]
                keep_names = [ne.surface for ne in normalized if ne.decision == "KEEP"]

        # Step 2: Summary Generation
        anchors = {"companies": [], "concepts": [], "industries": []}
        for ent in entities:
            if ent["decision"] == "KEEP":
                if ent["etype"] == "company": anchors["companies"].append(ent["surface"])
                elif ent["etype"] == "concept": anchors["concepts"].append(ent["surface"])
                elif ent["etype"] == "industry": anchors["industries"].append(ent["surface"])
        
        summary_query = build_summary_query(f"{title}\n{content}", anchors)
        summary_res = call_dify_api(summary_query, api_key=api_key, base_url=base_url)
        news_summary = {}
        if "answer" in summary_res:
            s_obj = safe_json_load(summary_res["answer"])
            if s_obj and "summary" in s_obj:
                news_summary = s_obj["summary"]

        # Step 3: Senior Trader Analysis
        # If no KEEP entities found, use a generic subject or skip? Let's use found entities or tile
        target_subject = ", ".join(keep_names) if keep_names else "全市场"
        trader_query = build_trader_prompt(content, target_subject)
        trader_res = call_dify_api(trader_query, api_key=api_key, base_url=base_url)
        
        ai_analysis = {}
        if "answer" in trader_res:
            t_obj = safe_json_load(trader_res["answer"])
            if t_obj:
                ai_analysis = t_obj

        # Merge results in 11xycode.py style
        record = {
            "NewsID": news_id,
            "timestamp": timestamp,
            "title": title,
            "source": final_source,
            "input_content": content,
            "target_subject": target_subject,
            "entities": entities,  # Added from 11zncode.py
            "summary": news_summary, # Added from 11zncode.py
            "ai_analysis": ai_analysis # Core from 11xycode.py
        }
        results.append(record)
        time.sleep(0.5)

    return results

if __name__ == "__main__":
    # Configuration
    API_URL = "https://agent.test.investoday.net/v1"
    API_KEY = "app-sfr1VIzNp"
    
    # Files
    INPUT_FILE = r"/Users/renzhenni/Library/Mobile Documents/com~apple~CloudDocs/学习/2026.01.29 今日投资/source/raw-news/df_200.csv"
    COMPANY_CSV = r"/Users/renzhenni/Library/Mobile Documents/com~apple~CloudDocs/学习/2026.01.29 今日投资/source/industry/company-list-with-old-names.csv"
    CONCEPT_CSV = r"/Users/renzhenni/Library/Mobile Documents/com~apple~CloudDocs/学习/2026.01.29 今日投资/source/industry/concept-list.csv"
    INDUSTRY_CSV = r"/Users/renzhenni/Library/Mobile Documents/com~apple~CloudDocs/学习/2026.01.29 今日投资/source/industry/industry-list.csv"
    
    OUTPUT_FILE = '综合分析结果_最终版.json'

    print("正在加载字典...")
    c_idx = CompanyAliasIndex.from_company_old_names_csv(COMPANY_CSV)
    ct_idx = SimpleNameIndex.from_csv(CONCEPT_CSV, "conceptName", "conceptCode")
    i_idx = SimpleNameIndex.from_csv(INDUSTRY_CSV, "industryName", "industryCode")

    if os.path.exists(INPUT_FILE):
        final_results = process_file_integrated(INPUT_FILE, API_KEY, API_URL, c_idx, ct_idx, i_idx)
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)
        
        print(f"\n[成功] 数据已保存至: {OUTPUT_FILE}")
        
        # Preview similar to 11xycode.py
        if final_results:
            print("\n" + "="*50)
            print("处理完成！预览首条分析：")
            print("="*50)
            first = final_results[0]
            print(f"标题: {first['title']}")
            print(f"主体: {first['target_subject']}")
            print(f"分析: {json.dumps(first['ai_analysis'], ensure_ascii=False, indent=2)}")
    else:
        print(f"错误: 找不到文件 {INPUT_FILE}")
