# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 15:56:41 2026

@author: HUAWEI
"""
# -*- coding: utf-8 -*-
"""
Neo4j 知识图谱构建（无 GDS）：
- 噪声topic过滤 + 行业归一 + NPMI归一化
- 写入 Neo4j：News / Company / Industry / MicroTopic / MacroTopic 与关系
- 写入 Industry-Industry 关系：CO_OCCUR {npmi, cooc, evidenceNewsIds}
- 社区发现：NetworkX greedy_modularity_communities，写回 Industry.communityId

依赖：
pip install neo4j pandas openpyxl numpy networkx

运行前请设置环境变量（示例）：
Windows PowerShell:
  $env:NEO4J_URI="bolt://localhost:7687"
  $env:NEO4J_USER="neo4j"
  $env:NEO4J_PASS="你的新密码"

macOS/Linux:
  export NEO4J_URI="bolt://localhost:7687"
  export NEO4J_USER="neo4j"
  export NEO4J_PASS="你的新密码"
"""

import os, json, re, math, time
from itertools import combinations
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import networkx as nx
from neo4j import GraphDatabase


# =========================
# 0) 配置：文件路径 & 参数
# =========================
JSON_PATH  = r"D:\current\今日投资\news_extraction\117_whole.json"
MICRO_XLSX = r"D:\current\今日投资\news_extraction\117_micro_bertopic_event_agg2.xlsx"
MACRO_XLSX = r"D:\current\今日投资\news_extraction\117_macro_bertopic_event_agg2.xlsx"
MAP_CSV    = r"D:\current\今日投资\知识图谱\上市公司行业分类表filtered_P0218.csv"

# 共现网络过滤阈值（越大越稳，越小越灵敏）
MIN_COOC = 2
MIN_NPMI = 0.05

# 噪声 topic 规则（你可按业务继续加词）
NOISE_PATTERNS = [
    r"融资|融券|两融|融资余额|融资净买入|融券余额|融券净卖出",
    r"龙虎榜|主力资金|北向资金|南向资金|资金流入|资金流出|净流入|净流出",
    r"成交额|成交量|换手率|涨跌幅|收盘|开盘|盘中|大盘|指数|ETF|板块异动",
    r"风险提示|免责声明|据统计|数据显示|截至.*?收盘",
]
noise_re = re.compile("|".join(f"(?:{p})" for p in NOISE_PATTERNS))

# 批量写入大小
BATCH = 2000


# =========================
# 1) Neo4j 连接（环境变量）
# =========================
def get_driver():
    uri = os.environ["NEO4J_URI"]
    user = os.environ.get("NEO4J_USER", "neo4j")
    pwd = os.environ["NEO4J_PASS"]
    return GraphDatabase.driver(uri, auth=(user, pwd))


# =========================
# 2) Cypher：约束/索引
# =========================
CY_CONSTRAINTS = [
    "CREATE CONSTRAINT news_id IF NOT EXISTS FOR (n:News) REQUIRE n.newsId IS UNIQUE",
    "CREATE CONSTRAINT company_code IF NOT EXISTS FOR (c:Company) REQUIRE c.stockCode IS UNIQUE",
    "CREATE CONSTRAINT industry_name IF NOT EXISTS FOR (i:Industry) REQUIRE i.name IS UNIQUE",
    "CREATE CONSTRAINT micro_tid IF NOT EXISTS FOR (t:MicroTopic) REQUIRE t.topicId IS UNIQUE",
    "CREATE CONSTRAINT macro_tid IF NOT EXISTS FOR (t:MacroTopic) REQUIRE t.topicId IS UNIQUE",
]

def ensure_constraints(driver):
    with driver.session() as s:
        for q in CY_CONSTRAINTS:
            s.run(q)


# =========================
# 3) 读取数据
# =========================
def load_data():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        news = json.load(f)
    micro = pd.read_excel(MICRO_XLSX)
    macro = pd.read_excel(MACRO_XLSX)
    mp = pd.read_csv(MAP_CSV)
    return news, micro, macro, mp


# =========================
# 4) 噪声 topic 判定
# =========================
def is_noise_topic(row) -> bool:
    text = f"{row.get('重点事件','')}\n{row.get('event_summary集合','')}"
    hits = len(noise_re.findall(str(text)))
    size = int(row.get("topic_size", 0)) if pd.notna(row.get("topic_size", 0)) else 0
    # 命中>=2 直接噪声；或 topic 特别大且命中>=1
    return (hits >= 2) or (size >= 80 and hits >= 1)


# =========================
# 5) 行业归一：显式行业 -> 标准 IndustryName
# =========================
def build_industry_normalizer(mp: pd.DataFrame):
    mp = mp.copy()
    mp["Symbol_str"] = mp["Symbol"].astype(str).str.zfill(6)
    sym2ind = mp.set_index("Symbol_str")["IndustryName"].to_dict()
    std_industries = sorted(set(sym2ind.values()))
    std_set = set(std_industries)

    def normalize(surface: str):
        if not isinstance(surface, str) or not surface.strip():
            return None
        s = surface.strip()
        if s in std_set:
            return s

        # 去掉常见后缀
        s2 = re.sub(r"(行业|板块|概念)$", "", s)
        if s2 in std_set:
            return s2

        # 启发式包含匹配（谨慎）
        cands = []
        for std in std_industries:
            if len(std) < 2:
                continue
            if std in s2 or s2 in std:
                cands.append(std)
        if not cands:
            return None
        cands = sorted(cands, key=lambda x: (abs(len(x) - len(s2)), len(x)))
        return cands[0]

    return sym2ind, normalize


# =========================
# 6) 从 JSON 展平主体（company/industry，丢 concept）
# =========================
def flatten_entities(news):
    rows = []
    for item in news:
        nid = str(item.get("NewsID"))
        title = item.get("title")
        ts = item.get("timestamp")
        for ev in item.get("events", []):
            et = ev.get("etype")
            if et not in ("company", "industry"):
                continue
            rows.append({
                "NewsID": nid,
                "title": title,
                "timestamp": ts,
                "etype": et,
                "surface": ev.get("surface"),
                "stockCode": str(ev.get("stockCode")) if ev.get("stockCode") is not None else None,
            })
    return pd.DataFrame(rows)


# =========================
# 7) 组装每条新闻的行业集合（公司映射行业 + 归一后的显式行业）
# =========================
def build_news_industries(df, sym2ind, norm_industry):
    df = df.copy()
    df["company_industry"] = np.where(
        df["etype"] == "company",
        df["stockCode"].map(sym2ind),
        None
    )
    df["surface_industry_std"] = np.where(
        df["etype"] == "industry",
        df["surface"].map(norm_industry),
        None
    )

    news_industries = {}
    for nid, grp in df.groupby("NewsID"):
        inds = set()

        c_inds = grp.loc[(grp["etype"] == "company") & grp["company_industry"].notna(), "company_industry"]
        inds |= set(c_inds.astype(str).tolist())

        s_inds = grp.loc[(grp["etype"] == "industry") & grp["surface_industry_std"].notna(), "surface_industry_std"]
        inds |= set(s_inds.astype(str).tolist())

        news_industries[str(nid)] = set(i for i in inds if i and i != "nan")

    return df, news_industries


# =========================
# 8) 解析 topic：NewsID -> topicId；并生成 contexts（过滤噪声）
# =========================
def parse_nids(nid_str):
    return [x.strip() for x in str(nid_str).split(",") if x.strip()]

def build_contexts(topic_df: pd.DataFrame, topic_type: str, news_industries: dict):
    # topic_type: "micro" / "macro"
    tdf = topic_df.copy()
    tdf["is_noise"] = tdf.apply(is_noise_topic, axis=1)

    contexts = []
    meta = []  # 保存每个 context 的 topicId 和 news_ids（用于证据）
    for _, r in tdf.iterrows():
        if bool(r["is_noise"]):
            continue
        tid = int(r["topic_id"])
        nids = parse_nids(r["NewsID集合"])
        inds = set()
        for nid in nids:
            inds |= news_industries.get(str(nid), set())
        if len(inds) >= 2:
            contexts.append(set(inds))
            meta.append({"topic_type": topic_type, "topic_id": tid, "news_ids": nids})
    return tdf, contexts, meta


# =========================
# 9) 计算 NPMI 产业边 + 证据新闻列表
# =========================
def calc_npmi_edges(contexts, contexts_meta, news_industries, nid2title):
    T = len(contexts)
    ind_count = Counter()
    pair_count = Counter()

    for inds in contexts:
        inds = sorted(set(inds))
        for i in inds:
            ind_count[i] += 1
        for a, b in combinations(inds, 2):
            pair_count[(a, b)] += 1

    def npmi(c_ij, c_i, c_j, total):
        if c_ij <= 0:
            return None
        p_ij = c_ij / total
        p_i = c_i / total
        p_j = c_j / total
        pmi = math.log(p_ij / (p_i * p_j))
        return pmi / (-math.log(p_ij))

    edges = []
    for (a, b), c_ij in pair_count.items():
        v = npmi(c_ij, ind_count[a], ind_count[b], T)
        if v is None:
            continue
        edges.append({
            "industry_a": a,
            "industry_b": b,
            "cooc_contexts": int(c_ij),
            "freq_a": int(ind_count[a]),
            "freq_b": int(ind_count[b]),
            "npmi": float(v),
        })
    edges_df = pd.DataFrame(edges).sort_values(["npmi", "cooc_contexts"], ascending=False)

    # 证据：边 -> 支撑它的 newsId 集合
    edge2news = defaultdict(set)
    for meta in contexts_meta:
        # 这个 context 内所有 news 的行业并集
        inds = set()
        for nid in meta["news_ids"]:
            inds |= news_industries.get(str(nid), set())
        inds = sorted(inds)
        for a, b in combinations(inds, 2):
            edge2news[(a, b)].update(str(x) for x in meta["news_ids"])

    def top_titles(nids, k=12):
        out = []
        for nid in list(nids)[:300]:
            t = nid2title.get(str(nid))
            if isinstance(t, str) and t.strip():
                out.append(t.strip())
            if len(out) >= k:
                break
        return out

    evidence_rows = []
    for _, r in edges_df.iterrows():
        a, b = r["industry_a"], r["industry_b"]
        nids = sorted(edge2news.get((a, b), set()))
        evidence_rows.append({
            "industry_a": a,
            "industry_b": b,
            "npmi": r["npmi"],
            "cooc_contexts": r["cooc_contexts"],
            "evidence_news_count": len(nids),
            "evidence_news_ids": ",".join(nids[:500]),  # 避免过大
            "top_evidence_titles": " || ".join(top_titles(nids, 12)),
        })
    evidence_df = pd.DataFrame(evidence_rows).sort_values(["npmi", "cooc_contexts"], ascending=False)

    return edges_df, evidence_df


# =========================
# 10) 社区发现（无 GDS）：NetworkX greedy modularity
# =========================
def detect_communities(edges_df: pd.DataFrame):
    F = edges_df[(edges_df["cooc_contexts"] >= MIN_COOC) & (edges_df["npmi"] >= MIN_NPMI)].copy()

    G = nx.Graph()
    for _, r in F.iterrows():
        G.add_edge(r["industry_a"], r["industry_b"], weight=float(r["npmi"]), cooc=int(r["cooc_contexts"]))

    if G.number_of_nodes() == 0:
        return G, pd.DataFrame(columns=["industry", "communityId", "community_size"])

    comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
    ind2comm = {}
    for cid, comm in enumerate(comms):
        for ind in comm:
            ind2comm[ind] = cid

    clusters = pd.DataFrame({
        "industry": list(ind2comm.keys()),
        "communityId": [ind2comm[i] for i in ind2comm.keys()],
    })
    clusters["community_size"] = clusters["communityId"].map(clusters["communityId"].value_counts().to_dict())
    clusters = clusters.sort_values(["community_size", "communityId", "industry"], ascending=[False, True, True])

    return G, clusters


# =========================
# 11) Neo4j 写入：通用批量 UNWIND
# =========================
def chunked(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf

def run_write(driver, cypher, rows):
    with driver.session() as s:
        for batch in chunked(rows, BATCH):
            s.execute_write(lambda tx: tx.run(cypher, rows=batch))

# 节点/关系写入 Cypher（全部 UNWIND）
CY_UPSERT_NEWS = """
UNWIND $rows AS r
MERGE (n:News {newsId: r.newsId})
SET n.title = r.title,
    n.timestamp = r.timestamp
"""

CY_UPSERT_COMPANY = """
UNWIND $rows AS r
MERGE (c:Company {stockCode: r.stockCode})
SET c.name = r.name
"""

CY_UPSERT_INDUSTRY = """
UNWIND $rows AS r
MERGE (i:Industry {name: r.name})
"""

CY_UPSERT_BELONGS = """
UNWIND $rows AS r
MATCH (c:Company {stockCode: r.stockCode})
MERGE (i:Industry {name: r.industryName})
MERGE (c)-[:BELONGS_TO]->(i)
"""

CY_UPSERT_MENTION = """
UNWIND $rows AS r
MATCH (n:News {newsId: r.newsId})
MERGE (i:Industry {name: r.industryName})
MERGE (i)-[:MENTIONED_IN]->(n)
"""

CY_UPSERT_MICRO_TOPIC = """
UNWIND $rows AS r
MERGE (t:MicroTopic {topicId: r.topicId})
SET t.isNoise = r.isNoise,
    t.size = r.size
"""

CY_UPSERT_MACRO_TOPIC = """
UNWIND $rows AS r
MERGE (t:MacroTopic {topicId: r.topicId})
SET t.isNoise = r.isNoise,
    t.size = r.size
"""

CY_LINK_NEWS_MICRO = """
UNWIND $rows AS r
MATCH (n:News {newsId: r.newsId})
MATCH (t:MicroTopic {topicId: r.topicId})
MERGE (n)-[:IN_MICRO_TOPIC]->(t)
"""

CY_LINK_NEWS_MACRO = """
UNWIND $rows AS r
MATCH (n:News {newsId: r.newsId})
MATCH (t:MacroTopic {topicId: r.topicId})
MERGE (n)-[:IN_MACRO_TOPIC]->(t)
"""

CY_UPSERT_INDUSTRY_EDGE = """
UNWIND $rows AS r
MERGE (a:Industry {name: r.industry_a})
MERGE (b:Industry {name: r.industry_b})
MERGE (a)-[e:CO_OCCUR]->(b)
SET e.npmi = r.npmi,
    e.cooc = r.cooc_contexts,
    e.evidenceNewsIds = r.evidence_news_ids
"""

CY_WRITE_COMMUNITY = """
UNWIND $rows AS r
MATCH (i:Industry {name: r.industry})
SET i.communityId = r.communityId,
    i.communitySize = r.community_size
"""


# =========================
# 12) 主流程：计算 -> 写入 Neo4j
# =========================
def main():
    print("Loading data...")
    news, micro, macro, mp = load_data()

    nid2title = {str(item.get("NewsID")): item.get("title") for item in news}

    print("Building normalizer...")
    sym2ind, norm_industry = build_industry_normalizer(mp)

    print("Flattening entities...")
    df = flatten_entities(news)

    print("Building news->industries...")
    df, news_industries = build_news_industries(df, sym2ind, norm_industry)

    print("Building contexts (noise-filtered micro/macro)...")
    micro2, micro_ctx, micro_meta = build_contexts(micro, "micro", news_industries)
    macro2, macro_ctx, macro_meta = build_contexts(macro, "macro", news_industries)

    contexts = micro_ctx + macro_ctx
    contexts_meta = micro_meta + macro_meta
    print(f"Contexts used: {len(contexts)}")

    print("Calculating NPMI edges + evidence...")
    edges_df, evidence_df = calc_npmi_edges(contexts, contexts_meta, news_industries, nid2title)

    print("Detecting communities (no GDS)...")
    G, clusters_df = detect_communities(edges_df)

    # 你也可以本地保存一份
    edges_df.to_csv("industry_edges_npmi.csv", index=False, encoding="utf-8-sig")
    evidence_df.to_excel("industry_edge_evidence.xlsx", index=False)
    clusters_df.to_csv("industry_clusters.csv", index=False, encoding="utf-8-sig")

    # 生成写入 Neo4j 的数据
    print("Preparing rows for Neo4j...")

    # News 节点
    news_rows = []
    seen_news = set()
    for item in news:
        nid = str(item.get("NewsID"))
        if nid in seen_news:
            continue
        seen_news.add(nid)
        news_rows.append({
            "newsId": nid,
            "title": item.get("title"),
            "timestamp": item.get("timestamp"),
        })

    # Company 节点（从 df 里提取）
    company_rows = []
    for code, grp in df[df["etype"]=="company"].groupby("stockCode"):
        if not isinstance(code, str) or not re.fullmatch(r"\d{6}", code):
            continue
        # name：用 surface 最常见的
        name = grp["surface"].dropna().astype(str).value_counts().index[0] if grp["surface"].notna().any() else code
        company_rows.append({"stockCode": code, "name": name})

    # Industry 节点（标准行业全集：来自公司映射；再加上显式行业归一成功的）
    industry_set = set(sym2ind.values())
    industry_set |= set(df["surface_industry_std"].dropna().astype(str).tolist())
    industry_rows = [{"name": x} for x in sorted(industry_set) if x and x != "nan"]

    # BELONGS_TO（公司行业表）
    belongs_rows = [{"stockCode": k, "industryName": v} for k, v in sym2ind.items()]

    # MENTIONED_IN（每条新闻出现的行业：使用 news_industries）
    mention_rows = []
    for nid, inds in news_industries.items():
        for ind in inds:
            mention_rows.append({"newsId": nid, "industryName": ind})

    # Topics 节点
    micro_rows = [{"topicId": int(r["topic_id"]), "isNoise": bool(r["is_noise"]), "size": int(r.get("topic_size", 0))}
                  for _, r in micro2.iterrows()]
    macro_rows = [{"topicId": int(r["topic_id"]), "isNoise": bool(r["is_noise"]), "size": int(r.get("topic_size", 0))}
                  for _, r in macro2.iterrows()]

    # News-Topic 关系（全量链接；噪声topic也可以留着，方便审计）
    link_micro_rows = []
    for _, r in micro2.iterrows():
        tid = int(r["topic_id"])
        for nid in parse_nids(r["NewsID集合"]):
            link_micro_rows.append({"newsId": str(nid), "topicId": tid})

    link_macro_rows = []
    for _, r in macro2.iterrows():
        tid = int(r["topic_id"])
        for nid in parse_nids(r["NewsID集合"]):
            link_macro_rows.append({"newsId": str(nid), "topicId": tid})

    # Industry-Industry 边：用 evidence_df（包含 npmi + cooc + evidence ids）
    edge_rows = []
    for _, r in evidence_df.iterrows():
        edge_rows.append({
            "industry_a": r["industry_a"],
            "industry_b": r["industry_b"],
            "npmi": float(r["npmi"]),
            "cooc_contexts": int(r["cooc_contexts"]),
            "evidence_news_ids": str(r["evidence_news_ids"]) if pd.notna(r["evidence_news_ids"]) else "",
        })

    # Community 写回
    comm_rows = []
    for _, r in clusters_df.iterrows():
        comm_rows.append({
            "industry": r["industry"],
            "communityId": int(r["communityId"]),
            "community_size": int(r["community_size"]),
        })

    # 过滤 Industry-Industry 边（只写入“可用网络”）
    edge_rows_filtered = [x for x in edge_rows if (x["cooc_contexts"] >= MIN_COOC and x["npmi"] >= MIN_NPMI)]

    # ===== 写入 Neo4j =====
    print("Connecting Neo4j...")
    driver = get_driver()
    try:
        print("Ensuring constraints...")
        ensure_constraints(driver)

        print("Writing News...")
        run_write(driver, CY_UPSERT_NEWS, news_rows)

        print("Writing Company...")
        run_write(driver, CY_UPSERT_COMPANY, company_rows)

        print("Writing Industry...")
        run_write(driver, CY_UPSERT_INDUSTRY, industry_rows)

        print("Writing BELONGS_TO...")
        run_write(driver, CY_UPSERT_BELONGS, belongs_rows)

        print("Writing MENTIONED_IN...")
        run_write(driver, CY_UPSERT_MENTION, mention_rows)

        print("Writing Topics...")
        run_write(driver, CY_UPSERT_MICRO_TOPIC, micro_rows)
        run_write(driver, CY_UPSERT_MACRO_TOPIC, macro_rows)

        print("Linking News-Topic...")
        run_write(driver, CY_LINK_NEWS_MICRO, link_micro_rows)
        run_write(driver, CY_LINK_NEWS_MACRO, link_macro_rows)

        print("Writing Industry-Industry CO_OCCUR edges (filtered)...")
        run_write(driver, CY_UPSERT_INDUSTRY_EDGE, edge_rows_filtered)

        print("Writing communities back to Industry...")
        run_write(driver, CY_WRITE_COMMUNITY, comm_rows)

        print("\n✅ Done.")
        print(f"- News: {len(news_rows)}")
        print(f"- Company: {len(company_rows)}")
        print(f"- Industry: {len(industry_rows)}")
        print(f"- Industry edges (written): {len(edge_rows_filtered)}")
        print(f"- Communities: {clusters_df['communityId'].nunique() if len(clusters_df)>0 else 0}")

        print("\nLocal outputs saved:")
        print("- industry_edges_npmi.csv")
        print("- industry_edge_evidence.xlsx")
        print("- industry_clusters.csv")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
