# -*- coding: utf-8 -*-
"""
MVP: step2_normalized_events2.jsonl (NewsID + canonicals) -> bucket -> embed(Investoday) -> HDBSCAN -> Union-Find
Outputs:
  - out/step2_canonicals_clustered.jsonl
  - out/step2_clusters_summary.csv

Env:
  INVESTODAY_API_KEY, INVESTODAY_BASE_URL, INVESTODAY_EMBED_MODEL
"""

from __future__ import annotations
import os, json, math, hashlib
from typing import Any, Dict, List, Tuple, Set, DefaultDict
from collections import defaultdict

import numpy as np
import pandas as pd
from openai import OpenAI


# ---------------------------
# Config
# ---------------------------
INPUT_JSONL = "source/step2/step2_normalized_events3.jsonl"
OUT_JSONL = "out/step2_canonicals_clustered.jsonl"
OUT_SUMMARY = "out/step2_clusters_summary.csv"

BASE_URL = os.getenv("INVESTODAY_BASE_URL", "https://agent.investoday.net/model/v1")
MODEL = os.getenv("INVESTODAY_EMBED_MODEL", "text-embedding-v4")
API_KEY = os.getenv("INVESTODAY_API_KEY")

TIME_WINDOW = "W"  # "W" 周 / "M" 月
HDBSCAN_MIN_CLUSTER_SIZE = 3
HDBSCAN_MIN_SAMPLES = 2
EMBED_BATCH_SIZE = 10
#报错原因已查明：你使用的 text-embedding-v4 向量模型在当前接口（agent.investoday.net）上有一个限制，即 单次请求的批处理大小（Batch Size）不得超过 10。原本代码中设置的 128 触发了 API 的 400 错误。

# ---------------------------
# Utils
# ---------------------------
def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()

def sha1_12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def to_time_bucket(date_str: str, window: str = "W") -> str:
    s = safe_str(date_str)
    if not s:
        return "UNKNOWN"
    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        return "UNKNOWN"
    if window.upper() == "M":
        return dt.strftime("%Y-%m")
    iso = dt.isocalendar()
    return f"{int(iso.year)}-W{int(iso.week):02d}"


# ---------------------------
# Read jsonl -> flat canonicals
# ---------------------------
def load_flat_canonicals(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        # 尝试读取整个文件作为 JSON (处理标准 JSON 数组)
        content = f.read().strip()
        if not content:
            return []
        
        try:
            # 尝试作为全量 JSON 加载
            data = json.loads(content)
            if isinstance(data, list):
                for obj in data:
                    news_id = obj.get("NewsID")
                    for c in (obj.get("canonicals") or []):
                        rec = dict(c)
                        rec["NewsID"] = news_id
                        records.append(rec)
                return records
        except json.JSONDecodeError:
            # 如果失败，则按 JSONL (逐行) 尝试
            pass
            
        # 回退到 JSONL 模式
        f.seek(0)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                news_id = obj.get("NewsID")
                for c in (obj.get("canonicals") or []):
                    rec = dict(c)
                    rec["NewsID"] = news_id
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


# ---------------------------
# Participants -> tokens
# ---------------------------
def participant_tokens(rec: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    parts = rec.get("participants") or {}
    if not isinstance(parts, dict):
        return out

    for c in parts.get("companies", []) or []:
        if isinstance(c, dict):
            code = safe_str(c.get("stockCode"))
            name = safe_str(c.get("name"))
            if code: out.add(f"C:{code}")
            if name: out.add(f"CNAME:{name}")

    for it in parts.get("industries", []) or []:
        if isinstance(it, dict):
            code = safe_str(it.get("stockCode")) or safe_str(it.get("code"))
            name = safe_str(it.get("name"))
            if code: out.add(f"I:{code}")
            if name: out.add(f"INAME:{name}")
        else:
            name = safe_str(it)
            if name: out.add(f"INAME:{name}")

    for ct in parts.get("concepts", []) or []:
        if isinstance(ct, dict):
            code = safe_str(ct.get("stockCode")) or safe_str(ct.get("code"))
            name = safe_str(ct.get("name"))
            if code: out.add(f"K:{code}")
            if name: out.add(f"KNAME:{name}")
        else:
            name = safe_str(ct)
            if name: out.add(f"KNAME:{name}")

    return out


def build_embed_text(rec: Dict[str, Any]) -> str:
    htype = safe_str(rec.get("high_level_type"))
    trig = safe_str(rec.get("trigger_norm"))
    act = safe_str(rec.get("core_action_norm"))
    sent = safe_str(rec.get("sentiment_coarse")) or safe_str(rec.get("sentiment_final"))

    toks = sorted(list(participant_tokens(rec)))[:30]
    pieces = [
        f"TYPE={htype}",
        f"TRIG={trig}",
        f"ACT={act}",
        f"SENT={sent}",
        "PARTS=" + ",".join(toks),
    ]
    pieces = [p for p in pieces if not p.endswith("=") and p != "PARTS="]
    return " | ".join(pieces)


# ---------------------------
# Embedding (your way)
# ---------------------------
class InvestodayEmbedder:
    def __init__(self, api_key: str, base_url: str, model: str):
        if not api_key:
            raise RuntimeError("Missing INVESTODAY_API_KEY env var.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed(self, texts: List[str]) -> np.ndarray:
        import time
        vectors: List[List[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start:start + EMBED_BATCH_SIZE]
            
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    resp = self.client.embeddings.create(model=self.model, input=batch)
                    vectors.extend([item.embedding for item in resp.data])
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        print(f"[Warning] Embedding failed (Attempt {attempt+1}): {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise e

        mat = np.asarray(vectors, dtype=np.float32)
        return l2_normalize(mat).astype(np.float32)


# ---------------------------
# Union-Find
# ---------------------------
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# ---------------------------
# Bucketing + HDBSCAN
# ---------------------------
def build_buckets(records: List[Dict[str, Any]], window: str = "W") -> Dict[Tuple[str, str, str], List[int]]:
    """
    新桶规则（更粗，减少碎桶）：
    bucket_key = (high_level_type, time_bucket, anchor)
    anchor 优先级：公司code > 概念code > 行业code > NONE
    并且忽略 '相关动态提及' 这种泛trigger作为锚点来源（只是不让它影响桶，不影响向量文本本身）
    """
    buckets: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)

    for idx, rec in enumerate(records):
        htype = safe_str(rec.get("high_level_type")) or "UNKNOWN"
        tb = to_time_bucket(safe_str(rec.get("date")), window=window)

        parts = rec.get("participants") or {}
        comps = (parts.get("companies") or []) if isinstance(parts, dict) else []
        concs = (parts.get("concepts") or []) if isinstance(parts, dict) else []
        inds  = (parts.get("industries") or []) if isinstance(parts, dict) else []

        # 选一个“主锚点”而不是把每个token都建桶（关键：减少桶数量）
        anchor = "__NONE__"

        # 公司锚点
        for x in comps:
            if isinstance(x, dict):
                code = safe_str(x.get("stockCode"))
                if code:
                    anchor = f"C:{code}"
                    break

        # 概念锚点
        if anchor == "__NONE__":
            for x in concs:
                if isinstance(x, dict):
                    code = safe_str(x.get("stockCode")) or safe_str(x.get("code"))
                    if code:
                        anchor = f"K:{code}"
                        break

        # 行业锚点
        if anchor == "__NONE__":
            for x in inds:
                if isinstance(x, dict):
                    code = safe_str(x.get("stockCode")) or safe_str(x.get("code"))
                    if code:
                        anchor = f"I:{code}"
                        break

        buckets[(htype, tb, anchor)].append(idx)

    return buckets


def run_hdbscan(vectors_norm: np.ndarray) -> np.ndarray:
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(vectors_norm).astype(np.int32)


def cluster(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    uf = UnionFind(n)

    embedder = InvestodayEmbedder(api_key=API_KEY, base_url=BASE_URL, model=MODEL)
    buckets = build_buckets(records, window=TIME_WINDOW)

    stats = {
        "n_events": n,
        "n_buckets": len(buckets),
        "processed_buckets": 0,
        "skipped_small_buckets": 0,
        "total_unions": 0,
    }

    for _, indices in buckets.items():
        if len(indices) < max(HDBSCAN_MIN_CLUSTER_SIZE, 3):
            stats["skipped_small_buckets"] += 1
            continue

        texts = [build_embed_text(records[i]) for i in indices]
        vec = embedder.embed(texts)
        labels = run_hdbscan(vec)

        rep: Dict[int, int] = {}
        for local_pos, lab in enumerate(labels.tolist()):
            if lab < 0:
                continue
            gidx = indices[local_pos]
            if lab not in rep:
                rep[lab] = gidx
            else:
                uf.union(rep[lab], gidx)
                stats["total_unions"] += 1

        stats["processed_buckets"] += 1

    root_to_members: DefaultDict[int, List[int]] = defaultdict(list)
    for i in range(n):
        root_to_members[uf.find(i)].append(i)

    for root, members in root_to_members.items():
        seed = safe_str(records[root].get("canonical_event_id")) or f"root:{root}"
        cid = sha1_12(seed + "|" + str(len(members)))
        for i in members:
            records[i]["global_cluster_id"] = cid
            records[i]["global_cluster_size"] = len(members)

    stats["n_global_clusters"] = len(root_to_members)
    return {"records": records, "stats": stats}


def write_outputs(records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)

    # jsonl
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # summary csv
    df = pd.DataFrame(records)
    df["date_dt"] = pd.to_datetime(df.get("date"), errors="coerce")
    g = df.groupby("global_cluster_id", as_index=False).agg(
        cluster_size=("global_cluster_size", "max"),
        first_date=("date_dt", "min"),
        last_date=("date_dt", "max"),
        news_ids=("NewsID", lambda s: sorted(set([int(x) for x in s.dropna().tolist()]))),
        sample=("trigger_norm", "first"),
        hlt=("high_level_type", "first"),
        act=("core_action_norm", "first"),
    )
    g = g.sort_values(["cluster_size", "first_date"], ascending=[False, True]).reset_index(drop=True)
    g.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")


def main():
    records = load_flat_canonicals(INPUT_JSONL)
    if not records:
        raise RuntimeError("No canonicals loaded. Check INPUT_JSONL path/content.")

    result = cluster(records)
    write_outputs(result["records"])

    print("[done] wrote:", OUT_JSONL)
    print("[done] wrote:", OUT_SUMMARY)
    print("[stats]", json.dumps(result["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
