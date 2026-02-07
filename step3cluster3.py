import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
# 取该 topic 的 topK 关键词（带权重）

def load_stopwords(path: str):
    """读取中文停用词表 stopwords_hit.txt（一行一个词）"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"stopwords file not found: {path}")
    words = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w and not w.startswith("#"):
                words.append(w)
    return words


def load_vec_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError("vec-data.json 顶层必须是 list 或包含 data 的 dict")

    for i, r in enumerate(data[:5]):
        if "NewsID" not in r or "event_summary" not in r:
            raise ValueError(f"第{i}条缺字段：需要 NewsID / event_summary")
        if "embedding" not in r:
            raise ValueError(f"第{i}条缺 embedding")
    return data


def dedupe_records(records):
    seen = set()
    out = []
    for r in records:
        key = (str(r["NewsID"]), (r.get("event_summary") or "").strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def normalize_text_for_match(s: str) -> str:
    """给关键词匹配用：去空白、统一小写、保留中英文数字"""
    s = (s or "").lower()
    s = re.sub(r"\s+", "", s)
    return s


def pick_key_event_by_keywords(topic_model: BERTopic, topic_id: int, summaries: list[str], topk: int = 12):
    """
    重点事件选择（替代 centroid / Representative_Docs）：
    - 取该 topic topk 关键词（含权重）
    - 对每条 summary：命中关键词的权重求和，选最大者
    """
    if not summaries:
        return ""
    if len(summaries) == 1:
        return summaries[0]

    kw = topic_model.get_topic(topic_id) or []  # list[(word, weight)]
    if not kw:
        # 极端情况：没关键词，就退化为“最长的一条”（至少信息量更大）
        return max(summaries, key=lambda x: len(x or ""))

    kw = kw[:topk]
    words = [(w, float(weight)) for w, weight in kw if w]

    best_score = None
    best_summary = summaries[0]

    for s in summaries:
        ns = normalize_text_for_match(s)
        hit = 0
        score = 0.0
        for w, weight in words:
            nw = normalize_text_for_match(w)
            if not nw:
                continue
            if nw in ns:
                hit += 1
                score += weight

        # 轻微奖励“覆盖率”，避免只命中一个高权重词就赢
        coverage = hit / max(1, len(words))
        final = score + 0.15 * coverage

        if best_score is None or final > best_score:
            best_score = final
            best_summary = s

    return best_summary


def run_bertopic(records, stopwords_path="stopwords_hit.txt", min_topic_size=5, language="multilingual"):
    docs = [(r.get("event_summary") or "").strip() for r in records]
    embs = np.array([r["embedding"] for r in records], dtype=np.float32)

    if len(docs) != embs.shape[0]:
        raise ValueError(f"docs数量 {len(docs)} != embeddings行数 {embs.shape[0]}")
    if embs.ndim != 2:
        raise ValueError("embedding 必须是二维矩阵 (n, dim)")

    stopwords = load_stopwords(stopwords_path)
    vectorizer = CountVectorizer(stop_words=stopwords)

    topic_model = BERTopic(
        min_topic_size=min_topic_size,
        language=language,
        vectorizer_model=vectorizer,
        calculate_probabilities=False,
        verbose=False,
    )

    topics, _ = topic_model.fit_transform(docs, embs)
    return topics, docs, embs, topic_model


def build_outputs(records, topics, docs, topic_model: BERTopic, drop_outliers=True, key_topk=12, sample_k=12):
    """
    输出三张表：
    1) event_agg：你要的三列 + topic_id/size
    2) doc_topic_map：每条新闻对应 topic（给后面 LLM/回溯用）
    3) topic_payload_for_llm：每个 topic 一行，直接喂 LLM
    """
    df = pd.DataFrame({
        "NewsID": [str(r["NewsID"]) for r in records],
        "event_summary": docs,
        "topic": topics,
    })

    if drop_outliers:
        df = df[df["topic"] != -1].copy()

    # 2) doc_topic_map
    doc_topic_map = df[["NewsID", "event_summary", "topic"]].copy()

    rows_agg = []
    rows_payload = []

    for topic_id, g in df.groupby("topic", sort=True):
        topic_id = int(topic_id)
        topic_newsids = g["NewsID"].tolist()
        topic_summaries = g["event_summary"].tolist()

        key_event = pick_key_event_by_keywords(topic_model, topic_id, topic_summaries, topk=key_topk)

        # BERTopic 关键词（给 LLM 起名用）
        kw = topic_model.get_topic(topic_id) or []
        kw = kw[:20]
        kw_str = ", ".join([f"{w}({weight:.4f})" for w, weight in kw])

        # 抽样给 LLM（别一次塞太多）
        sample_summaries = topic_summaries[:sample_k]

        rows_agg.append({
            "NewsID集合": ", ".join(topic_newsids),
            "event_summary集合": "\n".join(topic_summaries),
            "重点事件": key_event,
            "topic_id": topic_id,
            "topic_size": int(len(g)),
        })

        rows_payload.append({
            "topic_id": topic_id,
            "topic_size": int(len(g)),
            "topic_keywords": kw_str,
            "key_event_summary": key_event,
            "sample_event_summaries": "\n".join(sample_summaries),
            "news_ids": ", ".join(topic_newsids),
        })

    event_agg = pd.DataFrame(rows_agg).sort_values(["topic_size", "topic_id"], ascending=[False, True]).reset_index(drop=True)
    topic_payload_for_llm = pd.DataFrame(rows_payload).sort_values(["topic_size", "topic_id"], ascending=[False, True]).reset_index(drop=True)

    return event_agg, doc_topic_map, topic_payload_for_llm


def main(
    input_json="data/step2vec-data.json",
    output_xlsx="data/bertopic_event_agg.xlsx",
    stopwords_path="stopwords_hit.txt",
    min_topic_size=5,
    drop_outliers=True,
):
    records = dedupe_records(load_vec_json(input_json))

    topics, docs, embs, topic_model = run_bertopic(
        records,
        stopwords_path=stopwords_path,
        min_topic_size=min_topic_size,
        language="multilingual",
    )

    event_agg, doc_topic_map, topic_payload_for_llm = build_outputs(
        records, topics, docs, topic_model,
        drop_outliers=drop_outliers,
        key_topk=12,
        sample_k=12
    )

    topic_info = topic_model.get_topic_info()

    out_path = Path(output_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        event_agg.to_excel(writer, index=False, sheet_name="event_agg")
        doc_topic_map.to_excel(writer, index=False, sheet_name="doc_topic_map")
        topic_payload_for_llm.to_excel(writer, index=False, sheet_name="topic_payload_for_llm")
        topic_info.to_excel(writer, index=False, sheet_name="topic_info")

    print(f"✅ Done: {out_path.resolve()}")
    print(f"主题数(不含离群): {len(event_agg)}")


if __name__ == "__main__":
    main(
        input_json="data/step2vec-data.json",
        output_xlsx="data/bertopic_event_agg3.xlsx",
        stopwords_path="stopwords_hit.txt",
        min_topic_size=5,
        drop_outliers=True,
    )
