#下面我要对其进行bertopic分析，我的输入内容是data/vec-data.json
#我要按照这个
#参考我.ipynb这个文件里的内容：
# 1. 使用bertopic 
# 2. 输出原文，第一列是newsid多个类似的事件的集合, 第二列是 event_summary的多个类似的事件的集合，第三列是主题 。我这所有的内容都是一个xlsx

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic


def load_vec_json(json_path: str):
    """读取 vec-data.json，返回 records(list[dict])"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 兼容：有的文件可能是 {"data":[...]}
    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    if not isinstance(data, list):
        raise ValueError("vec-data.json 顶层必须是 list 或包含 data 的 dict")

    # 基本校验
    for i, r in enumerate(data[:5]):
        if "NewsID" not in r or "event_summary" not in r:
            raise ValueError(f"第{i}条缺字段：需要 NewsID / event_summary")
        if "embedding" not in r:
            raise ValueError(f"第{i}条缺 embedding（你要用 BERTopic+embedding 必须有）")
    return data


def dedupe_records(records):
    """
    去重：同一 (NewsID, event_summary) 保留一条
    （你贴的例子里 NewsID=5 完全重复，会污染主题）
    """
    seen = set()
    out = []
    for r in records:
        key = (str(r["NewsID"]), (r["event_summary"] or "").strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def pick_key_event(summaries, emb_matrix):
    """
    选重点事件：取 topic 内 embedding 的均值向量作为中心，
    选与中心 cosine 最相近的一条 summary
    """
    if len(summaries) == 0:
        return ""
    if len(summaries) == 1:
        return summaries[0]

    centroid = emb_matrix.mean(axis=0, keepdims=True)  # (1, dim)
    sims = cosine_similarity(emb_matrix, centroid).reshape(-1)  # (n,)
    best_idx = int(np.argmax(sims))
    return summaries[best_idx]


def run_bertopic(records, min_topic_size=5, language="multilingual"):
    """
    用你已有 embedding 跑 BERTopic。
    docs: event_summary
    embeddings: (n, dim)
    """
    docs = [(r.get("event_summary") or "").strip() for r in records]
    embs = np.array([r["embedding"] for r in records], dtype=np.float32)

    if len(docs) != embs.shape[0]:
        raise ValueError(f"docs数量 {len(docs)} != embeddings行数 {embs.shape[0]}")
    if embs.ndim != 2:
        raise ValueError("embedding 必须是二维矩阵 (n, dim)")

    # 你可以把 stopwords 换成你自己的中文停用词表
    vectorizer = CountVectorizer(stop_words=None)

    topic_model = BERTopic(
        min_topic_size=min_topic_size,
        language=language,
        vectorizer_model=vectorizer,
        calculate_probabilities=False,
        verbose=False,
    )

    topics, _ = topic_model.fit_transform(docs, embs)
    return topics, docs, embs, topic_model


def build_aggregation_df(records, topics, docs, embs, drop_outliers=True):
    """
    按 topic 聚合输出三列：
    - newsids（集合）
    - summaries（集合）
    - key_event（重点事件）
    """
    df = pd.DataFrame({
        "NewsID": [str(r["NewsID"]) for r in records],
        "event_summary": docs,
        "topic": topics,
    })

    if drop_outliers:
        df = df[df["topic"] != -1].copy()  # -1 是离群点（不成主题）

    # 保留原始顺序索引，用于取 embedding
    df["row_idx"] = df.index

    rows = []
    for topic_id, g in df.groupby("topic", sort=True):
        idxs = g["row_idx"].to_list()
        topic_embs = embs[idxs, :]
        topic_summaries = g["event_summary"].to_list()
        topic_newsids = g["NewsID"].to_list()

        key_event = pick_key_event(topic_summaries, topic_embs)

        rows.append({
            "NewsID集合": ", ".join(topic_newsids),
            "event_summary集合": "\n".join(topic_summaries),
            "重点事件": key_event,
            "topic_id": int(topic_id),
            "topic_size": int(len(g)),
        })

    out = pd.DataFrame(rows)
    # 可选：按 topic_size 降序，让大主题排前面
    out = out.sort_values(["topic_size", "topic_id"], ascending=[False, True]).reset_index(drop=True)
    return out


def main(
    input_json="data/vec-data.json",
    output_xlsx="data/bertopic_event_agg.xlsx",
    min_topic_size=5,
    drop_outliers=True,
):
    records = load_vec_json(input_json)
    records = dedupe_records(records)

    topics, docs, embs, topic_model = run_bertopic(
        records,
        min_topic_size=min_topic_size,
        language="multilingual",
    )

    agg_df = build_aggregation_df(records, topics, docs, embs, drop_outliers=drop_outliers)

    # 导出 xlsx（同时把 topic_info 也另存一个 sheet，方便你看主题词）
    topic_info = topic_model.get_topic_info()

    out_path = Path(output_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        agg_df.to_excel(writer, index=False, sheet_name="event_agg")
        topic_info.to_excel(writer, index=False, sheet_name="topic_info")

    print(f"✅ Done. 输出文件：{out_path.resolve()}")
    print(f"主题数(不含离群): {len(agg_df)}")


if __name__ == "__main__":
    main(
        input_json="data/step2vec-data.json",
        output_xlsx="data/bertopic_event_agg.xlsx",
        min_topic_size=5,      # 你可以改：3/5/10
        drop_outliers=True,    # True: 不要 -1 离群；False: 也聚合进来
    )
