import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
#取该 topic 的 第一条

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
            raise ValueError(f"第{i}条缺 embedding（你要用 BERTopic+embedding 必须有）")
    return data


def dedupe_records(records):
    seen = set()
    out = []
    for r in records:
        key = (str(r["NewsID"]), (r["event_summary"] or "").strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def get_builtin_cn_stopwords():
    """
    不是“中文停用词文件”，只是为了别让关键词全是虚词。
    你如果完全不想要停用词，后面 vectorizer 里改为 stop_words=None 即可。
    """
    return [
        "的","了","和","与","及","或","等","在","对","将","已","为","是","也","更","较","并","还","又","而","让","把",
        "进行","推进","推动","加强","加快","进一步","持续","不断","相关","有关","方面","领域","工作","表示","强调","明确",
        "出台","发布","提出","指出","称","认为","预计","可能","将会","通过","提供","支持","政策","措施"
    ]


def run_bertopic(records, min_topic_size=5, language="multilingual", use_builtin_stopwords=True):
    docs = [(r.get("event_summary") or "").strip() for r in records]
    embs = np.array([r["embedding"] for r in records], dtype=np.float32)

    if len(docs) != embs.shape[0]:
        raise ValueError(f"docs数量 {len(docs)} != embeddings行数 {embs.shape[0]}")
    if embs.ndim != 2:
        raise ValueError("embedding 必须是二维矩阵 (n, dim)")

    # ✅ 不需要外部中文停用词文件：用内置简版（可关闭）
    stopwords = get_builtin_cn_stopwords() if use_builtin_stopwords else None
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


def topic_keywords(topic_model, topic_id: int, top_n: int = 6):
    """
    用 c-TF-IDF 的 top keywords 作为“第三列：重点事件/主题标签”
    返回形如：'统一大市场 | 发改委 | 电力 | 交通运输 | ...'
    """
    if topic_id == -1:
        return "OUTLIER"
    words = topic_model.get_topic(topic_id)  # List[(word, weight)]
    if not words:
        return ""
    return " | ".join([w for (w, _) in words[:top_n]])


def build_aggregation_df(records, topics, docs, topic_model, drop_outliers=True, top_n_keywords=6):
    df = pd.DataFrame({
        "NewsID": [str(r["NewsID"]) for r in records],
        "event_summary": docs,
        "topic": topics,
    })

    if drop_outliers:
        df = df[df["topic"] != -1].copy()

    rows = []
    for topic_id, g in df.groupby("topic", sort=True):
        newsids = g["NewsID"].to_list()
        summaries = g["event_summary"].to_list()

        key_col = topic_keywords(topic_model, int(topic_id), top_n=top_n_keywords)

        rows.append({
            "NewsID集合": ", ".join(newsids),
            "event_summary集合": "\n".join(summaries),
            "重点事件": key_col,              # ✅ 改为关键词
            "topic_id": int(topic_id),
            "topic_size": int(len(g)),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["topic_size", "topic_id"], ascending=[False, True]).reset_index(drop=True)
    return out


def main(
    input_json = r"D:\current\今日投资\news_extraction\117_macro_ver.json",
    output_xlsx = r"D:\current\今日投资\news_extraction\117_macro_bertopic_event_agg2.xlsx",
    min_topic_size = 5,
    drop_outliers=True,
    top_n_keywords=6,
    use_builtin_stopwords=True,
):
    records = load_vec_json(input_json)
    records = dedupe_records(records)

    topics, docs, embs, topic_model = run_bertopic(
        records,
        min_topic_size=min_topic_size,
        language="multilingual",
        use_builtin_stopwords=use_builtin_stopwords,
    )

    agg_df = build_aggregation_df(
        records,
        topics,
        docs,
        topic_model,
        drop_outliers=drop_outliers,
        top_n_keywords=top_n_keywords,
    )

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
        input_json=r"D:\current\今日投资\news_extraction\117_macro_ver.json",
        output_xlsx=r"D:\current\今日投资\news_extraction\117_macro_bertopic_event_agg2.xlsx",
        min_topic_size=5,
        drop_outliers=True,
        top_n_keywords=6,            # 第三列关键词数量
        use_builtin_stopwords=True,  # 不想要停用词就改 False
    )
