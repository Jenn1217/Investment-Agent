import os
import re
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# =========================
# 配置区（按你的样本数据已匹配）
# =========================
INPUT_CSV =  r'C:\Users\Admin\Desktop\cluster_news\117.csv'        # 输入文件
OUT_DIR = r"C:\Users\Admin\Desktop\cluster_news\outputs117_2"                # 输出目录

SIM_THRESHOLD = 0.8

TEXT_COL = "F4020"   # 原文
TIME_COL = "F4003"   # 发布时间：样本为 "2025-04-02 15:45:02"
MEDIA_COL = "F4024"  # 媒体/渠道：样本为 "eastmoney"
TITLE_COL = "F4001"  # 标题
EXTRA_COL = "F4026"  # 你要求提取的列

DUP_DB_NAME = "重复新闻库.csv"
PATH_DB_NAME = "新闻传播路径.csv"
DEDUP_MAIN_NAME = "原始数据_去重后.csv"


# =========================
# 工具函数
# =========================
def clean_text(x: str) -> str:
    """轻量清洗：合并空白/换行。"""
    if pd.isna(x):
        return ""
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_time_series(series: pd.Series) -> pd.Series:
    """按样本：直接解析完整时间戳。"""
    # errors="coerce"：解析失败会变 NaT（会被排到最后）
    return pd.to_datetime(series.astype(str).str.strip(), errors="coerce", infer_datetime_format=True)


def build_duplicate_groups_by_tfidf(df: pd.DataFrame, text_col: str, sim_threshold: float) -> np.ndarray:
    """
    用 TF-IDF(字符ngram) + 余弦距离半径近邻，找相似文本并聚成“重复组”（并查集）。
    返回每行所属 group_id；非重复为 -1。
    """
    texts = df[text_col].fillna("").astype(str).map(clean_text).tolist()

    # ✅ 针对中文新闻：字符 ngram 更稳定
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(4, 6),
        min_df=1,       # 样本/小批数据不会报错
        max_df=0.98
    )
    X = vectorizer.fit_transform(texts)

    # 余弦相似度 >= threshold  <=> 余弦距离 <= 1-threshold
    radius = 1.0 - sim_threshold
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(X)

    neigh_ind = nn.radius_neighbors(X, radius=radius, return_distance=False)

    # 并查集
    n = len(df)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in neigh_ind[i]:
            if j == i:
                continue
            union(i, j)

    root_to_members = {}
    for i in range(n):
        r = find(i)
        root_to_members.setdefault(r, []).append(i)

    dup_roots = [r for r, mem in root_to_members.items() if len(mem) >= 2]

    group_id = np.full(n, -1, dtype=int)
    for gid, r in enumerate(dup_roots, start=1):
        for idx in root_to_members[r]:
            group_id[idx] = gid

    return group_id


# =========================
# 主流程
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)

    # 必要列检查
    need_cols = [TIME_COL, MEDIA_COL, TEXT_COL, TITLE_COL, EXTRA_COL]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV缺少必要列：{missing}")

    # 时间解析（按样本）
    df["_time"] = parse_time_series(df[TIME_COL])
    df["_time_sort"] = df["_time"].fillna(pd.Timestamp.max)

    # 相似聚类
    df["_group_id"] = build_duplicate_groups_by_tfidf(df, TEXT_COL, SIM_THRESHOLD)

    # ========== 1) 重复新闻库 ==========
    dup_df = df[df["_group_id"] != -1].copy()
    dup_df.sort_values(["_group_id", "_time_sort"], inplace=True)

    dup_out = os.path.join(OUT_DIR, DUP_DB_NAME)
    # ✅ 按你要求：只输出这5列（不额外带 group_id 也可以，但建议带上便于追溯）
    dup_df_out = dup_df[[TIME_COL, MEDIA_COL, TEXT_COL, TITLE_COL, EXTRA_COL, "_group_id"]].copy()
    dup_df_out.to_csv(dup_out, index=False, encoding="utf-8-sig")

    # ========== 2) 原始数据去重：重复组只保留最早时间的一条；同一时间只保留一条 ==========
    if len(dup_df) > 0:
        dup_groups = df[df["_group_id"] != -1].copy()
        dup_groups.sort_values(["_group_id", "_time_sort"], inplace=True)

        # 同一组同一时间多条 => 只保留一条
        dup_groups = dup_groups.drop_duplicates(subset=["_group_id", "_time_sort"], keep="first")

        # 每组只保留最早的一条
        dup_keep = dup_groups.groupby("_group_id", as_index=False).head(1)
        keep_idx = set(dup_keep.index.tolist())

        mask_keep = (df["_group_id"] == -1) | (df.index.isin(keep_idx))
        df_dedup = df[mask_keep].copy()
    else:
        df_dedup = df.copy()

    dedup_out = os.path.join(OUT_DIR, DEDUP_MAIN_NAME)
    df_dedup.drop(columns=["_time", "_time_sort", "_group_id"]).to_csv(dedup_out, index=False, encoding="utf-8-sig")

    # ========== 3) 新闻传播路径 ==========
    # 仅对“同一组内存在至少2个不同发布时间”的组生成路径
    path_rows = []
    if len(dup_df) > 0:
        for gid, g in dup_df.groupby("_group_id"):
            gg = g.copy()
            gg.sort_values("_time_sort", inplace=True)

            # 至少2个不同时间才算传播
            if gg["_time_sort"].nunique(dropna=True) < 2:
                continue

            event_title = gg.iloc[0][TITLE_COL]  # 最早那条标题作为事件标题

            medias = gg[MEDIA_COL].astype(str).fillna("").tolist()
            # 去掉连续重复媒体（可选）
            collapsed = []
            for m in medias:
                m2 = m.strip()
                if not m2:
                    continue
                if len(collapsed) == 0 or collapsed[-1] != m2:
                    collapsed.append(m2)

            path = "——>".join(collapsed)

            path_rows.append({
                "group_id": gid,
                "event_title(F4001_最早)": event_title,
                "start_time(F4003_最早)": gg.iloc[0][TIME_COL],
                "end_time(F4003_最晚)": gg.iloc[-1][TIME_COL],
                "path(F4024)": path,
                "items_in_group": len(gg)
            })

    path_df = pd.DataFrame(path_rows)
    path_out = os.path.join(OUT_DIR, PATH_DB_NAME)
    path_df.to_csv(path_out, index=False, encoding="utf-8-sig")

    print("完成！输出文件：")
    print("1) 重复新闻库：", dup_out)
    print("2) 原始数据_去重后：", dedup_out)
    print("3) 新闻传播路径：", path_out)


if __name__ == "__main__":
    main()
