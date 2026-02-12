# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 14:22:33 2026

@author: HUAWEI
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 14:22:33 2026
@author: HUAWEI
"""

import json
import os
from sentence_transformers import SentenceTransformer

# ======================
# 配置
# ======================
MODEL_PATH = r"D:\models\bge-m3"  # ✅ 你的本地 bge-m3 路径
IN_PATH = r"D:\current\今日投资\news_extraction\117_macro_extract.json"
OUT_PATH = r"D:\current\今日投资\news_extraction\117_macro_ver.json"

BATCH_SIZE = 32
DEVICE = "cpu"  # 有 NVIDIA GPU 可改成 "cuda"

# ======================
# 1. 读取输入数据
# ======================
with open(IN_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    raise ValueError("输入 JSON 顶层必须是 list")

texts = []
meta = []

for item in data:
    summary = (item.get("event_summary") or "").strip()
    if summary:
        texts.append(summary)
        meta.append({
            "NewsID": item.get("NewsID"),
            "event_summary": summary
        })

if not texts:
    raise ValueError("输入数据中没有可用的 event_summary（全部为空或缺失）")

# ======================
# 2. 加载本地模型（bge-m3）
# ======================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型路径不存在：{MODEL_PATH}")

model = SentenceTransformer(
    MODEL_PATH,
    device=DEVICE
)

# ======================
# 3. 向量化
# ======================
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    normalize_embeddings=True,
    show_progress_bar=True
)

# ======================
# 4. 写出结果
# ======================
results = []
for i, vec in enumerate(embeddings):
    results.append({
        **meta[i],
        "embedding_model": "bge-m3",
        "embedding_dim": int(len(vec)),
        "embedding": vec.tolist()
    })

out_dir = os.path.dirname(OUT_PATH)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False)

print(f"✅ 向量化完成，输出到 {OUT_PATH}")
print(f"样本数: {len(results)} | 向量维度: {results[0]['embedding_dim']}")
