# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 10:49:07 2026

@author: 123
"""

# 第一步：一句信息里包含着两个内容，这两个内容是一定要放在一起的，从原文中提取他俩。
#"NewsID": 5,"event_summary": "国家发改委明确将在电力、交通运输等领域率先突破，推进全国统一大市场建设，出台高含金量政策举措。",
# 原文：[
#输入
# 输出内容名称为data-extract.json
#我下一步要做的是要对这些内容进行向量化，
#
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: str) -> Any:
    """
    读取 JSON 文件并解析为 Python 对象。

    参数：
        path: JSON 文件路径（例如 'data/raw-data.json'）

    返回：
        解析后的对象，可能是 list / dict / 其他 JSON 类型
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_get_str(d: Dict[str, Any], key: str) -> Optional[str]:
    """
    安全读取 dict 里的字符串字段。

    返回：
        - 如果字段存在且是非空字符串：返回该字符串（strip 后）
        - 否则：返回 None
    """
    v = d.get(key, None)
    if isinstance(v, str):
        v = v.strip()
        return v if v else None
    return None


def extract_newsid_event_summaries(raw: Any) -> List[Dict[str, Any]]:
    """
    从原始数据中提取 (NewsID, event_summary) 成对信息。

    额外要求（已实现）：
        1) 删除 event_summary == "文中提及该主体，未识别到独立事件" 的记录
        2) 同一个 NewsID 内 event_summary 去重，只保留一次
    """
    if not isinstance(raw, list):
        raise ValueError("raw-data.json 顶层必须是 list（每个元素是一条新闻 dict）。")

    out: List[Dict[str, Any]] = []

    # 需要剔除的占位/无效摘要
    DROP_SUMMARY = "文中提及该主体，未识别到独立事件"

    for idx, news in enumerate(raw):
        if not isinstance(news, dict):
            # 跳过异常结构
            continue

        news_id = news.get("NewsID", None)
        # NewsID 必须存在，否则这条无法“成对”
        if news_id is None:
            continue

        events = news.get("events", [])
        if not isinstance(events, list):
            # events 异常就当没有
            events = []

        # ✅ 用于“同一 NewsID 内 event_summary 去重”
        seen_summaries = set()

        for ev in events:
            if not isinstance(ev, dict):
                continue

            event_summary = safe_get_str(ev, "event_summary")
            if event_summary is None:
                # 没有 summary 就不输出（因为你要求必须从原文提取这对信息）
                continue

            # ✅ 1) 删除掉指定占位文本
            if event_summary == DROP_SUMMARY:
                continue

            # ✅ 2) 同一 NewsID 内 event_summary 去重（只保留第一次出现）
            if event_summary in seen_summaries:
                continue
            seen_summaries.add(event_summary)

            record = {
                "NewsID": news_id,
                "event_summary": event_summary,

                # 下面这些是可选的“溯源字段”，不破坏你要的“两个内容放一起”
                # "event_ID": ev.get("event_ID"),
                # "etype": ev.get("etype"),
                # "event_time": ev.get("event_time"),
            }
            out.append(record)

    return out


def save_json(path: str, data: Any) -> None:
    """
    把 Python 对象写回 JSON 文件。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main(
    input_path: str = r"C:\Users\123\Desktop\cluster\117_全天_宏观.json",
    output_path: str = r"C:\Users\123\Desktop\cluster\117_macro_extract.json",
) -> None:
    """
    主流程函数：读取 → 提取 → 保存
    """
    raw = load_json(input_path)
    extracted = extract_newsid_event_summaries(raw)
    save_json(output_path, extracted)

    # 运行后立刻看到效果：输出多少条 pair
    print(f"[OK] 输入新闻条数: {len(raw) if isinstance(raw, list) else 'N/A'}")
    print(f"[OK] 输出 (NewsID, event_summary) 记录数: {len(extracted)}")
    if extracted:
        print("[SAMPLE] 第一条输出示例：")
        print(json.dumps(extracted[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
