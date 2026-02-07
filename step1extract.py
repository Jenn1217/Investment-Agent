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

    这个函数做什么：
        - 打开文件
        - 用 json.load 解析
        - 返回解析结果
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_get_str(d: Dict[str, Any], key: str) -> Optional[str]:
    """
    安全读取 dict 里的字符串字段。

    参数：
        d: 一个 dict
        key: 想取的字段名

    返回：
        - 如果字段存在且是非空字符串：返回该字符串（strip 后）
        - 否则：返回 None

    这个函数做什么：
        - 避免 KeyError
        - 避免 event_summary 不是字符串（比如 None、数字、list）
        - 避免空字符串污染输出
    """
    v = d.get(key, None)
    if isinstance(v, str):
        v = v.strip()
        return v if v else None
    return None


def extract_newsid_event_summaries(raw: Any) -> List[Dict[str, Any]]:
    """
    从原始数据中提取 (NewsID, event_summary) 成对信息。

    输入 raw 预期形态：
        raw 是一个 list，元素是每条新闻 dict（你给的示例就是这样）

    输出：
        List[Dict]:
            每条记录至少包含：
            - NewsID
            - event_summary
        并额外附带 event_ID / etype / event_time（方便你排查来源，不影响“两个内容必须一起”）

    这个函数做什么（核心逻辑）：
        - 遍历每条 news
        - 取出 NewsID
        - 遍历 news["events"] 里的每个 event
        - 把 event["event_summary"] 抽出来
        - 形成输出记录
    """
    if not isinstance(raw, list):
        raise ValueError("raw-data.json 顶层必须是 list（每个元素是一条新闻 dict）。")

    out: List[Dict[str, Any]] = []

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

        for ev in events:
            if not isinstance(ev, dict):
                continue

            event_summary = safe_get_str(ev, "event_summary")
            if event_summary is None:
                # 没有 summary 就不输出（因为你要求必须从原文提取这对信息）
                continue

            record = {
                "NewsID": news_id,
                "event_summary": event_summary,

                # 下面这些是可选的“溯源字段”，不破坏你要的“两个内容放一起”
                #"event_ID": ev.get("event_ID"),
                #"etype": ev.get("etype"),
                #"event_time": ev.get("event_time"),
            }
            out.append(record)

    return out


def save_json(path: str, data: Any) -> None:
    """
    把 Python 对象写回 JSON 文件。

    参数：
        path: 输出路径（例如 'data/extract-data.json'）
        data: 要写入的对象（通常是 list/dict）

    这个函数做什么：
        - 自动创建父目录（data/ 不存在也能写）
        - 用 json.dump 输出为 UTF-8 JSON
        - ensure_ascii=False 保证中文不变成 \\uXXXX
        - indent=2 让文件可读
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main(
    input_path: str = "data/raw-data.json",
    output_path: str = "data/extract-data.json",
) -> None:
    """
    主流程函数：读取 → 提取 → 保存

    这个函数做什么：
        1) load_json 读入 raw-data.json
        2) extract_newsid_event_summaries 抽取 (NewsID, event_summary)
        3) save_json 写出 extract-data.json
        4) 打印简单统计，便于你立刻验证是否成功
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
