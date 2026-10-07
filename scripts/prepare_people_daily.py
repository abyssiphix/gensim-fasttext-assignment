"""按文章编号恢复人民日报，标题规则生成候选，排除后固定选样与划分。"""
import argparse
import collections
import json
import re
import unicodedata
import zipfile
from pathlib import Path
from common import CHINESE, DATA, RESULTS, TOPICS, dump_json, stable_key, write_csv, write_jsonl

RULES = {
    "sports": r"体育|足球|篮球|排球|乒乓|羽毛球|网球|奥运|冬奥|世界杯|联赛|锦标赛|田径|游泳|围棋|象棋|体操",
    "finance": r"金融|银行|证券|股市|股票|保险|货币|信贷|贷款|债券|外汇|汇率|利率|财政|税收",
    "education": r"教育|学校|大学|中学|小学|高校|教师|教学|学生|校园|高考|师范|学费|幼儿园",
    "culture": r"文化|文艺|(?<!天)文学|艺术|电影|电视剧|音乐|戏剧|戏曲|京剧|书法|美术|绘画|博物馆|图书|作家|诗歌|歌舞",
    "technology": r"科技|科学|技术|科研|计算机|电脑|软件|互联网|网络|航天|卫星|发明|专利|电子|通信|电信",
    "health": r"医疗|医院|医药|医生|卫生|疾病|健康|疫苗|艾滋|癌症|患者|保健|计划生育|防病|防疫",
    "agriculture": r"农业|农民|农村|农田|农作物|水稻|粮食|棉花|种植|畜牧|渔业|农资|耕地|农机|林业",
    "law": r"法治|(?<!非)法制|法律|司法|法院|法庭|审判|检察|公安|警察|犯罪|罪犯|刑事|诉讼|判决|执法|扫黄|禁毒|反贪",
}
ID_RE = re.compile(r"^(\d{8}-\d{2}-\d{3})-(\d{3})/\S+\s+(.*)$")

def strip_pos(text):
    words = []
    for t in text.split():
        if "/" not in t:
            continue
        word = t.rsplit("/", 1)[0].lstrip("[")
        # 部分原词带有双重词性，如 近年来/l/t；去除残余后缀，保留 1/3 等字面斜杠。
        word = re.sub(r"(?:/[A-Za-z]+)+$", "", word)
        word = unicodedata.normalize("NFKC", word).strip()
        if word:
            words.append(word)
    return words

def articles_from_zip(path):
    grouped = collections.defaultdict(list)
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if not name.endswith(".txt") or name.startswith("__MACOSX/"):
                continue
            raw = z.read(name)
            try:
                text = raw.decode("utf8")
            except UnicodeDecodeError:
                text = raw.decode("gb18030")
            for line in text.splitlines():
                match = ID_RE.match(line.strip())
                if match:
                    aid, sid, content = match.groups()
                    grouped[aid].append((int(sid), strip_pos(content)))
    for aid in sorted(grouped):
        sentences = sorted(grouped[aid])
        title_tokens = sentences[0][1]
        body_tokens = [t for _, words in sentences[1:] for t in words]
        yield {"id": aid, "title": "".join(title_tokens), "body_tokens": body_tokens}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--review-file", type=Path, default=DATA / "review_exclusions.json")
    args = p.parse_args()
    exclusions = json.loads(args.review_file.read_text("utf8")) if args.review_file.exists() else {}
    candidates = {t: [] for t in TOPICS}
    seen = set()
    counts = collections.Counter()
    all_articles = list(articles_from_zip(args.archive))
    rejected = []
    for a in all_articles:
        aid, title, words = a["id"], a["title"], a["body_tokens"]
        digest = stable_key("".join(words))
        reason = None
        hits = [t for t in TOPICS if re.search(RULES[t], title)]
        if not 5 <= len(title) <= 60:
            reason = "title_length"
        elif len(words) < 80:
            reason = "short_body"
        elif digest in seen:
            reason = "duplicate_body"
        elif len(hits) != 1:
            reason = "multiple_topics" if hits else "no_topic"
        elif aid in exclusions:
            reason = "review: " + exclusions[aid]
        seen.add(digest)
        if reason:
            counts[reason] += 1
            rejected.append({"id": aid, "title": title, "reason": reason})
            continue
        a.update(topic=hits[0], body_sha256=digest)
        candidates[hits[0]].append(a)
    selected = []
    for topic in TOPICS:
        rows = sorted(candidates[topic], key=lambda x: stable_key("42:select:" + x["id"]))
        if len(rows) < 60:
            raise ValueError(f"{topic}: 只有 {len(rows)} 篇，未达到 60 篇")
        rows = rows[:60]
        rows.sort(key=lambda x: stable_key("42:split:" + x["id"]))
        for i, a in enumerate(rows):
            a.update(split="test" if i < 12 else "train", input_tokens=a["body_tokens"][:500])
            selected.append(a)
    write_jsonl(DATA / "people_daily_selected.jsonl.gz", selected)
    fields = ["id", "topic", "topic_zh", "split", "title", "body_tokens", "input_tokens", "body_sha256"]
    write_csv(DATA / "label_manifest.csv", [{"id": a["id"], "topic": a["topic"], "topic_zh": CHINESE[a["topic"]],
        "split": a["split"], "title": a["title"], "body_tokens": len(a["body_tokens"]),
        "input_tokens": len(a["input_tokens"]), "body_sha256": a["body_sha256"]} for a in selected], fields)
    write_csv(DATA / "rejected_articles.csv", rejected, ["id", "title", "reason"])
    dump_json(DATA / "label_rules.json", RULES)
    dump_json(RESULTS / "people_daily_stats.json", {"total_articles": len(all_articles), "candidate_counts": {t:len(v) for t,v in candidates.items()},
        "selected":len(selected), "per_class":{"train":48,"test":12}, "rejected_counts":dict(counts),
        "label_type":"标题规则弱标签，经本次候选内容检查；非原始人工金标准", "title_in_input":False})
    for n in (2, 4, 8):
        for split in ("train", "test"):
            # 全部实验复用同一排序与文章级划分。
            subset = [a for a in selected if a["topic"] in TOPICS[:n] and a["split"] == split]
            subset.sort(key=lambda a: stable_key("42:order:" + a["id"]))
            (DATA / f"{n}class.{split}.txt").write_text("".join(f"__label__{a['topic']} {' '.join(a['input_tokens'])}\n" for a in subset),encoding="utf8")
    print(json.dumps({t:len(v) for t,v in candidates.items()},ensure_ascii=False),flush=True)
    print("Selected",len(selected),flush=True)

if __name__ == "__main__":
    main()
