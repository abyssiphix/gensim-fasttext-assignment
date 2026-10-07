"""清洗 THUCNews 正文，按句子进行 jieba 分词；支持原始目录或 cnews TSV。"""
import argparse
import collections
import gzip
import json
import re
import unicodedata
from pathlib import Path
import jieba
from common import DATA, RESULTS, dump_json, sha256, stable_key, write_csv, write_jsonl

def raw_articles(source):
    if source.is_dir():
        files = sorted(p for p in source.rglob("*.txt") if not any(x.startswith(".") for x in p.relative_to(source).parts))
        # 均匀分布的固定位置，避免只抽取目录开头的一个主题。
        count = min(10000, len(files))
        for j in range(count):
            index = j * len(files) // count
            path = files[index]
            yield {"id":path.relative_to(source).as_posix(), "position":index,
                   "content":path.read_text("utf8"), "source_label":path.parent.name}
    else:
        for index, line in enumerate(source.read_text("utf8").splitlines()):
            if "\t" in line:
                label, content = line.split("\t", 1)
                yield {"id":f"cnews:{index}", "position":index,"content":content,"source_label":label}

def preprocess(raw):
    stats = collections.Counter()
    retained, sentences, manifest = [], [], []
    seen = set()
    for row in raw:
        stats["raw_articles"] += 1
        text = unicodedata.normalize("NFKC", row["content"]).strip()
        normalized = re.sub(r"\s+", "", text)
        digest = stable_key(normalized)
        reason = ""
        if not normalized:
            reason = "empty"
        elif len(normalized) < 100:
            reason = "short"
        elif digest in seen:
            reason = "duplicate"
        seen.add(digest)
        if reason:
            stats[reason] += 1
        else:
            article_sentences = []
            for fragment in re.split(r"[。！？!?；;\n\r]+", text):
                # 保留含中文、字母或数字的词，标点不作为词项。
                tokens = [w.strip() for w in jieba.lcut(fragment, HMM=True)
                          if re.search(r"[\u4e00-\u9fffA-Za-z0-9]", w)]
                if len(tokens) >= 3:
                    article_sentences.append(tokens)
            if not article_sentences:
                reason = "no_valid_sentence"
                stats[reason] += 1
            else:
                retained.append(row)
                sentences.extend(article_sentences)
                stats["retained_articles"] += 1
                stats["sentences"] += len(article_sentences)
                stats["tokens"] += sum(map(len, article_sentences))
        manifest.append({"id":row["id"],"position":row["position"],"source_label":row["source_label"],
                         "sha256":digest,"status":reason or "retained"})
    return retained, sentences, manifest, stats

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path)
    args = p.parse_args()
    cache = DATA / "thu_sample.jsonl.gz"
    if args.source:
        raw = list(raw_articles(args.source))
        write_jsonl(cache, raw)
    else:
        from common import read_jsonl
        raw = list(read_jsonl(cache))
    jieba.dt.tmp_dir = str(DATA)
    retained, sentences, manifest, stats = preprocess(raw)
    corpus = DATA / "sentences.txt.gz"
    with open(corpus,"wb") as f, gzip.GzipFile(filename="",mode="wb",fileobj=f,mtime=0) as z:
        for sent in sentences:
            z.write((" ".join(sent)+"\n").encode("utf8"))
    write_csv(DATA / "thu_manifest.csv", manifest, ["id","position","source_label","sha256","status"])
    stats = dict(stats)
    stats["raw_category_counts"] = dict(collections.Counter(a["source_label"] for a in raw))
    stats["retained_category_counts"] = dict(collections.Counter(a["source_label"] for a in retained))
    stats["corpus_sha256"] = sha256(corpus)
    stats["raw_sample_sha256"] = sha256(cache)
    dump_json(RESULTS / "thu_stats.json", stats)
    print(json.dumps(stats,ensure_ascii=False),flush=True)

if __name__ == "__main__":
    main()
