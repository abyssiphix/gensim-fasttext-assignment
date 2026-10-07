"""审计文章划分、正文输入、指标与文件校验值；任一不满足则退出失败。"""
import collections
import csv
import json
import re
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from common import DATA, RESULTS, ROOT, TOPICS, dump_json, read_jsonl, sha256, stable_key

def main():
    articles=list(read_jsonl(DATA/"people_daily_selected.jsonl.gz"))
    assert len(articles)==480 and len({a["id"] for a in articles})==480
    assert len({a["body_sha256"] for a in articles})==480
    counts=collections.Counter((a["topic"],a["split"]) for a in articles)
    assert all(counts[t,"train"]==48 and counts[t,"test"]==12 for t in TOPICS)
    for a in articles:
        assert a["input_tokens"]==a["body_tokens"][:500]
        assert all(not re.search(r"/[A-Za-z]+$",t) for t in a["body_tokens"])
    splits={}
    for n in (2,4,8):
        splits[n]={}
        for split in ["train","test"]:
            subset=sorted([a for a in articles if a["topic"] in TOPICS[:n] and a["split"]==split],key=lambda a:stable_key("42:order:"+a["id"]))
            lines=(DATA/f"{n}class.{split}.txt").read_text("utf8").splitlines()
            expected=[f"__label__{a['topic']} {' '.join(a['input_tokens'])}" for a in subset]
            assert lines==expected
            splits[n][split]={a["id"] for a in subset}
        assert not splits[n]["train"] & splits[n]["test"]
    for small,big in [(2,4),(4,8)]:
        for split in ["train","test"]:assert splits[small][split] <= splits[big][split]
    preds=list(csv.DictReader(open(RESULTS/"predictions.csv",encoding="utf-8-sig")))
    summaries=list(csv.DictReader(open(RESULTS/"classification_results.csv",encoding="utf-8-sig")))
    details=json.loads((RESULTS/"classification_details.json").read_text("utf8"))["experiments"]
    for summary in summaries:
        n=int(summary["topics"]);variant=summary["variant"]
        rows=[r for r in preds if int(r["topics"])==n and r["variant"]==variant]
        assert len(rows)==n*12 and {r["id"] for r in rows}==splits[n]["test"]
        labels={a["id"]:a["topic"] for a in articles}
        assert all(r["true_label"]==labels[r["id"]] and r["predicted_label"] in TOPICS[:n] for r in rows)
        actual=[r["true_label"] for r in rows];predicted=[r["predicted_label"] for r in rows]
        assert abs(accuracy_score(actual,predicted)-float(summary["accuracy"]))<1e-12
        assert abs(f1_score(actual,predicted,labels=TOPICS[:n],average="macro",zero_division=0)-float(summary["macro_f1"]))<1e-12
        cm=np.array(details[f"{n}class_{variant}"]["confusion_matrix"])
        assert cm.sum()==n*12 and (cm.sum(axis=1)==12).all()
        assert int(cm.trace())==int(summary["correct"])
    repeat=json.loads((RESULTS/"repeat_check.json").read_text("utf8"))
    assert repeat["identical_predictions"]==repeat["total"]==96
    embeddings=json.loads((RESULTS/"embedding_results.json").read_text("utf8"))
    assert embeddings["corpus_sha256"]==sha256(DATA/"sentences.txt.gz")
    assert embeddings["models"]["Word2Vec"]["vocabulary_size"]==embeddings["models"]["FastText"]["vocabulary_size"]
    for model in embeddings["models"].values():
        assert not model["missing_group_words"]
        assert model["within_pairs"]==24 and model["across_pairs"]==96
    dump_json(RESULTS/"audit.json",{"status":"passed","articles":480,"article_id_leakage":False,"duplicate_bodies":False,
        "nested_splits":True,"input_verified_from_body_tokens":True,"metric_recalculation":True,"repeat_predictions":96})
    checksums={str(p.relative_to(ROOT)):sha256(p) for folder in [DATA,RESULTS] for p in sorted(folder.rglob("*"))
               if p.is_file() and p.name not in ["checksums.json"] and not p.name.endswith(".cache")}
    dump_json(ROOT/"checksums.json",checksums)
    print("Audit passed: 划分、输入、指标、重复预测和文件校验值已验证。",flush=True)

if __name__=="__main__":main()
