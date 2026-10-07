"""共享路径、确定性散列与文件格式。"""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = Path(os.environ.get("ASSIGNMENT_MODEL_DIR", str(ROOT / "models")))
TOPICS = ["sports", "finance", "education", "culture", "technology", "health", "agriculture", "law"]
CHINESE = dict(zip(TOPICS, ["体育", "金融", "教育", "文化", "科技", "医疗", "农业", "法治"]))

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def stable_key(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()

def dump_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf8")

def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".gz"):
        # gzip 的 mtime 固定为零，使字节级校验值不依赖执行时刻。
        with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as z:
            for row in rows:
                z.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf8"))
    else:
        with open(path, "w", encoding="utf8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
