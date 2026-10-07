"""可选：下载固定版本原始数据，严格验证已记录的大小和 SHA-256。"""
import json
import time
import urllib.request
from common import DATA, ROOT, sha256

def main():
    sources=json.loads((DATA/"sources.json").read_text("utf8"))
    raw=ROOT/"raw";raw.mkdir(exist_ok=True)
    for key,filename in [("thu","cnews.test.txt"),("people_daily","people_daily.zip")]:
        source=sources[key];dest=raw/filename
        if dest.exists() and sha256(dest)==source["sha256"]:
            print("已存在且校验通过",filename);continue
        partial=dest.with_suffix(dest.suffix+".part")
        for attempt in range(3):
            try:
                req=urllib.request.Request(source["url"],headers={"User-Agent":"Gensim-fastText-homework"})
                with urllib.request.urlopen(req,timeout=30) as r,open(partial,"wb") as f:
                    for chunk in iter(lambda:r.read(1024*1024),b""):f.write(chunk)
                assert partial.stat().st_size==source["bytes"],"下载长度不符"
                assert sha256(partial)==source["sha256"],"SHA-256 不符，拒绝使用不同版本数据"
                partial.replace(dest)
                print("已下载并校验",filename,flush=True)
                break
            except Exception:
                if attempt==2:raise
                time.sleep(2)

if __name__=="__main__":main()
