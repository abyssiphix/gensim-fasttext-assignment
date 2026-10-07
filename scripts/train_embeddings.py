"""在完全相同的分词文件上训练 Word2Vec 和 Gensim FastText 并查询。"""
import collections
import gc
import itertools
import json
import time
import numpy as np
from gensim.models import FastText, Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import LineSentence
from common import DATA, MODELS, RESULTS, dump_json, sha256

GROUPS = {
    "sports":["足球","篮球","比赛","球队"],
    "finance":["银行","股票","贷款","利率"],
    "education":["学校","学生","教师","教育"],
    "technology":["电脑","软件","网络","科技"],
}
COMMON = dict(vector_size=100,window=5,min_count=3,sg=1,negative=5,hs=0,
              sample=0.001,workers=1,seed=42,epochs=5,alpha=0.025,min_alpha=0.0001)

class Progress(CallbackAny2Vec):
    def __init__(self, name): self.name, self.epoch = name, 0
    def on_epoch_end(self, model):
        self.epoch += 1
        print(f"{self.name}: 完成第 {self.epoch} 轮",flush=True)

def cosine(a,b):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def summarize(model):
    wv = model.wv
    within, across, pairs = [], [], []
    missing = [w for group in GROUPS.values() for w in group if w not in wv.key_to_index]
    for topic, group in GROUPS.items():
        for a,b in itertools.combinations(group,2):
            if a in wv.key_to_index and b in wv.key_to_index:
                value = cosine(wv[a],wv[b]);within.append(value)
                pairs.append({"group_a":topic,"group_b":topic,"word_a":a,"word_b":b,"cosine":value})
    for (ga,wa),(gb,wb) in itertools.combinations(GROUPS.items(),2):
        for a,b in itertools.product(wa,wb):
            if a in wv.key_to_index and b in wv.key_to_index:
                value=cosine(wv[a],wv[b]);across.append(value)
                pairs.append({"group_a":ga,"group_b":gb,"word_a":a,"word_b":b,"cosine":value})
    queries = {}
    for word in ["银行","学校","电脑","经济"]:
        queries[word] = [[w,float(s)] for w,s in wv.most_similar(word,topn=5)] if word in wv.key_to_index else []
    oov = {}
    for word in ["量子计算芯片","超导量子云平台","火星智能农场"]:
        if word in wv.key_to_index:
            raise ValueError("预先指定的词已进入词表，不能作为 OOV: " + word)
        if isinstance(model,FastText):
            vector=wv[word]
            oov[word]={"in_vocab":False,"can_infer":True,"norm":float(np.linalg.norm(vector)),
                       "neighbors":[[w,float(s)] for w,s in wv.most_similar(word,topn=3)]}
        else:
            oov[word]={"in_vocab":False,"can_infer":False,"reason":"KeyError: 词表外词无可查询向量"}
    return {"vocabulary_size":len(wv),"queries":queries,"oov":oov,"groups":GROUPS,"missing_group_words":missing,
            "within_mean":float(np.mean(within)),"across_mean":float(np.mean(across)),
            "similarity_gap":float(np.mean(within)-np.mean(across)),"within_pairs":len(within),"across_pairs":len(across),"pairs":pairs}

def main():
    MODELS.mkdir(parents=True,exist_ok=True)
    corpus=DATA / "sentences.txt.gz"
    output={"common_parameters":COMMON,"fasttext_extra":{"min_n":2,"max_n":4,"bucket":200000},
            "corpus_sha256":sha256(corpus),"models":{}}
    keys = None
    for name, cls in [("Word2Vec",Word2Vec),("FastText",FastText)]:
        start=time.perf_counter()
        extra = dict(min_n=2,max_n=4,bucket=200000) if cls is FastText else {}
        model=cls(sentences=LineSentence(str(corpus)),callbacks=[Progress(name)],**COMMON,**extra)
        elapsed=time.perf_counter()-start
        if keys is None: keys=set(model.wv.key_to_index)
        else: assert keys==set(model.wv.key_to_index),"两种模型必须使用相同词表"
        model.save(str(MODELS / f"{name.lower()}.model"))
        info=summarize(model);info["training_seconds"]=elapsed
        if cls is FastText:
            # 此处导出的是已合成的词表内向量，不含子词哈希矩阵。
            vec=MODELS / "fasttext.vec"
            model.wv.save_word2vec_format(str(vec),binary=False)
            info["vec_sha256"]=sha256(vec)
        output["models"][name]=info
        dump_json(RESULTS / "embedding_results.json",output)
        print(name,"seconds",round(elapsed,2),"vocab",len(model.wv),flush=True)
        del model
        gc.collect()

if __name__ == "__main__":
    main()
