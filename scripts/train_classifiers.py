"""官方 fastText 监督分类：2/4/8 类，预训练与从零初始化对照。"""
import gc
import json
import time
import fasttext
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from common import DATA, MODELS, RESULTS, TOPICS, dump_json, read_jsonl, sha256, stable_key, write_csv

PARAMS = dict(dim=100,lr=0.3,epoch=25,wordNgrams=2,loss="softmax",thread=1,
              seed=42,minCount=1,minn=0,maxn=0,bucket=200000,verbose=0)

def test_articles(n):
    return sorted([a for a in read_jsonl(DATA / "people_daily_selected.jsonl.gz")
                   if a["split"]=="test" and a["topic"] in TOPICS[:n]],key=lambda a:stable_key("42:order:"+a["id"]))

def evaluate(model,articles,n):
    texts=[" ".join(a["input_tokens"]) for a in articles]
    labels,probs=model.predict(texts,k=1)
    predicted=[x[0].removeprefix("__label__") for x in labels]
    actual=[a["topic"] for a in articles]
    rows=[{"id":a["id"],"title":a["title"],"true_label":a["topic"],"predicted_label":pred,
           "confidence":float(prob[0]),"correct":a["topic"]==pred} for a,pred,prob in zip(articles,predicted,probs)]
    return {"n_test":len(actual),"correct":sum(a==b for a,b in zip(actual,predicted)),
        "accuracy":float(accuracy_score(actual,predicted)),"macro_f1":float(f1_score(actual,predicted,labels=TOPICS[:n],average="macro",zero_division=0)),
        "confusion_matrix":confusion_matrix(actual,predicted,labels=TOPICS[:n]).tolist(),
        "class_metrics":classification_report(actual,predicted,labels=TOPICS[:n],output_dict=True,zero_division=0)},rows

def main():
    summaries,details,all_predictions=[],{},[]
    vec=MODELS / "fasttext.vec"
    if not vec.exists():raise FileNotFoundError("先运行词向量训练，生成 fasttext.vec")
    pretrained_words = set()
    with open(vec,encoding="utf8") as f:
        next(f)
        for line in f:pretrained_words.add(line.split(" ",1)[0])
    for n in (2,4,8):
        articles=test_articles(n)
        for variant in ("pretrained","scratch"):
            options=dict(PARAMS)
            if variant=="pretrained":options["pretrainedVectors"]=str(vec)
            start=time.perf_counter()
            model=fasttext.train_supervised(input=str(DATA / f"{n}class.train.txt"),**options)
            elapsed=time.perf_counter()-start
            result,predictions=evaluate(model,articles,n)
            key=f"{n}class_{variant}"
            details[key]=result
            summary={"topics":n,"variant":variant,"n_train":48*n,"n_test":result["n_test"],"correct":result["correct"],
                     "accuracy":result["accuracy"],"macro_f1":result["macro_f1"],"training_seconds":elapsed}
            summaries.append(summary)
            all_predictions.extend(dict(topics=n,variant=variant,**r) for r in predictions)
            if n==8 and variant=="pretrained":
                model.save_model(str(MODELS / "8class_pretrained.bin"))
                repeat=fasttext.train_supervised(input=str(DATA / "8class.train.txt"),**options)
                _,repeat_predictions=evaluate(repeat,articles,n)
                repeated=sum(a["predicted_label"]==b["predicted_label"] for a,b in zip(predictions,repeat_predictions))
                dump_json(RESULTS / "repeat_check.json",{"identical_predictions":repeated,"total":len(articles),"same_seed":42})
                del repeat
            print(key,json.dumps(summary),flush=True)
            del model;gc.collect()
    words=[t for a in read_jsonl(DATA / "people_daily_selected.jsonl.gz") for t in a["input_tokens"]]
    coverage=sum(t in pretrained_words for t in words)/len(words)
    dump_json(RESULTS / "classification_details.json",{"parameters":PARAMS,"pretrained_vec_sha256":sha256(vec),
        "title_in_input":False,"pretrained_token_coverage":coverage,"experiments":details})
    write_csv(RESULTS / "classification_results.csv",summaries,list(summaries[0]))
    write_csv(RESULTS / "predictions.csv",all_predictions,list(all_predictions[0]))

if __name__ == "__main__":
    main()
