"""一键复现。默认使用随包提供的分词语料与固定分类划分，不需要联网。"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["all","prepare","embeddings","classification","plots","audit"],nargs="?",default="all")
    p.add_argument("--model-dir",type=Path)
    args=p.parse_args()
    root=Path(__file__).resolve().parent
    env=dict(os.environ,PYTHONHASHSEED="0",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MPLCONFIGDIR=str(root/".mplcache"))
    if args.model_dir:env["ASSIGNMENT_MODEL_DIR"]=str(args.model_dir.resolve())
    stages=["embeddings","classification","plots","audit"] if args.stage=="all" else [args.stage]
    scripts={"prepare":"prepare_thu.py","embeddings":"train_embeddings.py","classification":"train_classifiers.py","plots":"plot_results.py","audit":"audit_results.py"}
    for stage in stages:
        subprocess.run([sys.executable,str(root/"scripts"/scripts[stage])],env=env,cwd=root,check=True)

if __name__=="__main__":main()
