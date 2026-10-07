"""由本次结果绘制分类曲线和八类混淆矩阵。"""
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import RESULTS, TOPICS

def main():
    figures=RESULTS/"figures";figures.mkdir(exist_ok=True)
    rows=list(csv.DictReader(open(RESULTS/"classification_results.csv",encoding="utf-8-sig")))
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),layout="constrained")
    for variant,label,color,marker in [("pretrained","Pretrained FastText","#1769aa","o"),("scratch","From scratch","#d87517","s")]:
        selected=[r for r in rows if r["variant"]==variant]
        for ax,metric,title in zip(axes,["accuracy","macro_f1"],["Accuracy","Macro F1"]):
            x=[int(r["topics"]) for r in selected];y=[float(r[metric]) for r in selected]
            ax.plot(x,y,marker=marker,color=color,label=label,linewidth=2)
            ax.set(xticks=[2,4,8],ylim=(0,1.06),xlabel="Number of topics",ylabel=title)
            ax.grid(alpha=.2)
            for a,b in zip(x,y):ax.annotate(f"{b:.3f}",(a,b),xytext=(0,9 if variant=="pretrained" else -18),textcoords="offset points",ha="center",fontsize=9)
    axes[0].plot([2,4,8],[.5,.25,.125],"--",color="gray",label="Uniform chance")
    axes[0].legend(loc="lower left",fontsize=9)
    fig.savefig(figures/"classification_comparison.png",dpi=220)
    plt.close(fig)
    details=json.loads((RESULTS/"classification_details.json").read_text("utf8"))["experiments"]
    for variant in ["pretrained","scratch"]:
        matrix=np.array(details[f"8class_{variant}"]["confusion_matrix"])
        fig,ax=plt.subplots(figsize=(7.3,6.3),layout="constrained")
        im=ax.imshow(matrix,cmap="Blues",vmin=0,vmax=12)
        ax.set(xticks=range(8),yticks=range(8),xticklabels=TOPICS,yticklabels=TOPICS,
               xlabel="Predicted topic",ylabel="Title-rule topic",title=f"8 topics: {variant} (12 test articles per topic)")
        plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
        for i in range(8):
            for j in range(8):ax.text(j,i,str(matrix[i,j]),ha="center",va="center",color="white" if matrix[i,j]>=7 else "black")
        fig.colorbar(im,ax=ax,shrink=.83,label="Articles")
        fig.savefig(figures/f"confusion_{variant}.png",dpi=220)
        plt.close(fig)

if __name__=="__main__":main()
