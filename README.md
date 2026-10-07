# Gensim 词语表示学习与 fastText 文本分类

用 Gensim Word2Vec / FastText 学习中文新闻词向量，再将 FastText 词向量导入官方 fastText 分类器，完成 **2、4、8 类新闻主题分类**，并与从零训练作对照。

本仓库包含完整源码、固定语料与划分、实测指标、逐篇预测和中文实验报告。词向量模型及分类模型可通过训练生成。

[阅读实验报告](docs/report.md) · [PDF 报告](docs/report.pdf) · [Word 报告](docs/report.docx) · [逐篇预测](results/predictions.csv) · [数据与结果校验](checksums.json)

## 实测结果

| 类别数 | 预训练准确率 | 预训练宏 F1 | 从零准确率 | 从零宏 F1 |
| --- | ---: | ---: | ---: | ---: |
| 2 | 1.000 (24/24) | 1.000 | 0.750 (18/24) | 0.743 |
| 4 | 0.938 (45/48) | 0.939 | 0.562 (27/48) | 0.538 |
| 8 | 0.844 (81/96) | 0.842 | 0.344 (33/96) | 0.301 |

8 类任务中，预训练模型答对 81/96 篇，从零模型答对 33/96 篇，准确率相差 50 个百分点。相同设置重训一次预训练模型，96 篇测试文章的预测全部一致。结果只适用于本次小样本、固定文章划分和标题规则弱标签。

![2、4、8 类分类准确率与宏平均 F1](results/figures/classification_comparison.png)

<details>
<summary>查看 8 类预训练模型混淆矩阵</summary>

![8 类预训练模型混淆矩阵](results/figures/confusion_pretrained.png)

每类 12 篇留出文章，共 96 篇。纵轴为标题规则标签，横轴为模型预测。

</details>

## 实验流程

1. 清洗 THUCNews 外部新闻，分句并用 jieba 分词。
2. 用同一份分词文件训练 skip-gram Word2Vec 和 Gensim FastText。
3. 查询近邻、检查三个词表外词，计算四组词的组内和组间平均余弦相似度。
4. 从 PeopleDaily1998 恢复文章，按标题规则选样并检查明显误命中，每类固定 48 篇训练、12 篇测试。
5. 将 Gensim FastText 的合成词向量导出为 `.vec`，分别训练导入词向量与从零初始化的官方 fastText 分类器。
6. 计算准确率、宏平均 F1 和混淆矩阵，分析错误案例并审计数据划分与指标。

Gensim FastText 不提供监督文本分类接口。这里的表示学习使用 `gensim`，分类使用官方 `fasttext.train_supervised`。`.vec` 仅包含词表内的合成词向量；监督阶段设置 `minn=maxn=0`。

## 环境和运行

建议 Python 3.10-3.12。实际验证环境为 Python 3.12，具体版本见 `results/environment.json`。`fasttext` 安装需要可用的 C++ 编译环境；macOS 可使用 Xcode Command Line Tools，Linux 可使用 g++。

在解压后的目录打开终端：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py all
```

Windows 的虚拟环境激活命令为 `.venv\Scripts\activate`。`run.py` 会自动给子进程设置 `PYTHONHASHSEED=0`，模型使用单线程和种子 42。依赖完整锁定清单另见 `requirements-lock.txt`。

随包提供固定分词语料与分类输入，默认复现实验不需要重新下载数据。模型保存在 `models/`，由于体积较大不随作业包分发，可以重新训练生成。报告中的耗时是本机单次测量，不保证在其他电脑上相同；不同平台的浮点计算也可能带来小幅差异。

也可分步运行：

```bash
python run.py embeddings
python run.py classification
python run.py plots
python run.py audit
```

如需从原始文件重新制作数据：

```bash
python scripts/download_sources.py
python scripts/prepare_thu.py --source raw/cnews.test.txt
python scripts/prepare_people_daily.py --archive raw/people_daily.zip
python run.py all
```

`python run.py prepare` 只从随包保存的 THUCNews 原始样本重新分词，不会变更分类标签。原始 THUCNews 目录也可用 `prepare_thu.py --source /path/to/THUCNews` 处理；这会生成新的样本与结果，不能继续沿用本报告数值。

## 数据与截图的区别

词向量语料使用 `qingyujean/document-level-classification` 固定版本中公开的 `cnews.test.txt` 全部 10000 行，覆盖十个主题，每主题 1000 行。本实验把它用作外部无标注语料，原文件的主题标签没有进入词向量训练。文件名中的 `test` 来自该副本原项目，与本实验的人民日报测试集无关。

这不是截图中 836075 条镜像的 100 组固定位置抽样。下载副本的大小和 SHA-256 已与 Git LFS 指针核对，仍不能声称它与其他镜像完全一致。清洗规则和保留条数见 `results/thu_stats.json`，原始位置见 `data/thu_manifest.csv`。

分类语料使用截图指向的 PeopleDaily1998 仓库，压缩包实际包含 1998 年 1-6 月。通过文章编号恢复全文，首行作为标题，其余各行作为正文。候选标签来自 `data/label_rules.json`；本次检查的排除理由保存在 `data/review_exclusions.json`。这套标签仍是弱标签，不是原始数据集提供的主题金标准。

每类确定性选 60 篇，再固定分为 48 篇训练和 12 篇测试；2/4/8 类分别使用同一标签顺序的前 2/4/8 类。标签和划分在分类训练之前固定。正文最多前 500 个词项，保留人民日报已有分词，移除编号和词性。

## 仓库结构

```text
run.py                         实验入口
requirements.txt               核心依赖版本
requirements-lock.txt          本次完整依赖版本
docs/report.md                 GitHub 可直接阅读的报告
docs/report.pdf                六页报告阅读版
docs/report.docx               可编辑报告
scripts/                       数据处理、训练、绘图和审计
data/sources.json              固定来源、大小、SHA-256
data/thu_sample.jsonl.gz        THUCNews 原始抽取样本
data/sentences.txt.gz           两种词向量模型共用语料
data/people_daily_selected.jsonl.gz  已选文章、弱标签及划分
data/label_manifest.csv         480 篇文章编号、主题与划分
data/review_exclusions.json     候选检查排除理由
data/{2,4,8}class.{train,test}.txt   分类输入
results/embedding_results.json  近邻、OOV 和词组相似度
results/classification_results.csv  六组实验的实测指标
results/classification_details.json  分类别指标和混淆矩阵
results/predictions.csv         逐篇测试预测
results/case_studies.json       三个错误案例与分析
results/audit.json              划分、输入、指标审计
results/figures/                分类曲线和混淆矩阵
checksums.json                 数据与结果 SHA-256
```

## 数据来源与结果边界

- 外部词向量语料来自 [THUCNews 公开正文副本](https://github.com/qingyujean/document-level-classification/tree/5f63589fc17ab3ac360e74fc96f67d3f42e04780)，原始 10000 篇，清洗后 9815 篇、193931 个句子、4635184 个词项，最终词表 65991 词。
- 分类语料来自 [PeopleDaily1998](https://github.com/chenhui-bupt/PeopleDaily1998)，本次恢复 18647 篇文章，选取八类共 480 篇。
- 分类标签由本实验的标题规则及候选检查产生，并非原数据提供的人工主题金标准。去除首行标题不能完全消除标题选样偏差。
- 原作业截图中的采样方式、数据规模和示例结果与本次不同；本仓库所有指标均来自实际训练，具体差异见报告。

## 提交说明

可通过 GitHub 的 **Code → Download ZIP** 下载整个项目。实验方法、误判分析与结果局限详见报告。
