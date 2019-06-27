# CAIL2019——阅读理解

该项目为 **CAIL2019—阅读理解** 的代码和模型提交说明。

## 选手交流群

QQ群：237633234

## 数据说明

本任务所使用的数据集是来自“中国裁判文书网”公开的法律文书，主要涉及民事和刑事的一审判决书，总共约1万份数据，并按比例划分训练、开发和测试。每份数据包括若干个问题，对于训练集，每个问题只包含一个标准回答，对于开发和测试集，每个问题包含3个标准回答。回答内容可以是案情片段，可以是YES或NO，也可以拒答即回答内容为空。

数据格式参考SquAD2.0的数据格式，整体为json格式的数据。并增设案由"casename"字段和领域"domain"字段，"domain"字段只有"civil"和"criminal"两种类型。"context"抽取自裁判文书的案情描述或原告诉称部分。

## 提交的文件格式及组织形式

你可以在 ``model`` 中找到最简单的提交代码的格式。你需要将你所有的代码压缩为一个 ``zip`` 文件进行提交，该 ``zip`` 文件内部形式可以参看 ``model/submit_sample.zip``。该 ``zip`` 文件**内部顶层**必须包含``main.py``，为运行的入口程序，我们会在该目录下使用``python main.py``或``python3 main.py``来运行你的程序。

## 代码的内容

对于你的代码，你需要从``../data/data.json``中读取数据进行预测，该数据格式与下发数据格式完全一致，但是"domain"、"casename"、"is_impossible"的值会置为""，"answers"的值会置为空列表。选手需要将预测的结果输出到``../result/result.json``中，预测结果文件为一个json格式的文件，具体可以查看 ``evaluate/result.json``。

你可以利用 ``model`` 下的文件进行进一步参考。**请注意**，在加载模型的时候请尽量使用相对路径，我们会将提交的压缩包解压到``/model``路径下然后运行。

## 评测脚本

我们在 ``evaluate`` 文件夹中提供了评分的代码，以供参考。

## 现有的系统环境

```
Package             Version               
------------------- ----------------------
absl-py (0.7.1)
asn1crypto (0.24.0)
astor (0.7.1)
bert-serving-client (1.9.1)
bert-serving-server (1.9.1)
bleach (3.1.0)
boto (2.49.0)
boto3 (1.9.146)
botocore (1.12.146)
bz2file (0.98)
certifi (2019.3.9)
chardet (3.0.4)
cryptography (2.1.4)
cupy (6.1.0)
cycler (0.10.0)
Cython (0.29.7)
decorator (4.1.2)
docutils (0.14)
fastrlock (0.4)
fasttext (0.8.3)
future (0.17.1)
gast (0.2.2)
gensim (3.7.3)
GPUtil (1.4.0)
grpcio (1.20.1)
h5py (2.9.0)
html5lib (1.0.1)
idna (2.8)
ipython (5.5.0)
ipython-genutils (0.2.0)
jieba (0.39)
jmespath (0.9.4)
joblib (0.13.2)
JPype1 (0.6.3)
Keras (2.2.4)
Keras-Applications (1.0.7)
keras-bert (0.50.0)
keras-embed-sim (0.4.0)
keras-layer-normalization (0.12.0)
keras-multi-head (0.20.0)
keras-pos-embd (0.10.0)
keras-position-wise-feed-forward (0.5.0)
Keras-Preprocessing (1.0.9)
keras-self-attention (0.41.0)
keras-transformer (0.23.0)
keyring (10.6.0)
keyrings.alt (3.0)
kiwisolver (1.1.0)
lda (1.1.0)
lightgbm (2.2.3)
Mako (1.0.10)
Markdown (3.1)
MarkupSafe (1.1.1)
matplotlib (3.0.3)
memory-profiler (0.55.0)
mock (3.0.5)
ninja (1.9.0.post1)
nltk (3.2.1)
numpy (1.16.3)
pandas (0.24.2)
pbr (3.1.1)
pexpect (4.2.1)
pickleshare (0.7.4)
Pillow (6.0.0)
pip (9.0.1)
prompt-toolkit (1.0.15)
protobuf (3.7.1)
psutil (5.6.2)
pycrypto (2.6.1)
Pygments (2.2.0)
pygobject (3.26.1)
pyhanlp (0.1.45)
pyltp (0.2.1)
pynvrtc (9.2)
pyparsing (2.4.0)
python-apt (1.6.3+ubuntu1)
python-dateutil (2.8.0)
pytorch-pretrained-bert (0.6.2)
pytz (2019.1)
pyxdg (0.25)
PyYAML (5.1)
pyzmq (18.0.1)
regex (2019.4.14)
requests (2.21.0)
s3transfer (0.2.0)
scikit-learn (0.21.0)
scikit-multilearn (0.2.0)
scipy (1.2.1)
SecretStorage (2.3.1)
setuptools (41.0.1)
simplegeneric (0.8.1)
six (1.12.0)
sklearn (0.0)
smart-open (1.8.3)
tensorboard (1.13.1)
tensorflow-estimator (1.13.0)
tensorflow-gpu (1.13.1)
tensorflow-hub (0.5.0)
termcolor (1.1.0)
tflearn (0.3.2)
Theano (1.0.4)
thulac (0.2.0)
torch (1.1.0)
torchtext (0.3.1)
torchvision (0.2.2.post3)
tqdm (4.31.1)
traitlets (4.3.2)
unattended-upgrades (0.1)
urllib3 (1.24.3)
wcwidth (0.1.7)
webencodings (0.5.1)
Werkzeug (0.15.2)
wheel (0.33.3)
xgboost (0.82)
```

等待补全中

如果你有需要的环境，请联系比赛管理员进行安装。

## 法小飞
由哈工大讯飞联合实验室出品的“法小飞”智能法律咨询助手应用了对话型阅读理解技术，在为用户提供精准答案的同时提升了用户的对话交互体验。“法小飞”是一个服务公众和律师的法律咨询助手，旨在利用自然语言处理技术和法律专业知识，为用户提供快速优质的法律咨询及相关服务。“法小飞”通过学习大量的法律知识，对当事人提出的法律问题进行自动解答，并且能够针对刑事和民事案件进行深入的案情分析，拥有类案推送、法条推荐、判决预测和律师推荐的功能。

<div align=center><img width="400" height="400" src="https://github.com/china-ai-law-challenge/CAIL2019/blob/master/%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3/%E6%AF%94%E8%B5%9B%E8%AF%B4%E6%98%8E/picture/iflylegal2.jpg"/></div>
