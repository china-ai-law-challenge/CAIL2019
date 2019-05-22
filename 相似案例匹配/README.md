# CAIL2019——相似案例匹配

该项目为 **CAIL2019——相似案例匹配** 的代码和模型提交说明。

## 选手交流群

QQ群：237633234

## 数据说明

本任务所使用的数据集是来自“中国裁判文书网”公开的法律文书,其中每份数据由三篇法律文书组成。

对于每篇法律文书，我们提供该文书的事实描述部分。在下发的数据中，文件的每一行对应一组数据，且每行的格式都为一个json数据。对于每份数据，我们用$(A,B,C)$来代表该组数据，其中$(A,B,C)$均对应某一篇文书。对于下发的训练数据，我们保证，我们的文书数据$A$与$B$的相似度是大于$A$与$B$的相似度的，即$sim(A,B)>sim(A,C)$。

我们的数据总共涉及三万组文书三元对，所有的文书三元组对都一定属于民间借贷、知识产权纠纷和海商海事案件中的某一种。

## 提交的文件格式及组织形式

你可以在 ``python_sample`` 中找到最简单的提交代码的格式。你需要将你所有的代码压缩为一个 ``zip`` 文件进行提交，该 ``zip`` 文件内部形式可以参看 ``python_smaple/python_sample.zip``。该 ``zip`` 文件**内部顶层**必须包含``main.py``，为运行的入口程序，我们会在该目录下使用``python3 main.py``来运行你的程序。

## 代码的内容

对于你的代码，你需要从``/input/input.txt``中读取数据进行预测，该数据格式与下发数据格式完全一致，但不保证$sim(A,B)>sim(A,C)$。选手需要从将预测的结果输出到``/output/output.txt``中，一行一个预测结果且要求顺序与读入的顺序完全一致。如果有$sim(A,B)>sim(A,C)$则选手需要输出``B``，否则输出``C``，一行一个结果且顺序与输入文件保持一致。

以上为 ``predictor.py`` 中你需要实现的内容，你可以利用 ``python_example`` 下的文件进行进一步参考。**请注意**，在加载模型的时候请尽量使用相对路径，我们会将提交的压缩包解压到``/work``路径下然后运行。

## 其他语言的支持

如上文所述，我们现阶段只支持 ``python`` 语言的提交，但是这并不代表你不能够使用其他语言进行预测。我们在 ``c++_sample`` 下提供了一种可能的 ``c++`` 的实现方法。我们现在仍然需要实现上文所述的 ``main.py`` ，但是我们在预测的时候利用 ``os.system`` 调用系统命令运行你编译好的可执行文件，或者其他运行你代码的命令。如果你担心可执行文件没有权限，可以像给出的例子在初始化的过程中加上权限。

## 评测脚本

我们在 ``judger`` 文件夹中提供了评分的代码，以供参考。

## 现有的系统环境

| 软件名称 | 版本号 |
| -------- | ------ |
| python   | 3.6.7  |
| g++      | 7.4.0  |
| gcc      | 7.4.0  |
| java     | 暂无   |
|          |        |

python库的环境列表：

```
Package             Version               
------------------- ----------------------
absl-py               0.7.1
asn1crypto            0.24.0
astor                 0.7.1
bleach                3.1.0
boto                  2.49.0
boto3                 1.9.146
botocore              1.12.146
bz2file               0.98
certifi               2019.3.9
chardet               3.0.4
cryptography          2.1.4
cycler                0.10.0
Cython                0.29.7
decorator             4.1.2
docutils              0.14
fasttext              0.8.3
future                0.17.1
gast                  0.2.2
gensim                3.7.3
grpcio                1.20.1
h5py                  2.9.0
html5lib              1.0.1
idna                  2.8
ipython               5.5.0
ipython-genutils      0.2.0
jieba                 0.39
jmespath              0.9.4
joblib                0.13.2
JPype1                0.6.3
keyring               10.6.0
keyrings.alt          3.0
kiwisolver            1.1.0
lightgbm              2.2.3
Mako                  1.0.10
Markdown              3.1
MarkupSafe            1.1.1
matplotlib            3.0.3
numpy                 1.16.3
pandas                0.24.2
pexpect               4.2.1
pickleshare           0.7.4
Pillow                6.0.0
pip                   9.0.1
prompt-toolkit        1.0.15
protobuf              3.7.1
pycrypto              2.6.1
Pygments              2.2.0
pygobject             3.26.1
pyhanlp               0.1.45
pyparsing             2.4.0
python-apt            1.6.3+ubuntu1
python-dateutil       2.8.0
pytz                  2019.1
pyxdg                 0.25
PyYAML                5.1
requests              2.21.0
s3transfer            0.2.0
scikit-learn          0.21.0
scikit-multilearn     0.2.0
scipy                 1.2.1
SecretStorage         2.3.1
setuptools            41.0.1
simplegeneric         0.8.1
six                   1.12.0
sklearn               0.0
smart-open            1.8.3
termcolor             1.1.0
tflearn               0.3.2
Theano                1.0.4
thulac                0.2.0
torch                 1.1.0
torchvision           0.2.2.post3
tqdm                  4.31.1
traitlets             4.3.2
unattended-upgrades   0.1
urllib3               1.25.2
wcwidth               0.1.7
webencodings          0.5.1
Werkzeug              0.15.2
wheel                 0.33.3
xgboost               0.82
keras-applications    1.0.7 
keras-preprocessing   1.0.9 
mock                  3.0.5 
tensorboard           1.13.1 
tensorflow-estimator  1.13.0 
tensorflow-gpu        1.13.1
```

等待补全中

如果你有需要的环境，请联系比赛管理员进行安装。
