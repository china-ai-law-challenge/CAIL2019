# CAIL2019--要素识别
本项目用于2019年法研杯要素识别任务的数据及基线模型等说明

### 选手交流群

QQ群：237633234

### 公开要素标签名称
在data文件夹下每个领域对应的目录下增加了要素对应的中文名称文件（如data/divorce/selectedtags.txt），以方便选手理解。

### 开放提交测试时间
- 6月3日（周一）上午10点我们已正式开放要素识别评测任务，选手点击头像即可打开上传模型界面。注意事项有以下几点： 1、每周仅限提交3次，于每周一00:00重置提交次数；
2、大家需认真查看github（  https://github.com/china-ai-law-challenge/CAIL2019/tree/master/要素识别  ）上的说明，其中baseline中包括了基准模型及其提交zip文件，注意模型提交方式（zip格式，必须有main.py文件）、输入输出路径以及系统环境，需要添加额外环境的可以在QQ群里联系相关负责人；
3、模型大小限制在2G以内，预测时间限制在2小时。

### 数据说明
- 本任务所使用的数据集是来自“中国裁判文书网”公开的法律文书,数据中的每一行为一篇裁判文书中提取部分段落的分句结果以及句子的要素标签列表；

- 此次比赛主要涉及三个领域：婚姻、劳动争议和借款纠纷


### 提交的文件格式及组织形式

你可以在 ``python_sample`` 中找到最简单的提交代码的格式。你需要将你所有的代码压缩为一个 ``zip`` 文件进行提交，该 ``zip`` 文件内部形式可以参看 ``python_smaple/python_sample.zip``。该 ``zip`` 文件**内部顶层**必须包含``main.py``，为运行的入口程序，我们会在该目录下使用``python3 main.py``来运行你的程序；baseline/predictor/ 下的predictor.zip是基准模型训练后提交的zip样例，大家可以参考，解压到当前文件夹后直接可以看到main.py 文件，而不是一个文件夹。

### 代码的内容

对于你的代码，你需要分别从``/input/labor/input.json``, ``/input/divorce/input.json``, ``/input/loan/input.json``中读取数据进行预测，该数据格式与下发数据格式完全一致，只是每一句的labels字段为空列表。选手需要从将预测的结果分别输出到``/output/labor/output.json``, ``/output/divorce/output.json``, ``/output/loan/output.json``中，要求顺序与读入的顺序完全一致,内容和待预测数据只有唯一差别，即labels字段由空数组变成模型预测的标签数组，且预测标签必须在taglist中。

以上为 ``predictor.py`` 中你需要实现的内容，你可以利用 ``python_example`` 下的文件进行进一步参考。**请注意**，在加载模型的时候请尽量使用相对路径，我们会将提交的压缩包解压到``/work``路径下然后运行。


## 其他语言的支持

如上文所述，我们现阶段只支持 ``python`` 语言的提交，但是这并不代表你不能够使用其他语言进行预测。我们现在仍然需要实现上文所述的 ``main.py`` ，但是我们在预测的时候利用 ``os.system`` 调用系统命令运行你编译好的可执行文件，或者其他运行你代码的命令。如果你担心可执行文件没有权限，可以像给出的例子在初始化的过程中加上权限。


### baseline说明
- 将下载的训练数据放到data/文件夹下，劳动争议领域训练用的数据在目录labor/ 下，该文件夹下还有少量的测试数据以及svm模型生成的预测数据；另外两个领域的在另外的相应文件夹下；

- baseline文件夹下包括了基于svm的基线模型的训练、预测等相关代码，其中,svm.py包含模型训练的代码，predictor文件夹中包含了数据处理和预测相关代码，修改相应的数据路径后，运行svm.py可以生成模型文件，然后运行predictor.py 可以生成预测文件；基于svm的模型，通过初赛数据的训练，在线上测试集上三个领域的平均得分约为0.4468；现在已经增加了训练好的基准模型以及可以直接用于提交的zip文件；

- judger.py 中包含计算模型最终得分的代码，基于micro-f1 和 macro-f1的平均值（该任务最终的得分是三个领域得分的平均值），运行该代码可以输出得分。

# 注意
- 重申！模型预测生成的文件，格式和训练数据及待预测的数据完全一致，utf-8无bom格式编码，内容和待预测数据只有唯一差别，即labels字段由空数组变成模型预测的标签数组，且预测标签必须在taglist中，否则都会报错。


## 现有的系统环境

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

