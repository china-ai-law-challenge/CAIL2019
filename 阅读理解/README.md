# CAIL2019——阅读理解

该项目为 **CAIL2019—阅读理解** 的代码和模型提交说明。

## 选手交流群

QQ群：237633234

## 数据说明

本任务所使用的数据集是来自“中国裁判文书网”公开的法律文书，主要涉及民事和刑事的一审判决书，总共约1万份数据，并按比例划分训练、开发和测试。每份数据包括若干个问题，对于训练集，每个问题只包含一个标准回答，对于开发和测试集，每个问题包含3个标准回答。回答内容可以是案情片段，可以是YES或NO，也可以拒答即回答内容为空。

数据格式参考SquAD2.0的数据格式，整体为json格式的数据。并增设案由"casename"字段和领域"domain"字段，"domain"字段只有"civil"和"criminal"两种类型。"context"抽取自裁判文书的案情描述或原告诉称部分。

## 提交的文件格式及组织形式

你可以在 ``model`` 中找到最简单的提交代码的格式。你需要将你所有的代码压缩为一个 ``zip`` 文件进行提交，该 ``zip`` 文件内部形式可以参看 ``model/submit_sample.zip``。该 ``zip`` 文件**内部顶层**必须包含``run.sh``，为运行的入口程序，我们会在该目录下使用``sh run.sh``来运行你的程序。

## 代码的内容

对于你的代码，你需要从``../data/data.json``中读取数据进行预测，该数据格式与下发数据格式完全一致。选手需要将预测的结果输出到``../result/result.json``中，预测结果文件为一个json格式的文件，具体可以查看 ``evaluate/result.json``。

你可以利用 ``model`` 下的文件进行进一步参考。**请注意**，在加载模型的时候请尽量使用相对路径，我们会将提交的压缩包解压到``/model``路径下然后运行。

## 评测脚本

我们在 ``evaluate`` 文件夹中提供了评分的代码，以供参考。

## 现有的系统环境

```
Package             Version               
------------------- ----------------------
absl-py                            0.7.1    
alabaster                          0.7.10   
anaconda-client                    1.6.9    
anaconda-navigator                 1.7.0    
anaconda-project                   0.8.2    
asn1crypto                         0.24.0   
astor                              0.7.1    
astroid                            1.6.1    
astropy                            2.0.3    
attrs                              17.4.0   
Babel                              2.5.3    
backports.shutil-get-terminal-size 1.0.0    
beautifulsoup4                     4.6.0    
bitarray                           0.8.1    
bkcharts                           0.2      
blaze                              0.11.3   
bleach                             2.1.2    
bokeh                              0.12.13  
boto                               2.48.0   
boto3                              1.9.149  
botocore                           1.12.149 
Bottleneck                         1.2.1    
certifi                            2018.1.18
cffi                               1.11.4   
chardet                            3.0.4    
click                              6.7      
cloudpickle                        0.5.2    
clyent                             1.2.2    
colorama                           0.3.9    
conda                              4.4.10   
conda-build                        3.4.1    
conda-verify                       2.0.0    
contextlib2                        0.5.5    
cryptography                       2.1.4    
cycler                             0.10.0   
Cython                             0.27.3   
cytoolz                            0.9.0    
dask                               0.16.1   
datashape                          0.5.4    
decorator                          4.2.1    
distributed                        1.20.2   
docutils                           0.14     
entrypoints                        0.2.3    
et-xmlfile                         1.0.1    
fastcache                          1.0.2    
filelock                           2.0.13   
Flask                              0.12.2   
Flask-Cors                         3.0.3    
gast                               0.2.2    
gensim                             3.7.3    
gevent                             1.2.2    
glob2                              0.6      
gmpy2                              2.0.8    
greenlet                           0.4.12   
grpcio                             1.20.1   
h5py                               2.7.1    
heapdict                           1.0.0    
html5lib                           1.0.1    
idna                               2.6      
imageio                            2.2.0    
imagesize                          0.7.1    
ipykernel                          4.8.0    
ipython                            6.2.1    
ipython-genutils                   0.2.0    
ipywidgets                         7.1.1    
isort                              4.2.15   
itsdangerous                       0.24     
jdcal                              1.3      
jedi                               0.11.1   
jieba                              0.39     
Jinja2                             2.10     
jmespath                           0.9.4    
jsonschema                         2.6.0    
jupyter                            1.0.0    
jupyter-client                     5.2.2    
jupyter-console                    5.2.0    
jupyter-core                       4.4.0    
jupyterlab                         0.31.5   
jupyterlab-launcher                0.10.2   
Keras-Applications                 1.0.7    
Keras-Preprocessing                1.0.9    
lazy-object-proxy                  1.3.1    
llvmlite                           0.21.0   
locket                             0.2.0    
lxml                               4.1.1    
Markdown                           3.1      
MarkupSafe                         1.0      
matplotlib                         2.1.2    
mccabe                             0.6.1    
mistune                            0.8.3    
mock                               3.0.5    
mpmath                             1.0.0    
msgpack-python                     0.5.1    
multipledispatch                   0.4.9    
navigator-updater                  0.1.0    
nbconvert                          5.3.1    
nbformat                           4.4.0    
networkx                           2.1      
nltk                               3.2.5    
nose                               1.3.7    
notebook                           5.4.0    
numba                              0.36.2   
numexpr                            2.6.4    
numpy                              1.14.0   
numpydoc                           0.7.0    
odo                                0.5.1    
olefile                            0.45.1   
openpyxl                           2.4.10   
packaging                          16.8     
pandas                             0.22.0   
pandocfilters                      1.4.2    
parso                              0.1.1    
partd                              0.3.8    
path.py                            10.5     
pathlib2                           2.3.0    
patsy                              0.5.0    
pep8                               1.7.1    
pexpect                            4.3.1    
pickleshare                        0.7.4    
Pillow                             5.0.0    
pip                                19.1.1   
pkginfo                            1.4.1    
pluggy                             0.6.0    
ply                                3.10     
prompt-toolkit                     1.0.15   
protobuf                           3.7.1    
psutil                             5.4.3    
ptyprocess                         0.5.2    
py                                 1.5.2    
pycodestyle                        2.3.1    
pycosat                            0.6.3    
pycparser                          2.18     
pycrypto                           2.6.1    
pycurl                             7.43.0.1 
pyflakes                           1.6.0    
Pygments                           2.2.0    
pylint                             1.8.2    
pyodbc                             4.0.22   
pyOpenSSL                          17.5.0   
pyparsing                          2.2.0    
PySocks                            1.6.7    
pytest                             3.3.2    
python-dateutil                    2.6.1    
pytz                               2017.3   
PyWavelets                         0.5.2    
PyYAML                             3.12     
pyzmq                              16.0.3   
QtAwesome                          0.4.4    
qtconsole                          4.3.1    
QtPy                               1.3.1    
requests                           2.18.4   
rope                               0.10.7   
ruamel-yaml                        0.15.35  
s3transfer                         0.2.0    
scikit-image                       0.13.1   
scikit-learn                       0.19.1   
scipy                              1.0.0    
seaborn                            0.8.1    
Send2Trash                         1.4.2    
setuptools                         38.4.0   
simplegeneric                      0.8.1    
singledispatch                     3.4.0.3  
six                                1.11.0   
sklearn                            0.0      
smart-open                         1.8.3    
snowballstemmer                    1.2.1    
sortedcollections                  0.5.3    
sortedcontainers                   1.5.9    
Sphinx                             1.6.6    
sphinxcontrib-websupport           1.0.1    
spyder                             3.2.6    
SQLAlchemy                         1.2.1    
statsmodels                        0.8.0    
sympy                              1.1.1    
tables                             3.4.2    
tblib                              1.3.2    
tensorboard                        1.13.1   
tensorflow-estimator               1.13.0   
tensorflow-gpu                     1.13.1   
termcolor                          1.1.0    
terminado                          0.8.1    
testpath                           0.3.1    
Theano                             1.0.4    
toolz                              0.9.0    
torch                              1.1.0    
tornado                            4.5.3    
tqdm                               4.31.1   
traitlets                          4.3.2    
typing                             3.6.2    
unicodecsv                         0.14.1   
urllib3                            1.22     
wcwidth                            0.1.7    
webencodings                       0.5.1    
Werkzeug                           0.14.1   
wheel                              0.30.0   
widgetsnbextension                 3.1.0    
wrapt                              1.10.11  
xgboost                            0.82     
xlrd                               1.1.0    
XlsxWriter                         1.0.2    
xlwt                               1.3.0    
zict                               0.1.3
```

等待补全中

如果你有需要的环境，请联系比赛管理员进行安装。
