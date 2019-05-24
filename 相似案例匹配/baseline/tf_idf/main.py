import json
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def trans(x):
    x = list(jieba.cut(x))
    return " ".join(x)


se = set()
f = open("input.txt", "r", encoding="utf8")
for line in f:
    x = json.loads(line)
    se.add(x["A"])
    se.add(x["B"])
    se.add(x["C"])

f = open("/input/input.txt", "r", encoding="utf8")
for line in f:
    x = json.loads(line)
    se.add(x["A"])
    se.add(x["B"])
    se.add(x["C"])

data = list(se)
for a in range(0, len(data)):
    data[a] = trans(data[a])

tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit(data)
sparse_result = tfidf_model.transform(data)

f = open("/input/input.txt", "r", encoding="utf8")
ouf = open("/output/output.txt", "w", encoding="utf8")
for line in f:
    x = json.loads(line)
    y = [
        trans(x["A"]),
        trans(x["B"]),
        trans(x["C"])
    ]

    y = tfidf_model.transform(y)
    y = y.todense()

    v1 = np.sum(np.dot(y[0], np.transpose(y[1])))
    v2 = np.sum(np.dot(y[0], np.transpose(y[2])))
    if v1 > v2:
        print("B", file=ouf)
    else:
        print("C", file=ouf)

ouf.close()
