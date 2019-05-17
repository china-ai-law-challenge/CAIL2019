#!/usr/bin/env python
# coding: utf-8
import json

import jieba
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.svm import LinearSVC

dim = 5000


def cut_text(alltext):
    count = 0
    cut = jieba
    train_text = []
    for text in alltext:
        count += 1
        if count % 2000 == 0:
            print(count)
        train_text.append(' '.join(cut.cut(text)))
    return train_text


def train_tfidf(train_data):
    tfidf = TFIDF(
        min_df=5,
        max_features=dim,
        ngram_range=(1, 3),
        use_idf=1,
        smooth_idf=1
    )
    tfidf.fit(train_data)

    return tfidf


def read_trainData(path, tag_path):
    fin = open(path, 'r', encoding='utf8')
    tag_dic, tagname_dic = init(tag_path)

    alltext = []
    tag_label = []

    line = fin.readline()
    while line:
        d = json.loads(line)
        for sent in d:
            alltext.append(sent['sentence'])
            tag_label.append(getlabel(sent, tag_dic))
        line = fin.readline()
    fin.close()

    return alltext, tag_label


def train_SVC(vec, label):
    SVC = LinearSVC()
    SVC.fit(vec, label)
    return SVC


def init(tags_path):
    f = open(tags_path, 'r', encoding='utf8')
    tag_dic = {}
    tagname_dic = {}
    line = f.readline()
    while line:
        tagname_dic[len(tag_dic)] = line.strip()
        tag_dic[line.strip()] = len(tag_dic)
        line = f.readline()
    f.close()
    return tag_dic, tagname_dic


def getlabel(d, tag_dic):
    # 做单标签
    # 返回多个类的第一个
    if len(d['labels']) > 0:
        print(d['labels'])
        return tag_dic[d['labels'][0]]
    return ''


if __name__ == '__main__':
    print('train_labor_model...')
    print('reading...')
    alltext, tag_label = read_trainData('../data/labor/data_small_selected.json', '../data/labor/tags.txt')
    print('cut text...')
    train_data = cut_text(alltext)
    print('train tfidf...')
    tfidf = train_tfidf(train_data)

    vec = tfidf.transform(train_data)

    print('tag SVC')
    tag = train_SVC(vec, tag_label)

    print('saving model')
    joblib.dump(tfidf, 'predictor/model_labor/tfidf.model')
    joblib.dump(tag, 'predictor/model_labor/tag.model')

    print('train_divorce_model...')
    print('reading...')
    alltext, tag_label = read_trainData('../data/divorce/data_small_selected.json', '../data/divorce/tags.txt')
    print('cut text...')
    train_data = cut_text(alltext)
    print('train tfidf...')
    tfidf = train_tfidf(train_data)

    vec = tfidf.transform(train_data)

    print('tag SVC')
    tag = train_SVC(vec, tag_label)

    print('saving model')
    joblib.dump(tfidf, 'predictor/model_divorce/tfidf.model')
    joblib.dump(tag, 'predictor/model_divorce/tag.model')

    print('train_loan_model...')
    print('reading...')
    alltext, tag_label = read_trainData('../data/loan/data_small_selected.json', '../data/loan/tags.txt')
    print('cut text...')
    train_data = cut_text(alltext)
    print('train tfidf...')
    tfidf = train_tfidf(train_data)

    vec = tfidf.transform(train_data)

    print('tag SVC')
    tag = train_SVC(vec, tag_label)

    print('saving model')
    joblib.dump(tfidf, 'predictor/model_loan/tfidf.model')
    joblib.dump(tag, 'predictor/model_loan/tag.model')
