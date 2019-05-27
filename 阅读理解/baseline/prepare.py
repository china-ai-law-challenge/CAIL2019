#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import nltk
import os, re
import sys
import tensorflow as tf
from collections import Counter
from tqdm import tqdm
from my.utils import get_word_span, get_word_idx, process_tokens, word_tokenize, prettify_json, extract_json

data_path = "data"
nltk_path = os.path.join(data_path, "nltk")
nltk.data.path.append(os.path.abspath(nltk_path))

def prepare_data(path, start_ratio=0.0, stop_ratio=1.0):
    sent_tokenize = nltk.sent_tokenize

    source_path = os.path.join(path)
    with tf.gfile.Open(source_path) as f:
        source_data = json.load(f)

    q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
    na = []
    cy = []
    x, cx = [], []
    answerss = []
    p = []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    start_ai = int(round(len(source_data['data']) * start_ratio))
    stop_ai = int(round(len(source_data['data']) * stop_ratio))
    print("processing json file...")
    for article_index, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
        xp, cxp = [], []
        pp = []
        x.append(xp)
        cx.append(cxp)
        p.append(pp)
        for paragraph_index, paragraph in enumerate(article['paragraphs']):
            # wordss
            context = paragraph['context']
            context = "KYN" + context
            # xi is 2d list, 1st d is sentence, 2nd d is tokenized words
            xi = list(map(word_tokenize, sent_tokenize(context)))
            xi = [process_tokens(tokens) for tokens in xi]  # process tokens
            # given xi, add chars, cxi is 3d list, char - level
            cxi = [[list(xijk) for xijk in xij] for xij in xi]
            xp.append(xi)
            cxp.append(cxi)
            pp.append(context)

            for xij in xi:
                for xijk in xij:
                    word_counter[xijk] += len(paragraph['qas'])
                    lower_word_counter[xijk.lower()] += len(paragraph['qas'])
                    for xijkl in xijk:
                        char_counter[xijkl] += len(paragraph['qas'])

            rxi = [article_index, paragraph_index]
            assert len(x) - 1 == article_index
            assert len(x[article_index]) - 1 == paragraph_index
            for qa in paragraph['qas']:
                # get words
                qi = word_tokenize(qa['question'])
                qi = process_tokens(qi)
                cqi = [list(qij) for qij in qi]
                yi = []
                cyi = []
                answers = []
                for answer in qa['answers']:
                    answer_text = answer['text']
                    answers.append(answer_text)
                    answer_start = answer['answer_start']
                    if answer_start == -1:
                        if re.sub(r"\s","",answer_text) == "YES":
                            yi.append([(0, 1), (0, 2)])
                            cyi.append([0, 1])
                        elif re.sub(r"\s","",answer_text) == "NO":
                            yi.append([(0, 2), (0, 3)])
                            cyi.append([0, 1])
                        else:
                            yi.append([(0, 0), (0, 1)])
                            cyi.append([0, 1])
                    else:
                        answer_start += 3
                        answer_stop = answer_start + len(answer_text)
                        # TODO : put some function that gives word_start, word_stop here
                        yi0, yi1 = get_word_span(context, xi, answer_start, answer_stop)
                        # yi0 = answer['answer_word_start'] or [0, 0]
                        # yi1 = answer['answer_word_stop'] or [0, 1]
                        assert len(xi[yi0[0]]) > yi0[1]
                        assert len(xi[yi1[0]]) >= yi1[1]
                        w0 = xi[yi0[0]][yi0[1]]
                        w1 = xi[yi1[0]][yi1[1] - 1]
                        i0 = get_word_idx(context, xi, yi0)
                        i1 = get_word_idx(context, xi, (yi1[0], yi1[1] - 1))
                        cyi0 = answer_start - i0
                        cyi1 = answer_stop - i1 - 1
                        # print(answer_text, w0[cyi0:], w1[:cyi1+1])
                        assert answer_text[0] == w0[cyi0], (answer_text, w0, cyi0)
                        assert answer_text[-1] == w1[cyi1]
                        assert cyi0 < 32, (answer_text, w0)
                        assert cyi1 < 32, (answer_text, w1)

                        yi.append([yi0, yi1])
                        cyi.append([cyi0, cyi1])

                if len(qa['answers']) == 0:
                    yi.append([(0, 0), (0, 1)])
                    cyi.append([0, 1])
                    na.append(True)
                else:
                    na.append(False)

                for qij in qi:
                    word_counter[qij] += 1
                    lower_word_counter[qij.lower()] += 1
                    for qijk in qij:
                        char_counter[qijk] += 1

                q.append(qi)
                cq.append(cqi)
                y.append(yi)
                cy.append(cyi)
                rx.append(rxi)
                rcx.append(rxi)
                ids.append(qa['id'])
                idxs.append(len(idxs))
                answerss.append(answers)

    # add context here
    data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx, 'cy': cy,
            'idxs': idxs, 'ids': ids, 'answerss': answerss, '*p': rx, 'na': na}
    shared = {'x': x, 'cx': cx, 'p': p,
              'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              'word2vec': {}, 'lower_word2vec': {}}

    return data, shared
