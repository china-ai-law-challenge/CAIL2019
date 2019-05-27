import argparse
import json
import nltk
import os
import random
import re
import sys
from collections import deque

import numpy as np
from tqdm import tqdm

def mytqdm(list_, desc="", show=True):
    if show:
        pbar = tqdm(list_)
        pbar.set_description(desc)
        return pbar
    return list_


def index(l, i):
    return index(l[i[0]], i[1:]) if len(i) > 1 else l[i[0]]


def fill(l, shape, dtype=None):
    out = np.zeros(shape, dtype=dtype)
    stack = deque()
    stack.appendleft(((), l))
    while len(stack) > 0:
        indices, cur = stack.pop()
        if len(indices) < shape:
            for i, sub in enumerate(cur):
                stack.appendleft([indices + (i,), sub])
        else:
            out[indices] = cur
    return out


def short_floats(o, precision):
    class ShortFloat(float):
        def __repr__(self):
            return '%.{}g'.format(precision) % self

    def _short_floats(obj):
        if isinstance(obj, float):
            return ShortFloat(obj)
        elif isinstance(obj, dict):
            return dict((k, _short_floats(v)) for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            return tuple(map(_short_floats, obj))
        return obj

    return _short_floats(o)


def argmax(x):
    return np.unravel_index(x.argmax(), x.shape)


def prettify_json(f):
    if not f.endswith(".json"):
        return
    print("prettifying : {}".format(f))
    parsed = json.load(open(f, 'r'))
    pretty_path = "{}.txt".format(f)

    with open(pretty_path, 'w') as p:
        p.write(json.dumps(parsed, indent=2))
        print("saved to : {}\n".format(pretty_path))


def word_tokenize(tokens):
    return list(tokens)
    #return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


def get_2d_spans(text, tokenss):
    spanss = []
    cur_idx = 0
    for tokens in tokenss:
        spans = []
        for token in tokens:
            if text.find(token, cur_idx) < 0:
                print(tokens)
                print("{} {} {}".format(token, cur_idx, text))
                raise Exception()
            cur_idx = text.find(token, cur_idx)
            spans.append((cur_idx, cur_idx + len(token)))
            cur_idx += len(token)
        spanss.append(spans)
    return spanss


def get_word_span(context, wordss, start, stop):
    spanss = get_2d_spans(context, wordss)
    idxs = []
    for sent_idx, spans in enumerate(spanss):
        for word_idx, span in enumerate(spans):
            if not (stop <= span[0] or start >= span[1]):
                idxs.append((sent_idx, word_idx))

    assert len(idxs) > 0, "{} {} {} {}".format(context, spanss, start, stop)
    return idxs[0], (idxs[-1][0], idxs[-1][1] + 1)


def get_phrase(context, wordss, span):
    """
    Obtain phrase as substring of context given start and stop indices in word level
    :param context:
    :param wordss:
    :param start: [sent_idx, word_idx]
    :param stop: [sent_idx, word_idx]
    :return:
    """
    start, stop = span
    flat_start = get_flat_idx(wordss, start)
    flat_stop = get_flat_idx(wordss, stop)
    if flat_start == 0: return ""
    elif flat_start == 1: return "YES"
    elif flat_start == 2: return "NO"
    words = sum(wordss, [])
    char_idx = 3
    char_start, char_stop = None, None
    for word_idx, word in enumerate(words):
        if word_idx < 3: continue
        char_idx = context.find(word, char_idx)
        assert char_idx >= 3
        if word_idx == flat_start:
            char_start = char_idx
        char_idx += len(word)
        if word_idx == flat_stop - 1:
            char_stop = char_idx
    assert char_start is not None
    assert char_stop is not None
    return context[char_start:char_stop]


def get_flat_idx(wordss, idx):
    return sum(len(words) for words in wordss[:idx[0]]) + idx[1]


def get_word_idx(context, wordss, idx):
    spanss = get_2d_spans(context, wordss)
    return spanss[idx[0]][idx[1]][0]


def process_tokens(temp_tokens):
    tokens = []
    for token in temp_tokens:
        flag = False
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens


def get_best_span(ypi, yp2i):
    max_val = 0
    best_word_span = (0, 1)
    best_sent_idx = 0
    for f, (ypif, yp2if) in enumerate(zip(ypi, yp2i)):
        argmax_j1 = 0
        for j in range(len(ypif)):
            val1 = ypif[argmax_j1]
            if val1 < ypif[j]:
                val1 = ypif[j]
                argmax_j1 = j

            val2 = yp2if[j]
            if val1 * val2 > max_val:
                best_word_span = (argmax_j1, j)
                best_sent_idx = f
                max_val = val1 * val2
    return ((best_sent_idx, best_word_span[0]), (best_sent_idx, best_word_span[1] + 1)), float(max_val)


def get_best_span_wy(wypi, th):
    chunk_spans = []
    scores = []
    chunk_start = None
    score = 0
    l = 0
    th = min(th, np.max(wypi))
    for f, wypif in enumerate(wypi):
        for j, wypifj in enumerate(wypif):
            if wypifj >= th:
                if chunk_start is None:
                    chunk_start = f, j
                score += wypifj
                l += 1
            else:
                if chunk_start is not None:
                    chunk_stop = f, j
                    chunk_spans.append((chunk_start, chunk_stop))
                    scores.append(score/l)
                    score = 0
                    l = 0
                    chunk_start = None
        if chunk_start is not None:
            chunk_stop = f, j+1
            chunk_spans.append((chunk_start, chunk_stop))
            scores.append(score/l)
            score = 0
            l = 0
            chunk_start = None

    return max(zip(chunk_spans, scores), key=lambda pair: pair[1])


def get_span_score_pairs(ypi, yp2i):
    span_score_pairs = []
    for f, (ypif, yp2if) in enumerate(zip(ypi, yp2i)):
        for j in range(len(ypif)):
            for k in range(j, len(yp2if)):
                span = ((f, j), (f, k+1))
                score = ypif[j] * yp2if[k]
                span_score_pairs.append((span, score))
    return span_score_pairs


def extract_json(input_json, size=100, seed=0):

    if not input_json.endswith(".json"):
        print("{} is not json file".format(input_json))
        return
    json_data = json.load(open(input_json, 'r'))

    article_size = len(json_data['data'])
    paragraph_size = 0
    question_size = 0
    for article_index, article in enumerate(json_data['data']):
        paragraph_size += len(article["paragraphs"])
        for paragraph_index, paragraph in enumerate(article["paragraphs"]):
            question_size += len(paragraph["qas"])
    print("total articles: {}".format(article_size))
    print("total paragraphs: {}".format(paragraph_size))
    print("total questions: {}".format(question_size))

    random.seed(seed)
    selected_indices = random.sample(range(0, question_size), size)
    print("selected_indices: {}".format(selected_indices))

    question_index = 0
    selected_data = {}
    articles = []
    for article_index, article in enumerate(json_data['data']):
        # print("article index:{}, title:{}".format(article_index, article["title"]))
        # print("    paragraphs: {}".format(len(article["paragraphs"])))
        paragraph_size += len(article["paragraphs"])
        paragraphs = []
        append_article = False
        for paragraph_index, paragraph in enumerate(article["paragraphs"]):
            # print("paragraph index:{}".format(paragraph_index))
            # print("       questions: {}".format(len(paragraph["qas"])))
            question_size += len(paragraph["qas"])
            qas = []
            append_paragraph = False
            for qa_index, qa in enumerate(paragraph["qas"]):
                if question_index in selected_indices:
                    # this is the question we want
                    append_article = True
                    append_paragraph = True
                    qas.append(qa)
                question_index += 1
            if append_paragraph:
                paragraph["qas"] = qas
                paragraphs.append(paragraph)
        if append_article:
            article["paragraphs"] = paragraphs
            articles.append(article)
    selected_data["data"] = articles
    selected_data["version"] = "1.1"
    output_dir = os.path.dirname(input_json)
    output_name = "{}-{}.json".format(os.path.splitext(os.path.basename(input_json))[0], size)
    output_path = os.path.join(output_dir, output_name)
    json.dump(selected_data, open(output_path, 'w'))
    prettify_json(output_path)
    return output_path


def print_data_stats(input_json):
    if not input_json.endswith(".json"):
        print("{} is not json file".format(input_json))
        return
    json_data = json.load(open(input_json, 'r'))

    article_size = len(json_data['data'])
    paragraph_size = 0
    question_size = 0
    context_string_lengths = []
    question_string_lengths = []
    for article_index, article in enumerate(json_data['data']):
        paragraph_size += len(article["paragraphs"])
        for paragraph_index, paragraph in enumerate(article["paragraphs"]):
            context_string_lengths.append(len(paragraph["context"]))
            question_size += len(paragraph["qas"])
            for qa in paragraph["qas"]:
                question_string_lengths.append(len(qa["question"]))

    print("total articles: {}".format(article_size))
    print("total paragraphs: {}".format(paragraph_size))
    print("total questions: {}".format(question_size))
    print("context_string_lengths (sorted:{}".format(sorted(context_string_lengths)))
    print("question_string_lengths (sorted): {}".format(sorted(question_string_lengths)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--print_data_stats")
    parser.add_argument('-e', "--extract_json", nargs='*')
    parser.add_argument('-p', "--prettify_json", nargs='*')

    args = parser.parse_args()
    if args.print_data_stats:
        print_data_stats(args.print_data_stats)
    elif args.extract_json:
        extract_json(args.extract_json[0], int(args.extract_json[1]))
    elif args.prettify_json:
        for j in args.prettify_json:
            prettify_json(j)
