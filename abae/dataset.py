# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

from collections import Counter, defaultdict

import nltk
import numpy as np
from chainer.datasets import TupleDataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter


def _pad_create(arr, max_len):
    s = min(max(map(len, arr)), max_len)
    ret = np.ones((len(arr), s), dtype=np.int32) * -1
    length = np.empty((len(arr), ), dtype=np.int32)
    for i, a in enumerate(arr):
        ret[i, :len(a)] = a
        length[i] = len(a)
    return ret, length


def create_dataset(data_iterator, vocab, size=-1, max_tokens=10000):
    texts = []
    labels = []
    for text, l in data_iterator:
        t = [vocab[v] for v in text if v in vocab]
        if len(t) > 0:
            texts.append(t)
            labels.append(l)
    texts, texts_len = _pad_create(texts, max_tokens)
    if size > 0:
        # Sample data AFTER all data has been loaded. This is because
        # There might be bias in data ordering.
        ind = np.random.permutation(len(texts))[:size]
        return TupleDataset(
            [texts[i] for i in ind], [texts_len[i] for i in ind],
            [labels[i] for i in ind])
    else:
        return TupleDataset(texts, texts_len, labels)

def aggregate_vocabs(data_iterator, min_tf, min_df, max_df):
    vectorizer = CountVectorizer(
        analyzer=lambda x: x, max_df=max_df, min_df=min_df)
    vectorizer.fit(map(itemgetter(0), data_iterator))

    return set(vectorizer.vocabulary_.keys())


_stop_words = set(stopwords.words('english')) | {'.', ',', '?', '!', ':', '"', "'", '&', '>'}
nltk.download(info_or_id='stopwords')
nltk.download(info_or_id='punkt')


def read_dataset(paths):
    for p in paths:
        with open(p) as fin:
            text = fin.read()
        for s in sent_tokenize(text):
            words = word_tokenize(s)
            yield [w for w in words if w not in _stop_words]


def read_20news():
    X = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    for text, t in zip(X.data, X.target):
        text = text.replace('\n', ' ')
        for s in sent_tokenize(text):
            if len(s) > 1000:
                # Just remove very long sentence because I know it only contains junk
                continue
            words = word_tokenize(s)
            yield [w for w in words if w not in _stop_words], t


