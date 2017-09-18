# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import itertools
import logging
import tempfile

import nltk
import numpy as np
from chainer.datasets import TupleDataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

import abae

logger = logging.getLogger(__name__)

def _pad_create(arr, max_len):
    s = min(max(map(len, arr)), max_len)
    # let memmap take care of opened file
    f = tempfile.TemporaryFile()
    ret = np.memmap(f, shape=(len(arr), s), dtype=np.int32)
    ret[:, :] = -1
    length = np.empty((len(arr), ), dtype=np.int32)
    for i, a in enumerate(arr):
        ret[i, :len(a)] = a
        length[i] = len(a)
    return ret, length


def create_dataset(texts, labels, vocab, label_dict, size=-1, max_tokens=10000):
    texts_new = []
    if labels is None:
        labels_new = None
        for text in texts:
            t = [vocab[v] for v in text if v in vocab]
            if len(t) > 0:
                texts_new.append(t)
    else:
        labels_new = []
        for text, l in zip(texts, labels):
            t = [vocab[v] for v in text if v in vocab]
            if len(t) > 0 and l in label_dict:
                texts_new.append(t)
                labels_new.append(label_dict[l])

    texts_new, texts_len = _pad_create(texts_new, max_tokens)
    if size > 0:
        # Sample data AFTER all data has been loaded. This is because
        # There might be bias in data ordering.
        ind = np.random.permutation(len(texts_new))[:size]
        if labels_new is None:
            return TupleDataset(
                [texts_new[i] for i in ind], [texts_len[i] for i in ind])
        else:
            return TupleDataset(
                [texts_new[i] for i in ind], [texts_len[i] for i in ind],
                [labels_new[i] for i in ind])
    else:
        if labels_new is None:
            return TupleDataset(texts_new, texts_len)
        else:
            return TupleDataset(texts_new, texts_len, labels_new)


def aggregate_vocabs(texts, min_tf, min_df, max_df):
    vectorizer = CountVectorizer(
        analyzer=lambda x: x, max_df=max_df, min_df=min_df)
    vectorizer.fit(texts)

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


def read_20news(subset):
    X = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'))
    texts = []
    labels = []
    for text, t in zip(X.data, X.target):
        text = text.replace('\n', ' ')
        for s in sent_tokenize(text):
            if len(s) > 1000:
                # Just remove very long sentence because I know it only contains junk
                continue
            words = word_tokenize(s)
            texts.append([w for w in words if w not in _stop_words])
            labels.append(t)
    return texts, labels


def prepare_20news(word2vec_path, n_topics):
    logger.info("Preparing data")

    logger.info("Loading data")
    train_texts, train_labels = read_20news('train')
    test_texts, test_labels = read_20news('test')
    logger.info("Loading word embedding")
    w2v, vocab = abae.word_embedding.create_word_emebedding(
        word2vec_path, itertools.chain(train_texts, test_texts))

    label_dict = {i:i for i in xrange(20)}

    logger.info("Creating dataset")
    train = create_dataset(train_texts, train_labels, vocab, label_dict)
    test = create_dataset(test_texts, test_labels, vocab, label_dict)
    logger.info("Initializing topics with k-means")
    topic_vectors = abae.topic_initializer.initialze_topics(w2v, n_topics)

    label_dict = {i: k for i, k in
                  enumerate(fetch_20newsgroups(subset='test').target_names)}
    return w2v, vocab, train, test, topic_vectors, label_dict


def prepare_beer_advocate(train_path, test_data_path, test_label_path,
                          word2vec_path, n_topics):
    assert (test_data_path is None) == (test_label_path is None)
    logger.info("Preparing data")

    label_dict = get_beer_advocate_label_dict()

    train_texts = read_beer_advocate_data(train_path)
    if test_data_path is not None:
        test_texts = read_beer_advocate_data(test_data_path)
        texts_iter = itertools.chain(train_texts, test_texts)
    else:
        texts_iter = train_texts

    logger.info("Loading word embedding")
    w2v, vocab = abae.word_embedding.create_word_emebedding(
        word2vec_path, texts_iter)

    # this produces generator so it cannot be reused
    logger.info("Creating dataset")
    train_texts = read_beer_advocate_data(train_path)
    train = create_dataset(train_texts, None, vocab, label_dict)

    if test_data_path is not None:
        test_texts = read_beer_advocate_data(test_data_path)
        test_labels = read_beer_advocate_label(test_label_path)
        test = create_dataset(test_texts, test_labels, vocab, label_dict)
    else:
        test = None
    logger.info("Initializing topics with k-means")
    topic_vectors = abae.topic_initializer.initialze_topics(w2v, n_topics)

    # Reverse label_dict
    label_dict = {v: k for k, v in label_dict.iteritems()}
    return w2v, vocab, train, test, topic_vectors, label_dict


def read_beer_advocate_data(path):
    with open(path) as fin:
        for line in fin:
            words = word_tokenize(line.strip())
            yield [w for w in words if w not in _stop_words]


def read_beer_advocate_label(path):
    labels = []
    with open(path) as fin:
        for line in fin:
            labels.append(line.strip())
    return labels


def get_beer_advocate_label_dict():
    # Do not add 'None' as it is not included in the original paper
    return {
        'feel': 0,
        'look': 1,
        'smell': 2,
        'taste': 3,
        'overall': 4
    }
