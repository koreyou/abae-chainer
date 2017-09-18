# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import six
from word_embedding_loader import WordEmbedding
import abae.dataset

def create_word_emebedding(path, texts):
    vocab = abae.dataset.aggregate_vocabs(texts, 10, 0, 0.25)
    w2v = WordEmbedding.load(path)
    new_vocab = {}
    indices = []
    for k, v in six.iteritems(w2v.vocab):
        if k in vocab:
            new_vocab[k] = len(indices)
            indices.append(v)
    vecs = w2v.vectors[indices]
    return vecs, new_vocab
