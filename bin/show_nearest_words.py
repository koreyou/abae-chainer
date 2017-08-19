# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import logging
import json

import six
import chainer
import click
from chainer import training
from chainer.training import extensions
import numpy as np

import abae

logging.basicConfig(level=logging.INFO)


@click.command()
@click.argument('filename')
@click.option('--vocab', '-v', required=True, type=click.Path(exists=True),
              help='Vocabulary json file')
@click.option('-k', default=7, type=int, help='Top k words to show')
def run(filename, vocab, k):
    """ Load trained model (from train.py) and find representative
    words from each topic.
    """
    model = abae.model.ABAE.load(filename)
    topics = model.T.data
    word_emb = model.sent_emb.embed.W.data
    with open(vocab) as fin:
        v = json.load(fin)
    rev_vocab = {v: k  for k, v in six.iteritems(v)}
    topics /= np.linalg.norm(topics, ord=2, axis=1, keepdims=True)
    word_emb /= np.linalg.norm(word_emb, ord=2, axis=1, keepdims=True)
    dist = np.dot(topics, word_emb.T)
    inds = np.argsort(dist, axis=1)[:, :-k-1:-1]
    for i in six.moves.range(inds.shape[0]):
        print('Topic #%d:' % (i + 1))
        for j in six.moves.range(inds.shape[1]):
            word = rev_vocab[inds[i, j]]
            print('  %1.3f %s' % (dist[i, inds[i, j]], word))
        print('')


if __name__ == '__main__':
    run()
