# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import json

import chainer
import chainer.links as L
import chainer.initializers as I
import chainer.functions as F
from chainer import reporter
from chainer import cuda
import numpy as np


def masked_softmax(x, length):
    xp = cuda.get_array_module(x.data)
    # mask: (B, T)
    mask = xp.tile(xp.arange(x.shape[1]).reshape(1, -1, 1), (x.shape[0], 1, 1))
    mask = mask < length.reshape(-1, 1, 1)
    padding = xp.ones(x.shape, dtype=x.dtype) * -np.inf
    z = F.where(mask, x, padding)
    return F.softmax(z)


class SentenceEmbedding(chainer.Chain):
    """ Embed series of word embedding to a sentence vector.

    Args:
        n_vocab (int): size of vocabulary. ``word_emb.shape[0]``
        word_embedding_size (int): size of word embedding.
            Typically, it is ``word_emb.shape[1]``
        fix_embedding (bool): Train word embedding.
    """
    def __init__(self, n_vocab, word_embedding_size, fix_embedding):
        super(SentenceEmbedding, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(
                n_vocab, word_embedding_size, ignore_label=-1)
            M_initializer = I._get_initializer(None)
            # This corresponds to tranpose(M) in original paper
            self.M = chainer.variable.Parameter(
                initializer=M_initializer,
                shape=(word_embedding_size, word_embedding_size))
            self.fix_embedding = fix_embedding

    def initialize(self, word_emb):
        if self.embed.W.shape != word_emb.shape:
            raise ValueError('word_emb size mismatch!')
        self.embed.W.copydata(chainer.Variable(word_emb))

    def __call__(self, x, length):
        """

        Args:
            x (numpy.ndarray or cupy.ndarray): sequences of vocabulary indices in shape
                (batchsize, tokens, feature size)
            lengths (numpy.ndarray or cupy.ndarray):
                number of tokens in each batch index

        Returns:
            chainer.Variable: Sentence embedding in shape (batchsize, feature size)

        """
        # e: (batchsize, feature size)
        e = self.embed(x)
        if self.fix_embedding:
            e.unchain_backward()
        # y: (batchsize, feature size)
        y = F.sum(e, axis=1) / length.astype(np.float32).reshape(-1, 1)
        # Equivalent to e.tran(M).y -> (batchsize, tokens, 1)
        d = F.batch_matmul(e, F.matmul(y, self.M))
        a = masked_softmax(d, length)
        # Sentence embedding z: (batchsize, feature size)
        z = F.sum(F.broadcast_to(a, e.shape) * e, axis=1)

        return z


class Topic(chainer.Chain):

    def __init__(self, n_topics):
        super(Topic, self).__init__()
        with self.init_scope():
            self.W = L.Linear(None, n_topics)

    def __call__(self, z):
        """

        Args:
            z (numpy.ndarray or cupy.ndarray): Sentence embedding in shape
                (batchsize, feature size)

        Returns:
            chainer.Variable: weight over apect embedding in shape
                (batchsize, n_topics)
        """

        # weight over apect embedding p: (batchsize, n_topics)
        p = self.W(z)

        return F.softmax(p)


class ABAE(chainer.Chain):

    def __init__(self, n_vocab, word_embedding_size, n_topics,
                 fix_embedding=False, orthogonality_penalty=1.0):
        super(ABAE, self).__init__()
        with self.init_scope():
            self.sent_emb = SentenceEmbedding(
                n_vocab, word_embedding_size, fix_embedding)
            self.pred_topic = Topic(n_topics)
            # Topic embedding
            self.T = chainer.variable.Parameter()
            self._orthogonality_penalty = orthogonality_penalty

    def initialize(self, word_emb, initial_t):
        self.T.initialize(initial_t.shape)
        # Use copydata to handle cupy/numpy in same way
        self.T.copydata(chainer.Variable(initial_t))
        self.sent_emb.initialize(word_emb)

    def __call__(self, x, x_length, ns, ns_length, label):
        """

        Args:
            x (numpy.ndarray or cupy.ndarray): sequences of vocabulary indices
                in shape (batchsize, tokens)
            x_length (numpy.ndarray or cupy.ndarray): number of tokens in each
                batch index of ``x``
            ns (numpy.ndarray or cupy.ndarray): Negative samples.
                sequences of vocabulary indices in shape (batchsize,
                n_negative_samples, tokens)
            ns_length (numpy.ndarray or cupy.ndarray): number of tokens in each
                negative sample in shape ``(batchsize, n_negative_samples)``
            label: Ignored

        Returns:
            chainer.Variable:

        """
        z = self.sent_emb(x, x_length)
        p = self.pred_topic(z)
        # reconstructed sentence embedding r: (batchsize, feature size)
        r = F.matmul(p, self.T)

        # Embed negative sampling
        bs, n_ns, _ = ns.shape
        ns = ns.reshape(bs * n_ns, -1)
        ns_length = ns_length.astype(np.float32).reshape(-1, 1)
        n = F.sum(self.sent_emb.embed(ns), axis=1) / ns_length
        if self.sent_emb.fix_embedding:
            n.unchain_backward()
        n = F.reshape(n, (bs, n_ns, -1))

        # Calculate contrasive max-margin loss
        # neg: (batchsize, n_ns)
        neg = F.sum(F.broadcast_to(F.reshape(r, (bs, 1, -1)), n.shape) * n, axis=-1)
        pos = F.sum(r * z, axis=-1)
        pos = F.broadcast_to(F.reshape(pos, (bs, 1)), neg.shape)
        mask = chainer.Variable(self.xp.zeros(neg.shape, dtype=p.dtype))
        loss_pred = F.sum(F.maximum(1. - pos + neg, mask))
        reporter.report({'loss_pred': loss_pred}, self)

        t_norm = F.normalize(self.T, axis=1)
        loss_reg = self._orthogonality_penalty * F.sqrt(F.sum(F.squared_difference(
            F.matmul(t_norm, t_norm, transb=True),
            self.xp.eye(self.T.shape[0], dtype=np.float32)
        )))
        reporter.report({'orthogonality_penalty': loss_reg}, self)
        loss = loss_pred + loss_reg
        reporter.report({'loss': loss}, self)
        return loss

    def predict_topic(self, x, x_length, ns, ns_length, label):
        """

        Args:
            x (numpy.ndarray or cupy.ndarray): sequences of vocabulary indices
                in shape (batchsize, tokens)
            x_length (numpy.ndarray or cupy.ndarray): number of tokens in each
                batch index of ``x``
            ns: Ignored
            ns_length: Ignored
            label (numpy.ndarray or cupy.ndarray): Target topic label.
                It will be directly returned for later evalution

        Returns:
            chainer.Variable:

        """
        z = self.sent_emb(x, x_length)
        return self.pred_topic(z), label

    def save(self, filename, compression=True):
        """Saves an object to the file in NPZ format.
        Args:
            filename (str): Target file name.
        """
        meta = {
            'n_vocab': self.sent_emb.embed.W.shape[0],
            'word_embedding_size': self.sent_emb.embed.W.shape[1],
            'n_topics': self.T.shape[0],
            'fix_embedding': self.sent_emb.fix_embedding,
            'orthogonality_penalty': self._orthogonality_penalty
        }
        with open(filename + '.json', 'wb') as fout:
            json.dump(meta, fout)
        chainer.serializers.save_npz(filename, self)

    @classmethod
    def load(cls, filename):
        """Loads an object from the file in NPZ format.

        Args:
            filename (str): Name of the file to be loaded.
        """
        with open(filename + '.json') as fin:
            meta = json.load(fin)
        obj = ABAE(**meta)
        chainer.serializers.load_npz(filename, obj)
        return obj
