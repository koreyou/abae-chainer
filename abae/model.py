# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

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

    def __init__(self, word_emb, fix_embedding):
        super(SentenceEmbedding, self).__init__()
        n_features = word_emb.shape[1]
        with self.init_scope():
            self.embed = L.EmbedID(
                word_emb.shape[0], n_features, initialW=word_emb,
                ignore_label=-1)
            M_initializer = I._get_initializer(None)
            # This corresponds to tranpose(M) in original paper
            self.M = chainer.variable.Parameter(
                initializer=M_initializer, shape=(n_features, n_features))
            self.fix_embedding = fix_embedding

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

    def __init__(self, word_emb, initialT, fix_embedding=False,
                 orthogonality_penalty=1.0):
        super(ABAE, self).__init__()
        with self.init_scope():
            self.sent_emb = SentenceEmbedding(word_emb, fix_embedding)
            self.pred_topic = Topic(initialT.shape[0])
            # Topic embedding
            self.T = chainer.variable.Parameter(initializer=initialT)
            self._orthogonality_penalty = orthogonality_penalty

    def __call__(self, x, x_length, ns, ns_length):
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
