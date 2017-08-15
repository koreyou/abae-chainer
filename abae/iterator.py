# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import chainer
from chainer.iterators import SerialIterator
import numpy as np


class NegativeSampleIterator(SerialIterator):
    def __init__(self, dataset, batch_size, ns_size, repeat=True, shuffle=True):
        self._ns_size = ns_size
        self._order_ns = None
        super(NegativeSampleIterator, self).__init__(
            dataset, batch_size, repeat=repeat, shuffle=shuffle)

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail
        N = len(self.dataset)

        i = self.current_position
        i_ns = self.current_position_ns

        batch_size = min(N - i, self.batch_size) if self._repeat else self.batch_size

        i_end = i + batch_size
        i_ns_end = i_ns + self._ns_size * batch_size

        indices = range(i, min(N, i_end))
        indices_ns = range(i_ns, min(N, i_ns_end))

        if i_end >= N:
            rest = i_end - N
            indices.extend(range(rest))
            self.current_position = rest
            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        if i_ns_end >= N:
            rest = i_ns_end - N
            indices_ns.extend(range(rest))
            self.current_position_ns = rest
        else:
            self.current_position_ns = i_ns_end

        if self._order is not None:
            indices = self._order[indices]
            indices_ns = self._order_ns[indices_ns]
            if i_end >= N:
                np.random.shuffle(self._order)
            if i_ns_end >= N:
                np.random.shuffle(self._order_ns)

        indices_ns = np.asarray(indices_ns).reshape(-1, self._ns_size)
        batch = []
        range_data = range(len(self.dataset[0]))
        for i, is_ns in zip(indices, indices_ns):
            batch.append(tuple(
                list(self.dataset[i]) +
                [[self.dataset[i_ns][j] for i_ns in is_ns] for j in range_data]
            ))
        return batch

    next = __next__

    def reset(self):
        super(NegativeSampleIterator, self).reset()
        if self._shuffle:
            self._order_ns = np.random.permutation(len(self.dataset))
        self.current_position_ns = 0


def concat_examples(batch, device=None, padding=None):
    x, x_len, ns, ns_len = chainer.dataset.convert.concat_examples(batch, device=device, padding=padding)
    xp = chainer.cuda.get_array_module(x)
    x = x[:, :xp.max(x_len)]
    ns = ns[:, :, :xp.max(ns_len)]
    return tuple((x, x_len, ns, ns_len))
