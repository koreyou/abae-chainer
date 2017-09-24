import copy

import numpy as np
from chainer import configuration
from chainer import cuda
from chainer import function
from chainer import reporter as reporter_module
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer.training import extension

from abae.model import ABAE


class TopicMatchEvaluator(extension.Extension):

    """Trainer extension to evaluate models on a validation set.

    This customized Evaluator first aggregates topic prediction from the Link,
    match each prediction to a label and report (macro) precision. Every
    prediction is matched to one label. Each label will have either 0 or
    multiple prediction matched. Precision for each label is calculated
    by talking union of the prediction, i.e. true positive is defined as a
    prediction whose label is true and any of the paired predictions is true.

    This evaluator use oracle to try find best macro precision. It greedily
    match a prediction to a label using precision.

    If dimension of the label is larger than the prediction dimension,
    this Evaluator does nothing.

    Args:
        iterator: Dataset iterator for the validation dataset. It can also be
            a dictionary of iterators. If this is just an iterator, the
            iterator is registered by the name ``'main'``.
        target: ``abae.ABAE`` object.
        label_dict (dict or None):  Mapping from prediction index (int) to the
            label (str)
        converter: Converter function to build input arrays.
            :func:`~chainer.dataset.concat_examples` is used by default.
        device: Device to which the training data is sent. Negative value
            indicates the host memory (CPU).

    Attributes:
        converter: Converter function.
        device: Device to which the training data is sent.
    """
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    def __init__(self, iterator, target, label_dict=None,
                 converter=convert.concat_examples, device=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if not isinstance(target, ABAE):
            raise TypeError(
                "Input to TopicMatchEvaluator must be abae.ABAE (%s given)" %
                type(target)
            )
        self._targets = {"main": target}

        self.converter = converter
        self.device = device
        self.label_dict = label_dict

    def get_iterator(self, name):
        """Returns the iterator of the given name."""
        return self._iterators[name]

    def get_all_iterators(self):
        """Returns a dictionary of all iterators."""
        return dict(self._iterators)

    def get_target(self, name):
        """Returns the target link of the given name."""
        return self._targets[name]

    def get_all_targets(self):
        """Returns a dictionary of all target links."""
        return dict(self._targets)

    def __call__(self, trainer=None):
        """Executes the evaluator extension.
        Unlike usual extensions, this extension can be executed without passing
        a trainer object. This extension reports the performance on validation
        dataset using the :func:`~chainer.report` function. Thus, users can use
        this extension independently from any trainer by manually configuring
        a :class:`~chainer.Reporter` object.
        Args:
            trainer (~chainer.training.Trainer): Trainer object that invokes
                this extension. It can be omitted in case of calling this
                extension manually.
        Returns:
            dict: Result dictionary that contains mean statistics of values
                reported by the evaluation function.
        """
        with configuration.using_config('train', False):
            result = self.evaluate()

        reporter_module.report(result)
        return result

    def evaluate(self):
        if hasattr(self, 'name'):
            prefix = self.name + '/'
        else:
            prefix = ''
        prefix += 'topic_match/'

        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        preds = []
        labels = []
        for batch in it:
            in_arrays = self.converter(batch, self.device)
            with function.no_backprop_mode():
                if isinstance(in_arrays, tuple):
                    pred, label = target.predict_topic(*in_arrays)
                elif isinstance(in_arrays, dict):
                    pred, label = target.predict_topic(**in_arrays)
                else:
                    pred, label = target.predict_topic(in_arrays)
            preds.append(cuda.to_cpu(pred.data))
            labels.append(cuda.to_cpu(label))

        preds = np.concatenate(preds)
        # Probability to one-hot argmax
        preds = np.eye(preds.shape[1], dtype=bool)[np.argmax(preds, axis=1)]
        labels = np.concatenate(labels)
        n_labels = np.max(labels)
        if n_labels > preds.shape[1]:
            return {}

        tps_agg = {}
        ps_agg = {}
        for j in xrange(n_labels):
            tps_agg[j] = 0.0
            ps_agg[j] = 0.0

        for i in xrange(preds.shape[1]):
            # Make sure that precision is always well defined
            ps = float(np.sum(preds[:, i]))
            if ps == 0.:
                continue
            best_j = None
            best_tps = 0
            for j in xrange(n_labels):
                tps = float(np.sum(np.logical_and(labels == j, preds[:, i])))
                if tps > best_tps:
                    best_tps = tps
                    best_j = j
            if best_j is None:
                continue
            tps_agg[best_j] += best_tps
            ps_agg[best_j] += ps

        results = {}
        for j in xrange(n_labels):
            if self.label_dict is None:
                name = 'precision_%d' % j
            else:
                name = 'precision_%s' % self.label_dict[j]
            prec = tps_agg[j] / ps_agg[j] if ps_agg[j] > 0. else 0.
            results[prefix + name] = prec

        results[prefix + 'macro_precision'] = np.average(results.values())
        return dict(results)
