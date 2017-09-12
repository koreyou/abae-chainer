import copy

import numpy as np
import six
from chainer import configuration
from chainer import cuda
from chainer import function
from chainer import reporter as reporter_module
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer.training import extension
from sklearn import metrics

from abae.model import ABAE


class TopicMatchEvaluator(extension.Extension):

    """Trainer extension to evaluate models on a validation set.

    This customized Evaluator first aggregates topic prediction from the Link,
    greedily match prediction to label and report f1 metric.
    If dimension of the label is larger than the prediction dimension,
    this Evaluator doe

    Args:
        iterator: Dataset iterator for the validation dataset. It can also be
            a dictionary of iterators. If this is just an iterator, the
            iterator is registered by the name ``'main'``.
        target: ``abae.ABAE`` object.
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

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None):
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

        results = {}
        preds = np.concatenate(preds)
        # Probability to one-hot argmax
        preds = np.eye(preds.shape[1], dtype=bool)[np.argmax(preds, axis=1)]
        labels = np.concatenate(labels)
        n_labels = np.max(labels)
        if n_labels > preds.shape[1]:
            return {}
        preds_indices = set(xrange(preds.shape[1]))
        # Greedily aggregate combinations
        for i in xrange(n_labels):
            best_j = None
            best_f1 = 0.
            labels_i = labels == i
            for j in preds_indices:
                # f1 is ill defined when no prediction on j is given
                if any(preds[:, j]):
                    f1 = metrics.f1_score(labels_i, preds[:, j])
                    if f1 > best_f1:
                        best_j = j
                        best_f1 = f1
            # best_j may NOT be defined if remaining preds are all zeros
            if best_j is not None:
                preds_indices.remove(best_j)
            results[prefix + 'f1_%d' % i] = best_f1
        results[prefix + 'macro_f1'] = np.average(results.values())
        return results
