# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import logging
import os
import json
import itertools

import chainer
import dill  # This is for joblib to use dill. Do NOT delete it.
from joblib import Memory
import click
from chainer import training
from chainer.training import extensions

import abae

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--epoch', '-e', type=int, default=15,
              help='Number of sweeps over the dataset to train')
@click.option('--frequency', '-f', type=int, default=-1,
              help='Frequency of taking a snapshot (in iterations)')
@click.option('--gpu', '-g', type=int, default=-1,
              help='GPU ID (negative value indicates CPU)')
@click.option('--out', '-o', default='result',
              help='Directory to output the result and temporaly file')
@click.option('--word2vec', required=True, type=click.Path(exists=True),
              help='Word2vec pretrained file path')
@click.option('--beer-train', default=None, type=click.Path(exists=True),
              help='Beer advocate train files')
@click.option('--beer-labels', default=None, type=click.Path(exists=True),
              help='Beer advocate test labels files')
@click.option('--beer-test', default=None, type=click.Path(exists=True),
              help='Beer advocate test files')
@click.option('--batchsize', '-b', type=int, default=50,
              help='Number of images in each mini-batch')
@click.option('--negative-samples', type=int, default=20,
              help='Number of images in each mini-batch')
@click.option('--ntopics', '-n', type=int, default=30,
              help='Number of topics to initialize on. Recommended n for beer data is 14.')
@click.option('--lr', type=float, default=0.001, help='Learning rate')
@click.option('--orthogonality_penalty', type=float, default=1.0,
              help='Orthogonality penalty coefficient lambda')
@click.option('--fix_embedding', type=bool, default=True,
              help='Fix word embedding during training')
@click.option('--resume', '-r', default='',
              help='Resume the training from snapshot')
def run(epoch, frequency, gpu, out, word2vec, beer_train, beer_labels, beer_test,
        batchsize, negative_samples, ntopics, lr, orthogonality_penalty,
        fix_embedding, resume):
    if (beer_labels is None) != (beer_test is None):
        raise click.BadParameter(
            "Both or neither beer-labels and beer-test can be specified")
    memory = Memory(cachedir=out, verbose=0)

    if beer_train is None:
        logger.info('Using 20newsgroup dataset')
        w2v, vocab, train, test, topic_vectors, label_dict = \
            memory.cache(abae.dataset.prepare_20news)(word2vec, ntopics)
    else:
        logger.info('Using beer adovocate dataset.')
        w2v, vocab, train, test, topic_vectors, label_dict = \
            memory.cache(abae.dataset.prepare_beer_advocate)(
                beer_train, beer_test, beer_labels, word2vec, ntopics)

    model = abae.model.ABAE(
        w2v.shape[0], w2v.shape[1], ntopics,
        fix_embedding=fix_embedding,
        orthogonality_penalty=orthogonality_penalty)
    model.initialize(w2v, topic_vectors)
    if gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=lr)
    optimizer.setup(model)

    train_iter = abae.iterator.NegativeSampleIterator(train, batchsize, negative_samples)

    # Set up a trainer
    updater = training.StandardUpdater(
        train_iter, optimizer, device=gpu, converter=abae.iterator.concat_examples_ns)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    if test is not None:
        logger.info("train: {},  test: {}".format(len(train), len(test)))
        # Evaluate the model with the test dataset for each epoch
        test_iter = abae.iterator.NegativeSampleIterator(
            test, batchsize, negative_samples, repeat=False, shuffle=False)
        trainer.extend(extensions.Evaluator(test_iter, model, device=gpu,
                                            converter=abae.iterator.concat_examples_ns),
                       trigger=(500, 'iteration'))
        trainer.extend(
            abae.evaluator.TopicMatchEvaluator(
                test_iter, model, label_dict=label_dict, device=gpu,
                converter=abae.iterator.concat_examples_ns
            ),
            trigger=(500, 'iteration'))
    else:
        logger.info("train: {}".format(len(train)))

    logger.info("With labels: %s" % json.dumps(label_dict))
    # Take a snapshot for each specified epoch
    frequency = epoch if frequency == -1 else max(1, frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.ParameterStatistics(model, trigger=(10, 'iteration')))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration')))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(resume, trainer)

    # Run the training
    trainer.run()

    # Save final model (without trainer)
    model.save(os.path.join(out, 'trained_model'))
    with open(os.path.join(out, 'vocab.json'), 'wb') as fout:
        json.dump(vocab, fout)


if __name__ == '__main__':
    run()
