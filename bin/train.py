# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import logging
import os
import json

import chainer
import dill  # This is for joblib to use dill. Do NOT delete it.
from joblib import Memory
import click
from chainer import training
from chainer.training import extensions

import abae

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option('--epoch', '-e', type=int, default=15,
              help='Number of sweeps over the dataset to train')
@click.option('--frequency', '-f', type=int, default=-1,
              help='Frequency of taking a snapshot')
@click.option('--gpu', '-g', type=int, default=-1,
              help='GPU ID (negative value indicates CPU)')
@click.option('--out', '-o', default='result',
              help='Directory to output the result and temporaly file')
@click.option('--word2vec', required=True, type=click.Path(exists=True),
              help='Word2vec pretrained file path')
@click.option('--batchsize', '-b', type=int, default=50,
              help='Number of images in each mini-batch')
@click.option('--negative-samples', type=int, default=20,
              help='Number of images in each mini-batch')
@click.option('--ntopics', '-n', type=int, default=14)
@click.option('--train_ratio', type=float, default=0.95,
              help='Number of data to be used for validation')
@click.option('--lr', type=float, default=0.001, help='Learning rate')
@click.option('--orthogonality_penalty', type=float, default=1.0,
              help='Orthogonality penalty coefficient lambda')
@click.option('--fix_embedding', type=bool, default=True,
              help='Fix word embedding during training')
@click.option('--resume', '-r', default='',
              help='Resume the training from snapshot')
def run(epoch, frequency, gpu, out, word2vec, batchsize, negative_samples,
        ntopics, train_ratio, lr, orthogonality_penalty, fix_embedding, resume):
    memory = Memory(cachedir=out, verbose=0)

    @memory.cache
    def prepare(word2vec_path, n_topics):
        raw_data = list(abae.dataset.read_20news())

        w2v, vocab = abae.word_embedding.create_word_emebedding(word2vec_path, raw_data)

        dataset = abae.dataset.create_dataset(raw_data, vocab)
        topic_vectors = abae.topic_initializer.initialze_topics(w2v, n_topics)
        return w2v, vocab, dataset, topic_vectors

    w2v, vocab, dataset, topic_vectors = prepare(word2vec, ntopics)

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

    train, test = chainer.datasets.split_dataset_random(dataset, int(len(dataset) * train_ratio))
    logging.info("train: {},  test: {}".format(len(train), len(test)))

    train_iter = abae.iterator.NegativeSampleIterator(train, batchsize, negative_samples)
    test_iter = abae.iterator.NegativeSampleIterator(
        test, batchsize, negative_samples, repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(
        train_iter, optimizer, device=gpu, converter=abae.iterator.concat_examples)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu),
                   trigger=(200, 'iteration'))

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
