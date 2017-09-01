
# Chainer Implementation of Attention-based Aspect Extraction

## Introduction

This is a Chainer implementation of Attention-based Aspect Extraction model[^1].

[^1]: [Ruidan He, Wee Sun Lee, Hwee Tou Ng and Daniel Dahlmeier. 2017. An Unsupervised Neural Attention Model for Aspect Extraction. ACL2017](http://aclweb.org/anthology/P/P17/P17-1036.pdf)

This model is an unsupervised topic modeling which is claimed to have following properties:

* Produces more coherent topics compared to conventional topic modelling such as LDA.
* Embed topic embedding to the same subspace as the word embedding.

### Disclaimer

I am not a author of the paper.
I think I implemented it right, but no validation was done to confirm the correctness of implementation so use at your own lisk.

## How to run

### Prerequisite

```python
pip install -r requirements.py
export PYTHONPATH="`pwd`:$PYTHONPTH"
```

### Training

```
python bin/train.py --word2vec ./GoogleNews-vectors-negative300.bin
```

`python bin/train.py --help` will give you explantions of the options.

### Printing topics

`show_nearest_words.py` basically prints words that are near to the
trained topic embedding.

```python
>> python bin/show_nearest_words.py -v result/vocab.json result/trained_model

Topic #1:
  0.382 outshone
  0.347 squeezed
  0.340 counting
  0.340 hammered
  0.338 blinked
  0.337 whacked
  0.333 swapped

Topic #2:
  0.639 POSSIBILITY
  0.638 INDEED
  0.633 FIRING
  0.632 SHAKY
  0.624 APPEARS
  0.622 COMPETITOR
  0.621 STATED
```
