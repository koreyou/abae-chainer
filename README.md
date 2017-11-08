
# Chainer Implementation of Attention-based Aspect Extraction

## Introduction

This is a Chainer implementation of Attention-based Aspect Extraction[^1].

[^1]: [Ruidan He, Wee Sun Lee, Hwee Tou Ng and Daniel Dahlmeier. 2017. An Unsupervised Neural Attention Model for Aspect Extraction. ACL2017](http://aclweb.org/anthology/P/P17/P17-1036.pdf)

It is an unsupervised topic modeling[^2] with following properties:

[^2]: It was presented as aspect extraction, but I interpreted it as a sort of topic modeling

* Produces more coherent topics compared to conventional topic modelling such as LDA.
* Embed topic embedding to the same subspace as the word embedding.

### Disclaimer

I am not the author of the paper.
I think I implemented it right, but no validation was done to confirm the correctness of implementation so use at your own risk.

## Usage

### Prerequisite

I have confirmed that it ran with Python 2.7.12.

```python
pip install -r requirements.py
export PYTHONPATH="`pwd`:$PYTHONPTH"
```

You need to obtain or train a word embedding such as ["GoogleNews-vectors-negative300.bin"](https://code.google.com/archive/p/word2vec/).

### Training

```
python bin/train.py --word2vec ./GoogleNews-vectors-negative300.bin
```

`python bin/train.py --help` will give you explantions of the options.

### Printing topics

`show_nearest_words.py` basically prints words that are near to the trained topic embedding.

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

## Experiments

### 20 newsgroups

Topic modeling with 20 newsgroups dataset[^3].
[^2]: http://qwone.com/~jason/20Newsgroups/

It was ran with commit dcb36c33dc6a2bb9c63835 with following command.

```
python bin/train.py --word2vec ~/Downloads/GoogleNews-vectors-negative300.bin
# obtain "Neighbouring words"
python bin/show_nearest_words.py -v result/vocab.json result/trained_model -k 4
```

Then I just grepped the "Neighbouring words" inside the corpus and manually decided "20newsgroups category".
"Predicted topic" is also manually given by me. `N/A` is basically topic that I could not associate with


| Predicted topic       | Neighbouring words | 20newsgroups category |
|-----------------------|--------------------|-----------------------|
| Religion              | saints, schismatics, priestly, sacerdotal | soc.religion.christian |
| Sports                | game, defensively, ballgame, teammates | rec.sport.baseball, rec.sports.hockey |
| Living environment    | residents, habitate, abodes, tourists | misc.forsale |
| Arabic names          | Hanan, Nabeel, Hammad, Israeli | soc.religion.christian, talk.politics.mideast | 
| Cars                  | ragtop, MX5, Kadett, XR6 | rec.motorcycles |
| Tech companies        | ICON, IQT, TeleSoft, ETI | comp.windows.x |
| OS                    | ramdisk, libtiff, Indeo, logfile | comp.windows.x, comp.os.ms-windows.misc |
| Physics               | photometer, interferometer, ionospheric, scintillation | sci.space |
| Clinical              | Candidiasis, hemolytic, uremic, candidiasis | sci.med |
| Eastern Europe        | Latvia, Hungarian, Russian, Austrian, | talk.politics.mideast |
| ALL CAPITALS          | SHAKY, COSTLY, VINCE, POSSIBILITY | N/A |
| Other languages       |  rin, na, te, kan | N/A |
| General               | Its, Their, The, Certain | N/A |
| Dates                 | April, Sunday, June, February | N/A |
| Active (grammar)      | sees, lends, puts, brings | N/A
| Active (grammar)      | bring, define, identify, gather | N/A |
| Active (grammar)      | Wants, Urges, Goes, Takes | N/A |
| Pronouns (grammar)    | youd, dont, I, thay | N/A |
| Passive (grammar)     | turned, shifted, hauled, brought | N/A |
| Conjunction (grammar) | And, But, Now, Nevertheless | N/A |
| Crimes                | police, perpetrator, policeman, woman | N/A |
| Furniture             | tub, jar, pillow, pail  | N/A |
| Numbers               | 1, 2, 3, 4 | N/A |
| Personality           | gutless, whining, obnoxious, spineless | N/A |
| Commerce              | provision, payments, payment, review | N/A |
| Characteristics       | Crazy, Soul, Mighty, Kiss | N/A |
| Education             | Education, Educator, Literature, Leadership | N/A |
| Woman names           |  Sandra, Denise, Jacqueline, Pamela | N/A |
| Random names          | Wolfe, Weaver, Wilson, Porter | N/A |
| Abstract concept      | terseness, dialectic, Tractatus, indeterminacy | N/A |

We can see that it captures different concepts on each topic embedding.


