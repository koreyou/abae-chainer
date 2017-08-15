# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import scipy.cluster


def initialze_topics(vectors, n_topics):
    """
    Initialize topic vectors as in the way in original paper. "We also
    initialize the aspect embedding matrix T with the centroids of clusters
    resulting from running k-means on word embeddings."

    Args:
        vectors (numpy.ndarray):
        n_topics (int):

    Returns:

    """
    vecs = scipy.cluster.vq.whiten(vectors)
    clusters, distortion = scipy.cluster.vq.kmeans(vecs, n_topics)
    return clusters
