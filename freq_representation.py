"""Functions for creating frequency based feature vector from text.

The function of interest is :func:`text_to_vector`, which creates
term frequency (TF) or term frequency-inverse document frequency (TF-IDF)
vectors from lists of documents. Results are output in form of a
term-document matrix.

:Author: Kjetil Valle <kjetilva@stud.ntnu.no>"""

import numpy as np
import math
from nltk.probability import FreqDist

import preprocess

def text_to_vector(docs, metric):
    """ Create frequency based feature-vector from text

    Metric must be either :attr:`FrequencyMetrics.TF` or :attr:`FrequencyMetrics.TF_IDF`.
    """
    doc_freqs = FreqDist() # Distribution over how many documents each word appear in.
    tf_dists = [] # List of TF distributions per document

    # Create freq_dist for each document
    for doc in docs:
        doc = preprocess.preprocess_text(doc)
        fd = FreqDist()
        for word in doc: fd.inc(word)
        doc_freqs.update(fd.samples())
        tf_dists.append(fd)


    all_tokens = doc_freqs.keys()
    num_docs = len(docs)
    num_features = len(all_tokens)


    # Build feature x document matrix
    matrix = np.zeros((num_features, num_docs))
    for i, fd in enumerate(tf_dists):
        if metric == FrequencyMetrics.TF:
            v = [fd.freq(word) for word in all_tokens]
        elif metric == FrequencyMetrics.TF_IDF:
            v = [fd.freq(word) * math.log(float(num_docs)/doc_freqs[word]) for word in all_tokens]
        else:
            raise ValueError("No such feature type: %s" % feature_type);
        matrix[:,i] = v

    return matrix

def text_to_dict(docs, metric):
    """ Create dictionaries of term frequencies based on documents

    Metric must be either :attr:`FrequencyMetrics.TF` or :attr:`FrequencyMetrics.TF_IDF`.
    """
    doc_freqs = FreqDist() # Distribution over how many documents each word appear in.
    tf_dists = [] # List of TF distributions per document

    # Create freq_dist for each document
    for doc in docs:
        doc = preprocess.preprocess_text(doc)
        fd = FreqDist()
        for word in doc: fd.inc(word)
        doc_freqs.update(fd.samples())
        tf_dists.append(fd)


    num_docs = len(docs)
    # Build dictionaries
    dicts = []
    for i, fd in enumerate(tf_dists):
        if i%100==0: print '    dict',str(i)+'/'+str(len(tf_dists))
        d = {}
        if metric == FrequencyMetrics.TF:
            for word in fd.samples():
                d[word] = fd.freq(word)
        elif metric == FrequencyMetrics.TF_IDF:
            for word in fd.samples():
                d[word] = fd.freq(word) * math.log(float(num_docs)/doc_freqs[word])
        else:
            raise ValueError("No such feature type: %s" % feature_type);
        dicts.append(d)
    return dicts

######
##
##  Term weighting metrics
##
######

class FrequencyMetrics:
    """Class holding constants for the different frequency metrics"""
    TF = 'tf'
    TF_IDF = 'tf-idf'

def get_metrics():
    """Get list of available :class:`FrequencyMetrics`"""
    return [FrequencyMetrics.TF,
            FrequencyMetrics.TF_IDF]


######
##
##  Test methods
##
######

def test():
    import data
    dataset = 'test/freq'
    corpus_path = '../data/'+dataset
    (documents, labels) = data.read_files(corpus_path)

    vectors = {}
    print '> Building vector representations..'
    for metric in get_metrics():
        print '   ', metric
        vectors[metric] = text_to_vector(documents, metric)

if __name__=='__main__':
    test()
