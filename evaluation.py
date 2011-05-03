"""Module containing methods for evaluating representations.

This module acts as an interface to evaluation against the :mod:`classify`
and :mod:`retrieval` modules through the :func:`evaluate_classification` and
:func:`evaluate_retrieval` functions, respectively.

:Author: Kjetil Valle <kjetilva@stud.ntnu.no>"""

from nltk.probability import FreqDist
import pprint as pp
import numpy
import scipy.spatial.distance
import random

import classify
import retrieval

def evaluate_retrieval(descriptions, solutions):
    """Return retrieval performance using cases consisting of *descriptions* and *solutions*."""
    return retrieval.evaluate_retrieval(descriptions, solutions)

def evaluate_classification(data, labels, mode='random-split'):
    """Return classification performance using set of *data* and a list of *labels*.

    Supported values for *mode* are 'split' or 'cross-validation'."""
    if mode=='split':
        return _classification_split(data['training'], labels['training'], data['test'], labels['test'])
    elif mode=='random-split':
        return _classification_random_split(data, labels)
    elif mode=='cross-validation':
        return _classification_cross_validation(data, labels)
    else:
        raise Exception('unrecognized classification evaluation mode: '+mode)

def _classification_cross_validation(features, labels, k=15):
    """Performs k-fold cross-validation.

    Does cross-validation over data with the KNN classifier.
    The data is split into k bins, the classifier trained
    on k-1 of them, and tested on the last, for all k combinations.

    *features* are a NxM feature matrix for N features and M documents.
    The list *labels* contain a label for each document.

    The data is split into *k* bins.

    Returns average accuracy of the classifier over the *k* tests.
    """
    if type(labels) is list:
        # cast to numpy array if needed
        labels = numpy.array(labels)

    # Ensure that k isn't too large for the dataset
    if k > len(labels):
        k = len(labels)

    # Randomize features indices and split into k bins
    indices = range(len(labels))
    random.shuffle(indices)
    offset = len(indices)/k
    indices = [indices[i:i+offset] for i in range(0, len(indices), offset)]

    scores = []
    # Leave-one-out testing with all bins
    for i in range(len(indices)):
        classifier = classify.KNN()

        # Train
        training_indices = [index for sublist in indices[:i]+indices[i+1:] for index in sublist]
        training_features = features[:,training_indices]
        training_labels = labels[training_indices]
        classifier.train(training_features, training_labels)

        # Test
        test_indices = indices[i]
        test_features = features[:,test_indices]
        test_labels = labels[test_indices]
        accuracy = classifier.test(test_features, test_labels)
        scores.append(accuracy)

    scores = numpy.array(scores)
    return {'mean':numpy.mean(scores), 'stdev':numpy.std(scores)}

def _classification_split(training_features, training_labels, test_features, test_labels):
    # Train
    classifier = classify.KNN()
    classifier.train(training_features, training_labels)
    accuracy = classifier.test(test_features, test_labels)
    return accuracy

def _classification_random_split(features, labels, train_size=0.6, random=False):
    """Trains and tests a classifier with a dataset.

    A KNN classifier is trained with part of the dataset, and tested
    against the remainder of the data.

    *features* is a NxM feature matrix for N features and M documents.
    The list *labels* contain a label for each document.

    Fraction *train_size* of the dataset is used to use train the classifier.

    Returns accuracy of the trained classifier over the test set.
    """
    if train_size >= 1 or train_size < 0 :
        raise ValueError, "Illegal value for 'train_size'."
    if len(features[0]) != len(labels):
        raise ValueError, "'features' and 'values' must be of equal length."

    labels = numpy.array(labels)

    if random:
        indices = numpy.random.permutation(len(labels))
        k = int(len(indices)*train_size)
        training_indices = indices[:k]
        test_indices = indices[k:]
    else:
        # this approach assumes that all categories are of equal size!
        categories = list(set(labels))
        docs_per_category = len(labels)/len(categories)
        k = int(docs_per_category*train_size)
        training_indices = []
        test_indices = []
        for cat in categories:
            indices = [i for i,label in enumerate(labels) if label==cat]
            training_indices += indices[:k]
            test_indices += indices[k:]

    training_features = features[:,training_indices]
    training_labels = labels[training_indices]
    test_features = features[:,test_indices]
    test_labels = labels[test_indices]

    return _classification_split(training_features, training_labels, test_features, test_labels)

if __name__ == "__main__":
    pass
