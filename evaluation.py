"""
Module containing methods for evaluating representations.

@author: Kjetil Valle <kjetilva@stud.ntnu.no>
"""

from nltk.probability import FreqDist
import pprint as pp
import numpy
import scipy.spatial.distance
import random

import classify
import retrieval

def evaluate_retrieval(descriptions, solutions):
    return retrieval.evaluate_retrieval(descriptions, solutions)

def evaluate_classification(data, labels, mode='split'):
    if mode=='split':
        return _classification_split(data, labels)
    elif mode=='cross-validation':
        return _classification_cross_validation(data, labels)
    else:
        raise Exception('unrecognized classification evaluation mode: '+mode)

def _classification_cross_validation(features, labels, k=15):
    """
    Performs k-fold cross-validation.

    Does cross-validation over data with the KNN classifier.
    The data is split into k bins, the classifier trained
    on k-1 of them, and tested on the last, for all k combinations.

    @type features: numpy array
    @param features:
        NxM feature matrix for N features and M documents.
    @type labels: list
    @param labels:
        Label for each document.
    @type k: number
    @param k:
        Number of bins to split the data into.

    @rtype: number
    @return:
        Average accuracy of the classifier over the k tests.
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

def _classification_split(features, labels, train_size=0.6, random=False):
    """
    Trains and tests a classifier with a dataset.

    A KNN classifier is trained with part of the dataset, and tested
    against the remainder of the data.

    @type features: numpy array
    @param features:
        NxM feature matrix for N features and M documents.
    @type labels: list
    @param labels:
        Labels for each document in the feature matrix.
    @type train_size: number
    @param train_size:
        Fraction of the dataset to use training the classifier.
    TODO: describe 'random'

    @rtype: number
    @return:
        Accuracy of the trained classifier over the test set.
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

    # Train
    classifier = classify.KNN()
    training_features = features[:,training_indices]
    training_labels = labels[training_indices]
    classifier.train(training_features, training_labels)

    # Test
    test_features = features[:,test_indices]
    test_labels = labels[test_indices]

    accuracy = classifier.test(test_features, test_labels)
    return accuracy

if __name__ == "__main__":
    pass
