"""Classification of feature-vectors using KNN classifier.

The :class:`KNN` class contains the classifier. It can :meth:`~KNN.classify` new datapoints
as soon as it is properly trained using the :meth:`~KNN.train` method.
The :meth:`~KNN.test` method provides a way to classify many vectors at once, and
return the classifiers accuracy compared to a gold standard.

:Author: Kjetil Valle <kjetilva@stud.ntnu.no>"""

import nltk
import numpy
import scipy.spatial.distance

from nltk.probability import FreqDist

class KNN:
    """K-nearest neighbors classifier.

    Classifier for labeled data in feature-vector format.
    Supports k-nearest classification against trained data samples, and
    1-nearest classification against class centroids.
    """

    def __init__(self, use_centroids=False, k=5):
        """Creates a new untrained classifier.

        If *use_centroids*, the classifier will calculate and use the best
        matching centroid of each the samples from each class when
        classifying. Otherwise the class among the k nearest is used.

        *k* is the number of neighbors to use when classifying. Ignored if
        centroids are used for classification.
        """
        self._use_centroids = use_centroids
        self._k = k

        self.features = None         # features trained against
        self.labels = None           # corresponding labels
        self.classes = None          # the set of unique labels used
        self.active_features = None  # reduced feature-vectors to use for classification
        self.centroids = None        # centroid vector for each class using all features
        self.active_centroids = None # centroid vectors using active features

    def train(self, features, labels):
        """Trains the KNN on a set of data.

        Uses NxM feature matrix *features* with M samples, each of N features.
        See output from :func:`data.read_files`.

        The list of *labels* correspond to each of the M samples.
        """
        self.labels = labels
        self.features = features
        self.classes = list(set(labels))

        # features initially active
        self.active_features = features

        if self._use_centroids:
            self.centroids = self._calculate_centroids()
            self.active_centroids = self.centroids

    def _calculate_centroids(self):
        """Calcualtes centroid vectors for each class in the list of labels.

        The centroids are calculated as the geometric mean of all document
        feature vectors belonging to that class.
        """
        fs = None
        for c in self.classes:
            indices = numpy.array([self.labels[i]==c for i in range(len(self.labels))])
            cluster = self.features[:,indices]
            centroid = numpy.average(cluster, axis=1)
            centroid = numpy.reshape(centroid,(len(centroid),1))
            if fs is None:
                fs = centroid
            else:
                fs = numpy.hstack((fs, centroid))
        return fs

    def classify(self, qs, distance_metric='cosine'):
        """Classifies a list of query cases.

        When classifying only those features that are **active** are
        used, all other features are ignored. The set of active features
        can be changed by :func:`~KNN.set_active_features`.

        Feature matrix *qs* is similar to that used in :func:`~KNN.train`, i.e a NxM
        matrix where N is number of features and M documents.

        The string *distance_metric* defines what metric to use when comparing feture vectors.
            See http://docs.scipy.org/doc/scipy/reference/spatial.distance.html#scipy.spatial.distance.cdist
            for list of supported metrics.

        Returns classification of each of the input cases.
        """
        if len(qs[:,0]) != len(self.active_features[:,0]):
            got = len(qs[:,0])
            expected = len(self.active_features[:,0])
            raise ValueError, "Wrong number of features in queries. got {0}, expected {1}".format(got, expected)

        # calculate distance matrix
        if self._use_centroids:
            dm = scipy.spatial.distance.cdist(self.active_centroids.T, qs.T, distance_metric)
            ans = self.classes
        else:
            dm = scipy.spatial.distance.cdist(self.active_features.T, qs.T, distance_metric)
            ans = self.labels

        # classify queries
        results = []
        for q in range(len(qs[0])):
            if q%100==0: print '    query',str(q)+'/'+str(len(qs[0]))
            if self._k==1:
                # No need to vote; dictator decides..
                i = numpy.argsort(dm[:,q])[0]
                results.append(ans[i])
            else:
                indices = numpy.argsort(dm[:,q])
                indices = indices[:self._k] # only the k closest

                # Marjority vote amongst the k
                votes = []
                for i in indices:
                    votes.append(ans[i])
                winner = max(set(votes), key=votes.count)
                results.append(winner)

        return results

    def set_active_features(self, list=None):
        """Changes the set of active feature.

        Takes a *list* of features to make active.
        Could either be a list of feature indices, or boolean list
        with length equal to number of features where `true` == active.
        If `None`, all features are activated.
        """
        if list is None:
            # Use all features
            self.active_features = self.features
            if self._use_centroids:
                self.active_centroids = self.centroids
        else:
            # Use specified features
            self.active_features = self.features[list,:]
            if self._use_centroids:
                self.active_centroids = self.centroids[list,:]

    def test(self, features, gold):
        """Tests this classifier against a set of labeled data.

        It is assumed that the classifier has been trained before
        this method is called.

        *features* is a NxM (features x documents) feature matrix, and
        *gold* a list of labels belonging to each of the documents in the
        feature matrix.

        Returns the accuracy of the classifier over the training data.
        """
        results = self.classify(features)
        accuracy = nltk.metrics.scores.accuracy(gold, results)
        return accuracy
