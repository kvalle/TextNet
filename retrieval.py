"""Evaluation method based on case retrieval.

Evaluate lists of cases with :func:`evaluate_retrieval`. For each problem
description the remaining descriptions are assessed, and the solution
corresponding to the best matching description is retrieved.
Actual solution is compared to retrieved solution using cosine of solution
vectors.

The overall evaluation score is equal to the average solution-solution
similarity over the case base.


..  For every case:
                                                   __________
     __________                                  _|________  |
    |          |    retrieve similar case       |          | |
    | problem  | ------------------------->     | problem  | |
    |          |                                |          | |
    |----------|     assess similarity of       |----------| |
    |          |     retrieved solution         |          | |
    | solution |  <---------------------------  | solution |_|
    |__________|                                |__________|

:Author: Kjetil Valle <kjetilva@stud.ntnu.no>"""

import numpy
import scipy.spatial.distance

def evaluate_retrieval(descriptions, solutions):
    """Perform retrieval operations for each input case, returning overall performance.

    Each description in *descriptions* is used to retrieve the solution
    (from *solutions*) corresponding to the most similar other description.
    The average similarity between query and retrieved solution is returned.
    """
    description_sim = 1.0 - scipy.spatial.distance.cdist(descriptions.T, descriptions.T, 'cosine')
    solution_sim = 1.0 - scipy.spatial.distance.cdist(solutions.T, solutions.T, 'cosine')
    # setting self-similarity to 0 to avoid retrieval of own solution:
    description_sim -= numpy.diag([1.0]*len(description_sim),0)
    matches = numpy.argmax(description_sim, axis=1)
    scores = numpy.array([solution_sim[i,j] for i,j in enumerate(matches)])
    return scores.mean()

### Tests

def test_evaluate_retrieval():
    desc = numpy.array([[.5, .3],[.5, .3]])
    sol = numpy.array([[.3, .4],[.7, .6]])
    sim = evaluate_retrieval(desc, sol)
    t = 0.00001
    gold = 0.983282004984
    assert(gold-t < sim < gold+t)

def run_tests():
    test_evaluate_retrieval()
    print "ok"

if __name__ == "__main__":
    run_tests()
