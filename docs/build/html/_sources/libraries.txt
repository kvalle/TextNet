Used Libraries 
==============

The implementation relies heavily on a few central third party libraries. These are briefly described
in this section.


NumPy and SciPy
---------------

SciPy is an open-source library for mathematics, science, and engineering. SciPy
depends on NumPy, a library which provides convenient and fast N-dimensional array and
matrix manipulation, as well as many tools for numerical computation. The libraries are easy
to use, but powerful for manipulating numbers in many ways.

The most useful aspects in this project were the nd-arrays and matrices from NumPy, and the
sparse matrix representations from SciPy. The scipy.spatial.distance.cdist function also
proved valuable for efficient computation of vector similarities.

NumPy and SciPy are available at http://scipy.org and http://numpy.scipy.org, respectively.

NetworkX 
--------

NetworkX is a Python package for the creation and manipulation of graphs and complex
networks. It enables study of structure and dynamics of networks, and comes with many useful
functions.

We have used NetworkX’s DiGraphs as our basic datastructure for the network representations.
Among the more useful features of the library were some of the graph centrality algorithms and
functions for extracting global and local properties from the graphs.

NetworkX is available from http://networkx.lanl.gov.


NLTK
---- 

Python’s Natural Language Toolkit (NLTK) is a powerful tool for working with natural lan-
guage processing (NLP). It provide functionality for a wide variety of NLP tasks.
Only a small, but useful part of the library is used in this project. Of most use were the stem-
mers, stop-word lists, tokenizers for tokens and sentences, and probability distributions for
calculating the frequency-based measures.

NLTK is available from http://nltk.org.

Matplotlib
----------

Matplotlib is a 2D plotting library able to produce high quality graphics of many types.
We have utilized it to create plots, histograms and bar charts, many of which are used in this
report. The library has a relatively easy interface, able to produce figures with a few lines of
code.

Matplotlib is available at http://matplotlib.sourceforge.net.

JPype
-----

JPype is a Java-to-Python integration library, allowing python programs full access to java
class libraries. This is not done through re-implementing python on the Java Virtual Machine,
like the JPython project, but rather through interfacing at the native level in both virtual ma-
chines.

Using JPype, we were able to use the Stanford dependency parser directly, even though it was
implemented in Java.

Available from http://jpype.sourceforge.net.

