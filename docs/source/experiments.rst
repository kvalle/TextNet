Experiments
===========

The experiment modules use the rest of the framework to evaluate different versions and aspects of the text network representations.
The various experiments are implemented as functions.

There are four modules.
:mod:`co_occurrence_experiments` for experiments with regular co-occurrence, and :mod:`higher_order_experiments` for higher order co-occurrence networks.
The dependency network representation is tested in :mod:`dependency_experiments`.
The general :mod:`experiments` module contain experiments concerned with several representations, or functions not directly tied to any representation such as, for example, the :func:`~experiments.dataset_stats` function.

**Disclaimer**: These modules are a mess and probably contain a lot of redundant code. 
This is because they contains experiments constructed for specific purposes that are hard to predict ahead of time. 
When done, the experiment functions are left as is, to be available for re-runs later if needed.
As a consequence of many of the experiments, the representations and/or other parts of the code have been changed, but the experiments should still hopefully work as expected.

:mod:`experiments` Module
-------------------------

.. automodule:: experiments
    :members:


:mod:`co_occurrence_experiments` Module
---------------------------------------

.. automodule:: co_occurrence_experiments
    :members:
    
:mod:`higher_order_experiments` Module
---------------------------------------
    
.. automodule:: higher_order_experiments
    :members:
    
:mod:`dependency_experiments` Module
---------------------------------------
    
.. automodule:: dependency_experiments
    :members:

