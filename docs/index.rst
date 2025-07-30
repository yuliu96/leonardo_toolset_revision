.. leonardo documentation master file, created by
   sphinx-quickstart on Sun Nov 24 16:50:21 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|Docs|

Leonardo
=======

**A toolset to remove sample-induced aberrations in light sheet microscopy images**

:mod:`Leonardo` is an AI-empowered image processing toolbox that is able to resolve sample-induced 
aberrations for light-sheet fluorescence microscopy: (1) :mod:`Leonardo-DeStripe` removes the stripe artifacts caused by light absorption; 
(2) :mod:`Leonardo-FUSE` reconstructs one single high-quality image from dual-sided illumination and/or dual-sided detection while 
eliminating optical distortions (ghosts) caused by light refraction.

.. image:: outline.jpg
    :alt: leonardo title figure
    :width: 900px
    :align: center
    :target: https://doi.org/10.21203/rs.3.rs-5853941/v1


|
Manuscript
----------

Preprint is available `here <https://doi.org/10.21203/rs.3.rs-5853941/v1>`_.

.. toctree::
   :caption: General
   :maxdepth: 2

   installation
   tutorials
   api

.. |Docs| image:: https://img.shields.io/readthedocs/basicpy
    :target: https://basicpy.readthedocs.io/en/latest/
    :alt: Documentation
