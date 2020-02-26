spider -- The University of Michigan DARPA D3M SPIDER Project CodeBase
=================================

This codebase includes all methods and primitives that comprise the SPIDER 
project.  These do include the featurization primitives that we contributed as 
well as the featurization baseclass that provides a standard protocol for 
accessing and implementing the featurization across the D3M program.


Maintained by Jason Corso, Laura Balzano and DARPA D3M SPIDER project team members
at the University of Michigan.

Brief Description and File System Layout

    spider/                     --->  main package
      cluster/                  --->  sub-package for clustering primitives
      pipelines/                --->  contains python code defining example pipelines using Michigan primitives; can be used to generate "runnable" JSON pipelines
      supervised_learning/      --->  sub-package for supervised learning primitives and examples
      tests/                    --->  sub-package with unit tests 
      unsupervised_learning/    --->  sub-package for learning linear subspace from unlabelled data

Primitives Included

    # Cluster Primitives
    spider.cluster.ekss
    spider.cluster.kss
    spider.cluster.ssc_cvx
    spider.cluster.ssc_admm
    spider.cluster.ssc_omp

    # Supervised Learning Primitives
    spider.supervised_learning.owl

    # Unsupervised Learning Primitives
    spider.unsupervised_learning.grasta


License
-------

MIT license.
