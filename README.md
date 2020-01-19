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
      featurization/            --->  sub-package for featurization primitives and examples
      distance/                 --->  sub-package for distance measurement primitives and examples
      cluster/                  --->  sub-package for clustering primitives
      dimensionality_reduction/ --->  sub-package for dimensionality reduction primitives
      preprocessing/            --->  sub-package for video preprocessing primitives and examples
      supervised_learning/      --->  sub-package for supervised learning primitives and examples
      unsupervised_learning/    --->  sub-package for learning linear subspace from unlabelled data
      tests/                    --->  sub-package with unit tests 
      pipelines/                --->  contains python code defining example pipelines using Michigan primitives; can be used to generate "runnable" JSON pipelines

Primitives Included

    # Featurization Primitives
    spider.featurization.vgg16
    spider.featurization.audio
    spider.featurization.audo_slicer
    spider.featurization.logmelspectrogram
    spider.featurization.i3d

    # Distance Primitives
    spider.distance.rfd

    # Cluster Primitives
    spider.cluster.kss
    spider.cluster.ekss
    spider.cluster.ssc_cvx
    spider.cluster.ssc_admm
    spider.cluster.ssc_omp

    # Dimensionality Reduction Primitives
    spider.dimensionality_reduction.pcp_ialm
    spider.dimensionality_reduction.go_dec
    spider.dimensionality_reduction.rpca_lbd

    # Preprocessing Primitives
    spider.preprocessing.trecs
    
    # Supervised Learning Primitives
    spider.supervised_learning.owl
    spider.supervised_learning.goturn

    # Unsupervised Learning Primitives
    spider.unsupervised_learning.grasta
    spider.unsupervised_learning.grasta_masked
    spider.unsupervised_learning.grouse

Executables Created

    spider/distance/examples/rfd.py
    spider/featurization/examples/vgg16.py
    spider/featurization/examples/audio.py
    spider/featurization/examples/audio_slicer.py
    spider/featurization/examples/logmelspectrogram.py
    spider/featurization/examples/train_audio.py
    spider/preprocessing/examples/trecs.py
    spider/featurization/examples/i3d.py
    spider/supervised_learning/examples/owl.py
    spider/supervised_learning/examples/goturn.py


License
-------

MIT license.

Setup
-----

This section describes how to set up this project.

1. Clone the repository to your machine and `cd` into its directory.

    ```
    git clone git@github.com:dvdmjohnson/d3m_michigan_primitives.git
    cd spider
    ```

2. To let pip build and install this and the remaining dependencies run:

    ```
    pip3 install --upgrade --upgrade-strategy only-if-needed --no-cache-dir -e .
    ```

3. To install bbn primitives (needed for pipelines and tests) run:

    ```
    pip3 install -e git+https://gitlab.datadrivendiscovery.org/BBN/d3m-bbn-primitives.git@697dabc03c46c1900483bea89d576e82b5a5e4c5#egg=bbn_primitives
    ```
    
    ...and enter d3m login info.

4. To install common primitives (needed for pipelines and tests) run:

    ```
    pip3 install --upgrade-strategy only-if-needed -e git+https://gitlab.com/datadrivendiscovery/common-primitives.git@32508af64512aa0151c8358a0a18c0af5ae18418#egg=common_primitives
    ```

5. To install sklearn wrappers (possibly needed for pipelines) run:

    ```
    pip3 install -e git+https://gitlab.com/datadrivendiscovery/sklearn-wrap.git@889f93af9439fbeb29db961b03e46eaa9e2a7888#egg=sklearn-wrap
    ```

6. Then, to run unit tests:

    ```
    python3 setup.py test
    ```


Uninstall
---------

If you have pip installed, it is easy to uninstall spider

    pip3 uninstall spider

