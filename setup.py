import setuptools

setuptools.setup(
    name='spider',
    version='0.0.5',
    author='Jason Corso, Laura Balzano and The University of Michigan DARPA D3M Spider Team',
    author_email='jjcorso@umich.edu,girasole@umich.edu,davjoh@umich.edu, alsoltan@umich.edu',
    url='https://github.com/dvdmjohnson/d3m_michigan_primitives',
    license='MIT',
    description='DARPA D3M Spider Project Code',
    install_requires=[
        'd3m (==2020.1.9)',
        'numpy (>=1.14.3)',
        'scipy (>=0.19.0)',
        'scikit-image (>=0.13.1)',
        'scikit-learn (>=0.18.1)',
        'matplotlib (>=1.5.1)',
        'Pillow (==6.2.1)',
        'h5py (>=2.7.0)',
        'opencv-python (>=3.0.0)',
        'keras (>=2.0.4)',
        'tensorflow-gpu (>=1.1.0)',
        'pandas (>=0.19.2)',
        'typing (>=3.6.2)',
        'stopit (>=1.1.1)',
        'librosa (==0.6.2)',
        'torch (==1.3.1)',
        'torchvision (==0.4.2)',
        'cvxpy (>=1.0.23)',
    ],
    packages=[
        'spider',
        'spider.preprocessing',
        'spider.preprocessing.trecs',
        'spider.featurization',
        'spider.featurization.vgg16',
        'spider.featurization.audio_featurization',
        'spider.featurization.logmelspectrogram',
        'spider.featurization.audio_slicer',
        'spider.featurization.i3d',
        'spider.distance',
        'spider.distance.rfd',
        'spider.cluster',
        'spider.cluster.ssc_cvx',
        'spider.cluster.ssc_admm',
        'spider.cluster.kss',
        'spider.cluster.ekss',
        'spider.cluster.ssc_omp',
        'spider.dimensionality_reduction',
        'spider.dimensionality_reduction.pcp_ialm',
        'spider.dimensionality_reduction.go_dec',
        'spider.dimensionality_reduction.rpca_lbd',
        'spider.supervised_learning',
        'spider.supervised_learning.owl',
        'spider.supervised_learning.goturn',
        'spider.unsupervised_learning.grasta',
        'spider.unsupervised_learning.grasta_masked',
        'spider.unsupervised_learning.grouse',
        'spider.pipelines'
    ],
    keywords='d3m_primitive',
    entry_points = {
        'd3m.primitives': [
            'data_compression.grasta.Umich = spider.unsupervised_learning.grasta.grasta:GRASTA',
            # 'data_compression.grasta_masked.Umich = spider.unsupervised_learning.grasta_masked.grasta_masked:GRASTA_MASKED',
            # 'data_compression.grouse.Umich = spider.unsupervised_learning.grouse.grouse:GROUSE',
            # 'data_compression.go_dec.Umich = spider.dimensionality_reduction.go_dec.go_dec:GO_DEC',
            # 'data_compression.pcp_ialm.Umich = spider.dimensionality_reduction.pcp_ialm.pcp_ialm:PCP_IALM',
            # 'data_compression.rpca_lbd.Umich = spider.dimensionality_reduction.rpca_lbd.rpca_lbd:RPCA_LBD',
            # 'data_preprocessing.trecs.Umich = spider.preprocessing.trecs.trecs:TRECS',
            # 'feature_extraction.vgg16.Umich = spider.featurization.vgg16.vgg16:VGG16',
            # 'feature_extraction.audio_featurization.Umich = spider.featurization.audio_featurization.audio_featurization:AudioFeaturization',
            # 'data_preprocessing.audio_slicer.Umich = spider.featurization.audio_slicer.audio_slicer:AudioSlicer',
            # 'feature_extraction.log_mel_spectrogram.Umich = spider.featurization.logmelspectrogram.logmelspectrogram:LogMelSpectrogram',
            # 'feature_extraction.i3d.Umich = spider.featurization.i3d.i3d:I3D',
            # 'similarity_modeling.rfd.Umich = spider.distance.rfd.rfd:RFD',
            # 'clustering.ekss.Umich = spider.cluster.ekss.ekss:EKSS',
            # 'clustering.kss.Umich = spider.cluster.kss.kss:KSS',
            # 'clustering.ssc_admm.Umich = spider.cluster.ssc_admm.ssc_admm:SSC_ADMM',
            # 'clustering.ssc_cvx.Umich = spider.cluster.ssc_cvx.ssc_cvx:SSC_CVX',
            # 'clustering.ssc_omp.Umich = spider.cluster.ssc_omp.ssc_omp:SSC_OMP',
            'regression.owl_regression.Umich = spider.supervised_learning.owl.owl:OWLRegression',
            # 'learner.goturn.Umich = spider.supervised_learning.goturn.goturn:GoTurn'
        ],
    },
    test_suite='spider.tests.suite'
)
