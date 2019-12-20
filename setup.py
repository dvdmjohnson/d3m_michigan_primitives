#!/usr/bin/env python3

import setuptools.command.build_py
import distutils.cmd
import distutils.log
import setuptools
import subprocess
import unittest
import os, os.path
import sys
import urllib.request, urllib.parse, urllib.error



from setuptools.command.develop import develop
from setuptools.command.install import install

def spider_dl_progress_percentage(count, blocksize, totalsize):
    """ Outputs percentage progress to stdout to track download progress
        
    Arguments:
        count: number of blocks of data retrieved
        blocksize: size of the blocks retrieved
        totalsize: total size of blocks to be retrieved

    Returns:
        None

    Raises:
        None
    """

    percent = int(count*blocksize*100/totalsize)
    sys.stdout.write("\r" + "...%d%%" % percent)
    sys.stdout.flush()

def spider_download_weights(directory, filename, url):
    """ Download VGG16 weights from the given url, and save them as the given filename
        in the specified directory
        
    Arguments:
        directory: path to the directory to which the weights file should be saved
        filename: filename with which to name the downloaded weights file
        url: url from which the weights file should be downloaded

    Returns:
        filepath: the confirmed to exist path to the weights file to be
            loaded into the model

    Raises: 
        AssertError: throws an error if the given path in which to save the
            weights file does not exist
    """
    assert os.path.isdir(directory), 'path to weights directory does not exist.'
    filepath = os.path.join(directory, filename)

    if not os.path.isfile(filepath):
        print(("Weights file not found, downloading from: " + str(url)))
        download = urllib.request.FancyURLopener()
        download.retrieve(url, filepath, reporthook = spider_dl_progress_percentage)
        print("\n")

    assert os.path.isfile(filepath), 'file was not downloaded or saved properly'

class customDevelop(develop):
    """A custom command to install pip dependencies in proper order."""
    
    def run(self):
        """Run command."""
        thisdir = os.path.dirname(os.path.abspath(__file__))
        c = 'pip3 install --upgrade --upgrade-strategy only-if-needed -e '
        calt = 'pip3 install --upgrade --upgrade-strategy only-if-needed '
        cl = []
        #initial install
        develop.run(self)
        #second round dependency installation
        #cl.append(calt + 'librosa')
        #cl.append(calt + 'cvxpy')

        # for com in cl:
            # self.announce(
                  # 'Running command: %s' % str(com),
                  # level=distutils.log.INFO)
            # subprocess.check_call(com, shell=True)

        # #check and download VGG16 weights file
        # weights_url = 'https://umich.box.com/shared/static/dzmxth5l7ql3xggc0hst5h1necjfaurt.h5'
        # weightsdir = thisdir + '/spider/featurization/vgg16/weights'
        # weightsfile = 'vgg16_weights.h5'
        # if not os.path.isdir(weightsdir):
            # os.makedirs(weightsdir)
        # spider_download_weights(weightsdir, weightsfile, weights_url)
        
        # #Download I3D weights file
        # weights_url = 'https://umich.box.com/shared/static/xl06t9sb2c0qnnbh00v6dqr0fq98au0m.npy'
        # weightsdir = thisdir + '/spider/featurization/i3d/weights'
        # weightsfile = 'i3d_rgb_kinetics.npy'
        # if not os.path.isdir(weightsdir):
            # os.makedirs(weightsdir)
        # spider_download_weights(weightsdir, weightsfile, weights_url)

        # #Download CaffeNet weights file
        # weights_url = 'https://umich.box.com/shared/static/lbx9uo2cvruamhey0clcit0w7tufis8w.pth'
        # weightsdir = thisdir + '/spider/supervised_learning/goturn/weights'
        # weightsfile = 'caffenet.pth'
        # if not os.path.isdir(weightsdir):
            # os.makedirs(weightsdir)
        # spider_download_weights(weightsdir, weightsfile, weights_url)

# class buildPyCommand(setuptools.command.build_py.build_py):
#     """Custom build command."""
#     
#     def run(self):
#         self.run_command('pip')
#         setuptools.command.build_py.build_py.run(self)

setuptools.setup(
    name="spider",
    version="0.0.5",
    author="Jason Corso, Laura Balzano and The University of Michigan DARPA D3M Spider Team",
    author_email="jjcorso@umich.edu,girasole@umich.edu,davjoh@umich.edu, alsoltan@umich.edu",
    url="https://github.com/dvdmjohnson/d3m_michigan_primitives",
    license="MIT",
    description="DARPA D3M Spider Project Code",
    install_requires=[
        "d3m (==2019.11.10)",
        "numpy (>=1.14.3)",
        "scipy (>=0.19.0)",
        "scikit-image (>=0.13.1)",
        "scikit-learn (>=0.18.1)",
        "matplotlib (>=1.5.1)",
        "Pillow (>=4.1.1)",
        "h5py (>=2.7.0)",
        "opencv-python (>=3.0.0)",
        "keras (>=2.0.4)",
        "tensorflow-gpu (>=1.1.0)",
        "pandas (>=0.19.2)",
        "typing (>=3.6.2)",
        "stopit (>=1.1.1)",
        "librosa (>=0.5.1)",
        "torch (>=0.3.1)",
        "cvxpy (>=1.0.23)",
        "resampy (==0.2.1)",
    ],
    packages=["spider",
                "spider.preprocessing",
                "spider.preprocessing.trecs",
                "spider.featurization",
                "spider.featurization.vgg16", 
                "spider.featurization.audio_featurization",
                "spider.featurization.logmelspectrogram",
                "spider.featurization.audio_slicer",
                "spider.featurization.i3d",
                "spider.distance",
                "spider.distance.rfd",
                "spider.cluster",
                "spider.cluster.ssc_cvx",
                "spider.cluster.ssc_admm",
                "spider.cluster.kss",
                "spider.cluster.ekss",
                "spider.cluster.ssc_omp",
                "spider.dimensionality_reduction",
                "spider.dimensionality_reduction.pcp_ialm",
                "spider.dimensionality_reduction.go_dec",
                "spider.dimensionality_reduction.rpca_lbd",
                "spider.supervised_learning",
                "spider.supervised_learning.owl",
                "spider.supervised_learning.goturn",
                "spider.unsupervised_learning.grasta",
                "spider.unsupervised_learning.grasta_masked",
                "spider.unsupervised_learning.grouse",
                "spider.pipelines"],
    keywords='d3m_primitive',
    #scripts    =[],
    #ext_modules=[],
    cmdclass={
              'develop': customDevelop,
#               'build_py': buildPyCommand,
              },
    entry_points = {
    'd3m.primitives': [
        'data_compression.grasta.Umich = spider.unsupervised_learning.grasta.grasta:GRASTA',
        'data_compression.grasta_masked.Umich = spider.unsupervised_learning.grasta_masked.grasta_masked:GRASTA_MASKED',
        'data_compression.grouse.Umich = spider.unsupervised_learning.grouse.grouse:GROUSE',
        'data_compression.go_dec.Umich = spider.dimensionality_reduction.go_dec.go_dec:GO_DEC',
        'data_compression.pcp_ialm.Umich = spider.dimensionality_reduction.pcp_ialm.pcp_ialm:PCP_IALM',
        'data_compression.rpca_lbd.Umich = spider.dimensionality_reduction.rpca_lbd.rpca_lbd:RPCA_LBD',
        'data_preprocessing.trecs.Umich = spider.preprocessing.trecs.trecs:TRECS',
        'feature_extraction.vgg16.Umich = spider.featurization.vgg16.vgg16:VGG16',
        'feature_extraction.audio_featurization.Umich = spider.featurization.audio_featurization.audio_featurization:AudioFeaturization',
        'data_preprocessing.audio_slicer.Umich = spider.featurization.audio_slicer.audio_slicer:AudioSlicer',
        'feature_extraction.log_mel_spectrogram.Umich = spider.featurization.logmelspectrogram.logmelspectrogram:LogMelSpectrogram',
        'feature_extraction.i3d.Umich = spider.featurization.i3d.i3d:I3D',
        'similarity_modeling.rfd.Umich = spider.distance.rfd.rfd:RFD',
        'clustering.ekss.Umich = spider.cluster.ekss.ekss:EKSS',
        'clustering.kss.Umich = spider.cluster.kss.kss:KSS',
        'clustering.ssc_admm.Umich = spider.cluster.ssc_admm.ssc_admm:SSC_ADMM',
        'clustering.ssc_cvx.Umich = spider.cluster.ssc_cvx.ssc_cvx:SSC_CVX',
        'clustering.ssc_omp.Umich = spider.cluster.ssc_omp.ssc_omp:SSC_OMP',
        'regression.owl_regression.Umich = spider.supervised_learning.owl.owl:OWLRegression',
        'learner.goturn.Umich = spider.supervised_learning.goturn.goturn:GoTurn'
        ],
    },

    test_suite='spider.tests.suite'
)


