import typing
from d3m.primitive_interfaces import base, featurization
from d3m import container, utils
import d3m.metadata.base as metadata_module
import d3m.metadata.hyperparams as hyperparams

import numpy as np
import math
import os
import sys
from .utils import audio_feature_extraction

__all__ = ('AudioFeaturization',)

Inputs = container.List
Outputs = container.ndarray

class AudioFeaturizationHyperparams(hyperparams.Hyperparams):
    sampling_rate = hyperparams.Bounded[int](
        lower=1,
        upper=192000,
        default=44100,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='uniform sampling rate of the audio data')
    frame_length = hyperparams.Bounded[float](
        lower=0.0,
        upper=10.0,
        default=0.050,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='duration in seconds that defines the length of the audio processing window')
    overlap = hyperparams.Bounded[float](
        lower=0.0,
        upper=0.999,
        default=0.025,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='duration in seconds that defines the step size taken along the time series during subsequent processing steps')

class AudioFeaturization(featurization.FeaturizationTransformerPrimitiveBase[Inputs, Outputs, AudioFeaturizationHyperparams]):

    """
    Audio featurization primitive for extracting the following bag of features:
      - Zero Crossing Rate
      - Energy
      - Entropy of Energy
      - Spectral Centroid
      - Spectral Spread
      - Spectral Entropy
      - Spectral Flux
      - Spectral Rolloff
      - MFCCs
      - Chroma Vector
      - Chroma Deviation
    """
    metadata = metadata_module.PrimitiveMetadata({
        "id": "2363d81d-7b05-361d-969b-72f3b5070107",
        'version': '0.0.5',
        'name': 'AudioFeaturization',
        'description': """Computes and concatenates a number of handcrafted audio understanding features from input audio 
                           streams.  If long enough, each stream is split into multiple shorter segments and features are
                           computed for each.  The computed features include:
                               - Zero Crossing Rate
                               - Energy
                               - Entropy of Energy
                               - Spectral Centroid
                               - Spectral Spread
                               - Spectral Entropy
                               - Spectral Flux
                               - Spectral Rolloff
                               - MFCCs
                               - Chroma Vector
                               - Chroma Deviation""",
        'keywords': ['feature extraction', 'audio'],
        'source': {
            'name': 'Michigan',
            'contact': 'mailto:davjoh@umich.edu',
            'uris': [
                'https://github.com/dvdmjohnson/d3m_michigan_primitives/blob/master/spider/featurization/audio_featurization/audio_featurization.py',
                'https://github.com/dvdmjohnson/d3m_michigan_primitives'],
            'citation': """@article{giannakopoulos2015pyaudioanalysis,
                title={pyAudioAnalysis: An Open-Source Python Library for Audio Signal Analysis},
                author={Giannakopoulos, Theodoros},
                journal={PloS one},
                volume={10},
                number={12},
                year={2015},
                publisher={Public Library of Science}"""
        },
        'installation': [
            {'type': metadata_module.PrimitiveInstallationType.PIP,
             'package_uri': 'git+https://github.com/dvdmjohnson/d3m_michigan_primitives.git@{git_commit}#egg=spider'.format(
             git_commit=utils.current_git_commit(os.path.dirname(__file__)))
            },
            {'type': metadata_module.PrimitiveInstallationType.UBUNTU,
                 'package': 'ffmpeg',
                 'version': '7:2.8.11-0ubuntu0.16.04.1'}],
        'python_path': 'd3m.primitives.feature_extraction.audio_featurization.Umich',
        'hyperparams_to_tune': ['frame_length'],
        'algorithm_types': [metadata_module.PrimitiveAlgorithmType.INFORMATION_ENTROPY,
                               metadata_module.PrimitiveAlgorithmType.SIGNAL_ENERGY,
                               metadata_module.PrimitiveAlgorithmType.FREQUENCY_TRANSFORM,
                               metadata_module.PrimitiveAlgorithmType.AUDIO_STREAM_MANIPULATION],
        'primitive_family': metadata_module.PrimitiveFamily.FEATURE_EXTRACTION
    })

    def __init__(self, *, hyperparams: AudioFeaturizationHyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, base.DockerContainer] = None) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._sampling_rate: int = hyperparams['sampling_rate']
        self._frame_length: float = hyperparams['frame_length']
        self._overlap: float = hyperparams['overlap']
        self._step: int = max(int((self._frame_length - self._overlap) * self._sampling_rate), 1)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        """
        Compute the bag of features for each ndarray in the input list,
        yielding an output list of the same length. Each entry in the output
        list is an ndarray with variable number of rows corresponding to the
        length of that audio clip and the frame_length and overlap hyperparams.
        """

        X = inputs

        features = []
        missing = []
        meanfeats = None
        featcount = 0
        for i, x in enumerate(X):

            # Handle multi-channel audio data
            if x.ndim > 1 and x.shape[1] > 1:
                x = x.mean(axis=np.argmin(x.shape))
            sampling_rate = self._sampling_rate

            frame_length = self._frame_length

            # Handle time series of insufficient length by padding the sequence with wrapped data
            if x.shape[0] < frame_length * sampling_rate:
                diff = int(frame_length * sampling_rate) - x.shape[0]
                if x.shape[0] == 0:     #handle missing data
                    missing.append(i)
                    features.append(None)
                    continue
                x = np.pad(x, [math.floor(diff/2), math.ceil(diff/2)], 'wrap')

            # Perform audio feature extraction
            feats = audio_feature_extraction(x, sampling_rate, frame_length * sampling_rate, self._step).T.mean(axis=0)
            #currently we are smashing features from each subsequence together, because TA2 can't really handle them separately
            features.append(feats)
            if meanfeats is None:
                meanfeats = feats
            else:
                meanfeats = meanfeats + feats
            featcount = featcount + 1

        #fill in missing data
        meanfeats = meanfeats / featcount
        for i in missing:
            features[i] = meanfeats

        outarray = np.asarray(features)

        return base.CallResult(container.ndarray(outarray, generate_metadata=True))

    #placeholder for now, just calls base version.
    @classmethod
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]], hyperparams: AudioFeaturizationHyperparams) -> typing.Optional[metadata_module.DataMetadata]:
        return super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)
