import typing
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base, featurization
from d3m import container, utils

import librosa
import numpy as np
import stopit
import sys
import os
import time
import warnings

__all__ = ('LogMelSpectrogram',)

Inputs = container.List
Outputs = container.List

class LogMelSpectrogramHyperparams(hyperparams.Hyperparams):
    sampling_rate = hyperparams.Bounded[int](
        lower=1,
        upper=192000,
        default=44100,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='uniform sampling rate of the audio data')
    mel_bands = hyperparams.Bounded[int](
        lower=1,
        upper=2048,
        default=128,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='integer number of mel frequency filters used to compute the spectrogram')
    n_fft = hyperparams.Bounded[int](
        lower=32,
        upper=8192,
        default=1024,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='integer length in samples of the fft window')
    hop_length = hyperparams.Bounded[int](
        lower=32,
        upper=8192,
        default=1024,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='integer length in samples between successive audio frames')

class LogMelSpectrogram(featurization.FeaturizationTransformerPrimitiveBase[Inputs, Outputs, LogMelSpectrogramHyperparams]):

    """
    Utility for computing audio spectrograms
    Compute the spectrogram of each ndarray in the input list, yielding an
    output list of the same length. Each entry in the output list is a 2D
    array containing the spectrogram of the input audio.
    """

    metadata = metadata_module.PrimitiveMetadata({
        "id": "1e4a39a4-10ee-3d1a-9dc7-24b915f86130",
        'version': '0.0.5',
        'name': 'LogMelSpectrogram',
        'description': """Computes the mel-frequency cepstrum of input audio clips, describing the short-term power
                           spectrum of the clips' sound.""",
        'keywords': ['feature_extraction', 'audio'],
        'source': {
            'name': 'Michigan',
            'contact': 'mailto:davjoh@umich.edu',
            'uris': [
                'https://github.com/dvdmjohnson/d3m_michigan_primitives/blob/master/spider/featurization/logmelspectrogram/logmelspectrogram.py',
                'https://github.com/dvdmjohnson/d3m_michigan_primitives']
        },
        'installation': [
            {'type': metadata_module.PrimitiveInstallationType.PIP,
             'package': 'librosa',
             'version': '0.5.1'
            },
            {'type': metadata_module.PrimitiveInstallationType.PIP,
             'package': 'cvxpy',
             'version': '0.4.11'
            },
            {'type': metadata_module.PrimitiveInstallationType.PIP,
             'package_uri': 'git+https://github.com/dvdmjohnson/d3m_michigan_primitives.git@{git_commit}#egg=spider'.format(
             git_commit=utils.current_git_commit(os.path.dirname(__file__)))
            },
            {'type': metadata_module.PrimitiveInstallationType.UBUNTU,
                 'package': 'ffmpeg',
                 'version': '7:2.8.11-0ubuntu0.16.04.1'}],
        'python_path': 'd3m.primitives.feature_extraction.log_mel_spectrogram.Umich',
        'hyperparams_to_tune': ['mel_bands', 'n_fft'],
        'algorithm_types': [metadata_module.PrimitiveAlgorithmType.FREQUENCY_TRANSFORM,
                            metadata_module.PrimitiveAlgorithmType.AUDIO_STREAM_MANIPULATION],
        'primitive_family': metadata_module.PrimitiveFamily.FEATURE_EXTRACTION
    })

    def __init__(self, *, hyperparams: LogMelSpectrogramHyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, base.DockerContainer] = None) -> None:
        """
        Utility for computing audio spectrograms
        """
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._sampling_rate: int = hyperparams['sampling_rate']
        self._mel_bands: int = hyperparams['mel_bands']
        self._n_fft: int = hyperparams['n_fft']
        self._hop_length: int = hyperparams['hop_length']

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        """
        Compute the spectrogram of each ndarray in the input list, yielding an
        output list of the same length. Each entry in the output list is a 2D
        array containing the spectrogram of the input audio.
        """

        X = inputs

        with stopit.ThreadingTimeout(timeout) as timer:

            features = container.List()
            for i, x in enumerate(X):

                # Handle multi-channel audio data
                if x.ndim > 2 or (x.ndim == 2 and x.shape[0] > 2):
                    raise ValueError(
                        'Time series ' + str(i) + ' found with ' + \
                        'incompatible shape ' + str(x.shape)  + '.'
                    )
                elif x.ndim == 2:
                    x = x.mean(axis=0)

                # Handle time series of insufficient length
                if x.shape[0] < self._n_fft:
                    warnings.warn(
                        'Cannot construct a fft window of length ' +          \
                        str(self._n_fft) + ' seconds from input ' +      \
                        str(i) + ' of length ' + str(x.shape[0]) + '. ' + \
                        'Returning empty np.array in output index ' +         \
                        str(i+1) + '.',
                        RuntimeWarning
                    )
                    features.append(np.array([]))
                    continue

                # Compute the mel-scaled spectrogram
                melspec = librosa.feature.melspectrogram(
                    x,
                    sr=self._sampling_rate,
                    n_mels = self._mel_bands,
                    n_fft=self._n_fft,
                    hop_length=self._hop_length
                )

                # Convert spectrogram amplitude to log-scale
                features.append(container.ndarray(librosa.power_to_db(melspec).T))

            return base.CallResult(features)

        if timer.state != timer.EXECUTED:
            raise TimeoutError('LogMelSpectrogram produce timed out.')
            
    #placeholder for now, just calls base version.
    @classmethod
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]], hyperparams: LogMelSpectrogramHyperparams) -> typing.Optional[metadata_module.DataMetadata]:
        return super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)