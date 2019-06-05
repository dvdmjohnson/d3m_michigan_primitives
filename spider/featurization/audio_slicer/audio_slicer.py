import typing
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base, featurization
from d3m import container, utils

import numpy as np
import os


__all__ = ('AudioSlicer',)

Inputs = container.List
Outputs = container.List

class AudioSlicerHyperparams(hyperparams.Hyperparams):
    sampling_rate = hyperparams.Bounded[int](
        lower=1,
        upper=192000,
        default=44100,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='uniform sampling rate of the audio data')
    frame_length = hyperparams.Bounded[float](
        lower=0.0,
        upper=100.0,
        default=0.050,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='duration in seconds that defines the length of the audio processing window')
    overlap = hyperparams.Bounded[float](
        lower=0.0,
        upper=99.99,
        default=0.025,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='duration in seconds that defines the step size taken along the time series during subsequent processing steps')
    pad = hyperparams.Enumeration[int](
        values=[0,1],
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='indicates whether clips shorter than frame_length should be padded with zeros to a duration of exactly frame_length')

class AudioSlicer(featurization.FeaturizationTransformerPrimitiveBase[Inputs, Outputs, AudioSlicerHyperparams]):
    
    """
    Audio utility for splitting audio data stored in a numpy array into
    equal-length clips with optional overlap
    """
    metadata = metadata_module.PrimitiveMetadata({
        "id": "a0a60c4c-4e69-3ebe-81c9-7432c5b41c38",
        'version': '0.0.5',
        'name': 'AudioSlicer',
        'description': """Slices long audio segments into a number of equal-length shorter segments""",
        'keywords': ['feature_extraction', 'audio'],
        'source': {
            'name': 'Michigan',
            'contact': 'mailto:davjoh@umich.edu',
            'uris': [
                'https://github.com/dvdmjohnson/d3m_michigan_primitives/blob/master/spider/featurization/audio_slicer/audio_slicer.py',
                'https://github.com/dvdmjohnson/d3m_michigan_primitives'],
        },
        'installation': [
            {'type': metadata_module.PrimitiveInstallationType.PIP,
             'package_uri': 'git+https://github.com/dvdmjohnson/d3m_michigan_primitives.git@{git_commit}#egg=spider'.format(
             git_commit=utils.current_git_commit(os.path.dirname(__file__)))
            },
            {'type': metadata_module.PrimitiveInstallationType.UBUNTU,
                 'package': 'ffmpeg',
                 'version': '7:2.8.11-0ubuntu0.16.04.1'}],
        'python_path': 'd3m.primitives.data_preprocessing.audio_slicer.Umich',
        'hyperparams_to_tune': ['frame_length'],
        'algorithm_types': [metadata_module.PrimitiveAlgorithmType.AUDIO_STREAM_MANIPULATION],
        'primitive_family': metadata_module.PrimitiveFamily.DATA_PREPROCESSING
    })

    def __init__(self, *, hyperparams: AudioSlicerHyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, base.DockerContainer] = None) -> None:
        """
        Audio utility for splitting audio data stored in a numpy array into
        equal-length clips with optional overlap
        """
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._sampling_rate: int = hyperparams['sampling_rate']
        self._frame_length: float = hyperparams['frame_length']
        self._overlap: float = hyperparams['overlap']
        self._pad: bool = hyperparams['pad']
        self._samples_per_clip: int = int(self._frame_length * self._sampling_rate)
        self._step: int = max(int((self._frame_length - self._overlap) * self._sampling_rate), 1)

        if self._overlap >= self._frame_length:
            raise ValueError(
                'AudioSlicer: Consecutive audio frames of length ' +\
                str(self._frame_length) + ' seconds cannot facilitate ' +\
                str(self._overlap) + 'seconds of overlap.'
            )

    def produce(self, *, inputs: Inputs, iterations: int = None) -> base.CallResult[Outputs]:
        """
        Splits each audio file into slices. Each file is represented in the
        input list as an ndarray. The output list contains the same number of
        elements, with each element having a number of rows determined by the
        clip length and the hyperparameters frame_length, overlap, and pad.
        """

        X = inputs

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

            # Iterate through audio extracting clips
            clips = []
            for i in range(0, len(x), self._step):
                if i + self._samples_per_clip <= len(x):
                    clips.append(x[i : i + self._samples_per_clip])
                elif self._pad:
                    clips.append(
                        np.concatenate([
                            x[i:],
                            np.zeros(
                                self._samples_per_clip - len(x[i:]))
                        ])
                    )

            features.append(container.ndarray(clips))

        return base.CallResult(container.ndarray(features, generate_metadata=True))

    #placeholder for now, just calls base version.
    @classmethod
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]], hyperparams: AudioSlicerHyperparams) -> typing.Optional[metadata_module.DataMetadata]:
        return super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)