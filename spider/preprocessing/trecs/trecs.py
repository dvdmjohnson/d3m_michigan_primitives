import typing
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base, transformer
from d3m import container, utils
import collections
import os
import warnings
import stopit
import numpy as np
import cv2
import copy
from common_primitives.utils import combine_columns

__all__ = ('TRECS',)

# see https://gitlab.com/datadrivendiscovery/d3m/tree/devel/d3m/container for valid types
Inputs = container.DataFrame
Outputs = container.List


# describes the hyperparameters of the primitive (i.e. input parameters that affect the primitive's behavior and do not
# change during fitting or producing operations).  See https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/metadata/hyperparams.py
class TRECSHyperparams(hyperparams.Hyperparams):
    default_alpha = hyperparams.Bounded[float](
        lower=0.2,
        upper=3.0,
        default=1.0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="the resampling factor to use when resampling video frames")
    output_frames = hyperparams.Bounded[int](
        lower=1,
        upper=500,
        default=50,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="the number of frames in the output video after resampling")
    trecs_method = hyperparams.Enumeration[str](
        values=['cvr', 'rr', 'sr'],
        default='cvr',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/MetafeatureParameter'],
        description='the choices of resampling strategies')


# note that private variables (i.e class members not defined in the primitive interfaces) are prefaced with "_"
# also note that hyperparameters are referenced using their name as a key value input to the hyperparams object
# the * argument is used here (and on any other methods that takes inputs) to force callers to use keyword inputs
class TRECS(transformer.TransformerPrimitiveBase[Inputs, Outputs, TRECSHyperparams]):
    """
    Primitive for resampling video frames according T-RECS methods.
    """

    # your metadata object must contain all required metadata fields, or your primitive won't validate
    # (and thus won't run.)
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'a99664ce-9c9f-4385-9cd3-c12f41706d8a',
        'version': "0.0.5",
        'name': 'TRECS',
        'description': """Resamples input video frames according to the T-RECS method used to train deep neural networks for
                          for activity recognition that are more robust to variations in the speed of input videos.""",
        'keywords': ['preprocessing', 'video_resampling', 'activity_recognition'],
        'source': {
            'name': 'Michigan',
            'contact': 'mailto:davjoh@umich.edu',
            'uris': [
                # link to file and repo
                'https://gitlab.datadrivendiscovery.org/michigan/spider/blob/master/spider/preprocessing/trecs/trecs.py',
                'https://gitlab.datadrivendiscovery.org/michigan/spider'],
            'citation': """@article{ganesh2018t,
        title={T-RECS: Training for Rate-Invariant Embeddings by Controlling Speed for Action Recognition},
        author={Ganesh, Madan Ravi and Hofesmann, Eric and Min, Byungsu and Gafoor, Nadha and Corso, Jason J},
        journal={arXiv preprint arXiv:1803.08094},
        year={2018}
        } """
        },
        # link to install package
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
             'package_uri': 'git+https://gitlab.datadrivendiscovery.org/michigan/spider.git@{git_commit}#egg=spider'.format(
                 git_commit=utils.current_git_commit(os.path.dirname(__file__)))
             },
            {'type': metadata_module.PrimitiveInstallationType.UBUNTU,
             'package': 'ffmpeg',
             'version': '7:2.8.11-0ubuntu0.16.04.1'}],
        # entry-points path to the primitive (see setup.py)
        'python_path': 'd3m.primitives.data_preprocessing.trecs.Umich',
        # a list of hyperparameters that are high-priorities for tuning
        'hyperparams_to_tune': ['trecs_method', 'output_frames', 'default_alpha'],
        # search https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/metadata/schemas/v0/definitions.json for list of valid algorithm types
        'algorithm_types': [
            metadata_module.PrimitiveAlgorithmType.IMAGE_TRANSFORM],
        # again search https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/metadata/schemas/v0/definitions.json for list of valid primitive families
        'primitive_family': metadata_module.PrimitiveFamily.DATA_PREPROCESSING
    })

    def __init__(self, *, hyperparams: TRECSHyperparams, random_seed: int = 0,
                 docker_containers: typing.Dict[str, base.DockerContainer] = None) -> None:
        """
        Primitive for resampling video frames according T-RECS methods.
        """
        # random_seed and docker_containers inputs should be handled by the superclass
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._trecs_method = hyperparams['trecs_method']
        self._output_frames = hyperparams['output_frames']
        self._default_alpha = hyperparams['default_alpha']

        self._random_state = np.random.RandomState(random_seed)

        assert self._trecs_method in ['cvr', 'sr', 'rr'], "Requested method is not available"

    # training data is set here, rather than in the constructor.  For most methods, outputs will be of type Outputs -
    def set_training_data(self, *, inputs: Inputs) -> None:
        pass

    def _resample_video_sinusoidal(self, video, sample_dims, frame_count, tracker):
        """Return video sampled to a rate chosen sinuloidally based on a random value selected from a uniform distribution.
           Resampling factors are chosen from a range of 0.2 to 3.0.
        Args:
            :video:       Raw input data
            :sample_dims: Number of frames in the output video
            :frame_count: Total number of frames
            :tracker:     Random value sampled from a uniform distribution
        Return:
            Sampled video
        """
        center_alpha = 1.6
        upper_limit = 3.0
        lower_limit = 0.2

        # Sinusoidal variation with alpha being the DC offset
        r_alpha = (center_alpha + (upper_limit - lower_limit) / 2.0 * np.sin(float(tracker)))

        output = self._resample_video(video, sample_dims, frame_count, r_alpha)

        return output

    def _resample_video(self, video, sample_dims, frame_count, alpha):
        """Return video with linearly sampled frames
        Args:
            :video:       Raw input data
            :sample_dims: Number of frames in the output video
            :frame_count: Total number of frames
            :alpha        relative sampling rate
        Return:
            Sampled video
        """

        output = []

        r_alpha = alpha * float(frame_count) / float(sample_dims)

        for i in range(sample_dims):
            output_frame = int(min(r_alpha * i, frame_count - 1))
            output.append(video[output_frame])

        output = np.array(output)

        return output

    # Produce a resampled video given videos as numpy arrays stored in the 'data' key of a DataFrame
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:

        with stopit.ThreadingTimeout(timeout) as timer:
            output_video_list = container.List()

            if 'data' not in inputs.keys():
                raise ValueError("Input DataFrame must contain a 'data' column with numpy arrays containing videos.")

            for input_vid in inputs['data']:
                assert input_vid.ndim == 4, "Data is not in the right shape"
                frame_count, height, width, channels = input_vid.shape

                if self._trecs_method == 'cvr':
                    alpha = self._default_alpha
                    output_video = self._resample_video(input_vid, self._output_frames, frame_count, alpha)

                elif self._trecs_method == 'sr':
                    alpha = self._random_state.uniform(0.2, 3.0)
                    output_video = self._resample_video_sinusoidal(input_vid, self._output_frames, frame_count, alpha)

                elif self._trecs_method == 'rr':
                    alpha = self._random_state.uniform(0.2, 3.0) * np.pi
                    output_video = self._resample_video(input_vid, self._output_frames, frame_count, alpha)

                else:
                    raise ValueError('The requested method is not yet implemented: ', + self._trecs_method)

                output_video_list.append(container.ndarray(output_video))

            return base.CallResult(container.List(output_video_list, generate_metadata=True))

        if timer.state != timer.EXECUTED:
            raise TimeoutError('TRECS produce timed out.')

    # package the full state of the primitive (including hyperparameters and random state)
    # as a serializable dict
    def __getstate__(self) -> dict:
        return {
            'constructor': {
                'hyperparams': self.hyperparams,
                'random_seed': self.random_seed,
                'docker_containers': self.docker_containers,
            },
            'random_state': self._random_state,
        }

    # restores the full state of the primitive (as stored by __getstate__())
    def __setstate__(self, state: dict) -> None:
        self.__init__(**state['constructor'])  # type: ignore
        self._random_state = state['random_state']
        self._rf.random_state = self._random_state

        # placeholder for now, just calls base version.

    @classmethod
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]],
                   hyperparams: TRECSHyperparams) -> typing.Optional[metadata_module.DataMetadata]:
        return super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)
