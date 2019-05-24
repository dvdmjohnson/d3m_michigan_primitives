'''
    spider.tests: test_featurization_base.py

    Max Morrison

    Unit tests for the spider.featurization.audio module
'''

import librosa
import numpy as np
import os
import unittest
import warnings
from d3m import container

from spider.featurization.audio_featurization import AudioFeaturization, AudioFeaturizationHyperparams

class TestAudio(unittest.TestCase):

    def setUp(self):
        """Setup the audio testing environment for Python unittest"""

        data_path = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(data_path, 'data/32318.mp3')

        self._audio, self._sr = librosa.load(data_file, sr=None)

        self._hyperparams = AudioFeaturizationHyperparams(
            sampling_rate=self._sr,
            frame_length=0.050,
            overlap=0.025)
        self._featurizer = AudioFeaturization(hyperparams=self._hyperparams)

    def test_single_time_series(self):
        """Verify the output shape of a single audio time series"""

        features = container.ndarray(self._featurizer.produce(inputs=[self._audio]).value)

        # Different versions of librosa / ffmpeg may load audio
        # with small differences in sample length
        self.assertTrue(features.shape == (1, 34))

    def test_multiple_time_series(self):
        """Verify the output shape of a list of audio time series"""

        data = np.array_split(self._audio, 5)

        features = container.ndarray(self._featurizer.produce(inputs=data).value)

        self.assertEqual(features.shape, (5, 34))
