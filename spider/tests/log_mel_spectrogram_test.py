'''
    spider.tests: test_log_mel_spectrogram.py

    Max Morrison

    Unit tests for the spider.preprocessing.logmelspectrogram module
'''

import librosa
import unittest
import numpy as np
import warnings
from spider.featurization.logmelspectrogram import LogMelSpectrogram, LogMelSpectrogramHyperparams

class TestLogMelSpectrogram(unittest.TestCase):

    def setUp(self):
        """Setup the audio testing environment for Python unittest"""

        # Get a sample clip for testing
        audio_file = librosa.util.example_audio_file()
        self._audio, self._sr = librosa.load(audio_file)
        self._audio_length = len(self._audio)
        hyperparams = LogMelSpectrogramHyperparams(
            sampling_rate=self._sr,
            mel_bands=128,
            n_fft=1024,
            hop_length=1024)
        self._featurizer = LogMelSpectrogram(hyperparams=hyperparams)

    def testDefault(self):
        """Test the default parameters of the logmelspectrogram"""

        features = self._featurizer.produce(inputs=[self._audio]).value

        self.assertEqual(features[0].shape, (1324, 128))

    def testSizing(self):
        """Test the effect of parameter changes on output size"""

        hyperparams = LogMelSpectrogramHyperparams(
            sampling_rate=self._sr,
            mel_bands=256,
            n_fft=2048,
            hop_length=2048)
        featurizer = LogMelSpectrogram(hyperparams=hyperparams)
        features = featurizer.produce(inputs=[self._audio]).value

        self.assertEqual(features[0].shape, (662, 256))

    def testNoop(self):
        """Test that an empty array is well-handled"""

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')            
            features = self._featurizer.produce(inputs=[np.array([])]).value

            self.assertEqual(len(w), 1)
            self.assertEqual(features[0].size, 0)

    def testMultiple(self):
        """Test that multiple time series in a single call are well-handled"""

        audio = np.array_split(self._audio, 3)

        features = self._featurizer.produce(inputs=audio).value

        for feature in features:
            self.assertEqual(feature.shape, (442, 128))
