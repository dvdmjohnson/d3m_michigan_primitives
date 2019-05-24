'''
    spider.tests: test_audio_slicer.py

    Max Morrison

    Unit tests for the spider.preprocessing.audio_slicer module
'''

import librosa
import unittest
import math
import numpy as np
from spider.featurization.audio_slicer import AudioSlicer, AudioSlicerHyperparams

class TestAudioSlicer(unittest.TestCase):

    def setUp(self):
        """Setup the audio testing environment for Python unittest"""

        # Get a sample clip for testing
        audio_file = librosa.util.example_audio_file()
        self._audio, self._sr = librosa.load(audio_file)
        self._audio_length = len(self._audio)
        self._frame_length = 3.0
        self._samples_per_clip = int(self._frame_length * self._sr)
        hyperparams = AudioSlicerHyperparams(
            sampling_rate=self._sr,
            frame_length=self._frame_length,
            overlap=0.050,
            pad=1)
        self._featurizer = AudioSlicer(hyperparams=hyperparams)

    def testDefault(self):
        """Test the default setting of the audio_slicer"""

        num_clips = math.ceil(
            self._audio_length / float(self._samples_per_clip)
        )

        features = self._featurizer.produce(inputs=[self._audio]).value

        # Should be 2D numpy array
        self.assertEqual(features[0].ndim, 2)

        # Should have num_clips rows with samples_per_clip columns
        self.assertEqual(
            features[0].shape, (num_clips, self._samples_per_clip)
        )

        # Should not have changed the underlying data
        self.assertTrue(np.array_equal(
            self._audio[:self._samples_per_clip],
            features[0][0,:]
        ))

    def testOverlap(self):
        """Test the overlap functionality to split clips with shared data"""

        hyperparams = AudioSlicerHyperparams(
            sampling_rate=self._sr,
            frame_length=self._frame_length,
            overlap=1.5,
            pad=1)
        featurizer = AudioSlicer(hyperparams=hyperparams)

        num_clips = math.ceil(self._audio_length / (float(self._samples_per_clip) / 2))

        features = featurizer.produce(inputs=[self._audio]).value

        # Should have num_clips rows with samples_per_clip columns
        self.assertEqual(features[0].shape, (num_clips, self._samples_per_clip))

        # Should not have changed the underlying data
        second_slice = self._audio[self._samples_per_clip // 2 : 3*self._samples_per_clip // 2]
        self.assertTrue(np.array_equal(second_slice, features[0][1,:]))

    def testNoop(self):
        """Test that clipping audio at its same length does nothing"""

        samples_per_clip = self._audio_length
        num_clips = 1

        frame_length = float(self._audio_length) / self._sr

        hyperparams = AudioSlicerHyperparams(
            sampling_rate=self._sr,
            frame_length=frame_length,
            overlap=0.0,
            pad=1)
        featurizer = AudioSlicer(hyperparams=hyperparams)

        features = featurizer.produce(inputs=[self._audio]).value

        # Should have 1 row and same length column as original
        self.assertEqual(features[0].shape, (num_clips, samples_per_clip))

        # Should not have changed the underlying data
        self.assertTrue(np.array_equal(self._audio, features[0][0,:]))

    def testMultiple(self):
        """Test processing of multiple clips in one call"""

        audio = np.array_split(self._audio, 3)

        hyperparams = AudioSlicerHyperparams(
            sampling_rate=self._sr,
            frame_length=1.0,
            overlap=0.050,
            pad=1)
        featurizer = AudioSlicer(hyperparams=hyperparams)
        num_clips = 22

        samples_per_clip = int(1.0 * self._sr)

        features = featurizer.produce(inputs=audio).value

        for feature in features:
            self.assertEqual(feature.shape, (num_clips, samples_per_clip))

    def testPadding(self):
        """Test the padding capabilities for short clips"""

        hyperparams = AudioSlicerHyperparams(
            sampling_rate=self._sr,
            frame_length=1.0,
            overlap=0.0,
            pad=1)
        featurizer = AudioSlicer(hyperparams=hyperparams)

        short_clip = self._audio[:self._sr - 1]

        features = featurizer.produce(inputs=[short_clip]).value

        self.assertEqual(features[0].shape, (1, self._sr))

        hyperparams = AudioSlicerHyperparams(
            sampling_rate=self._sr,
            frame_length=1.0,
            overlap=0.050,
            pad=0)
        featurizer = AudioSlicer(hyperparams=hyperparams)

        features = featurizer.produce(inputs=[short_clip]).value

        self.assertEqual(features[0].size, 0)

if __name__ == '__main__':
    unittest.main()

