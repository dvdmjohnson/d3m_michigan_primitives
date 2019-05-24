import librosa
import numpy as np
import os
import pandas as pd
from spider.featurization.logmelspectrogram import LogMelSpectrogram

# Read the test data csv
csv_file='data/testAudioData.csv'
df = pd.read_csv(csv_file)

# Read in the audio data specified by the csv
data = []
for idx, row in df.iterrows():
	filename = os.path.join('data/raw_data', row['filename'])
	datum, sampling_rate = librosa.load(filename)
	data.append(datum)

mel_bands = 128   # Number of frequency bins to sample
n_fft = 1024      # FFT window length in samples
hop_length = 1024 # Distance between adjacent windows in samples

featurizer = LogMelSpectrogram(sampling_rate, mel_bands, n_fft, hop_length)
features = featurizer.produce(data)

np.savetxt('features.csv', features[0])