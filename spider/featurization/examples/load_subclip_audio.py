import argparse
import librosa
import numpy as np

def make_subclips(audio, sr, clip_size, pad=True):
    # Given a list of audio files and corresponding sample rates,
    # return a 2D list of subclips, each of size clip_size
    # Optional padding takes care of audio files shorter than clip size
    clips = []
    for idx, a in enumerate(audio):

        # Size of a single clip in samples
        step = int(sr[idx] * clip_size)

        # Optional padding for short clips
        overhang = len(a) % step
        if overhang != 0 and pad:
            a = np.concatenate([a, np.zeros(step - overhang)])

        subclips = []
        for start in range(0, len(a), step):

            end = start + step
            if end > len(a):
                break

            subclips.append(a[start : end])

    return subclips

def main(audio_file, clip_size):

    # In python 2.7, librosa.load does not correctly handle 24-bit wav files.
    # This is resolved in python 3.x
    # 
    # If the sr parameter is set to None, loads the actual sampling rate
    # from the audio file. Otherwise, will load the audio file and resample
    # it to the given sample rate. This is good if you want all audio at the
    # same sample rate, but can be slow. Default is 22050 Hz.
    audio, sr = librosa.load(audio_file, sr=None)

    # We just have one audio file here, but this should work for any number
    audio_subclips = make_subclips([audio], [sr], 1.0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file', type=str, required=True)
    parser.add_argument('--clip_size', type=float, default=0)
    args = parser.parse_args()
    main(args.audio_file, args.clip_size)
