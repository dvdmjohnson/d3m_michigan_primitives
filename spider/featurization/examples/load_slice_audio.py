import argparse
import librosa

def slice_audio(audio, sr, start, end):
    # Given a list of audio files and corresponding sample rates,
    # start locations in seconds, and end locations in seconds, return a list
    # of sliced audio clips
    sliced_audio = []
    for i in range(len(audio)):
    	sliced_audio.append(
            audio[i][int(start[i] * sr[i]) : int(end[i] * sr[i])])
    return sliced_audio

def main(audio_file, start, end):

    # In python 2.7, librosa.load does not correctly handle 24-bit wav files.
    # This is resolved in python 3.x
    # 
    # If the sr parameter is set to None, loads the actual sampling rate
    # from the audio file. Otherwise, will load the audio file and resample
    # it to the given sample rate. This is good if you want all audio at the
    # same sample rate, but can be slow. Default is 22050 Hz.
    audio, sr = librosa.load(audio_file, sr=None)

    # We just have one audio file here, but this should work for any number
    sliced_audio = slice_audio([audio], [sr], [start], [end])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file', type=str, required=True)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--end', type=float, default=None)
    args = parser.parse_args()
    main(args.audio_file, args.start, args.end)