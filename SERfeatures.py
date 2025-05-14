import os
import glob
import numpy as np
import librosa

# Emotion labels
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
observed_emotions = list(emotions.values())

# Extracting Features
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    result = np.array([])

    if chroma:
        stft = np.abs(librosa.stft(X))

    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))

    if chroma:
        chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_feat))

    if mel:
        mel_feat = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel_feat))

    return result

# Data loader Function
def load_data(test_size=0.2):
    from sklearn.model_selection import train_test_split
    x, y = [], []
    files = sorted(glob.glob('RAVDESS_data/**/*.wav', recursive=True))

    print(f"Number of audio files found: {len(files)}")

    for file in files:
        file_name = os.path.basename(file)
        emotion = emotions.get(file_name.split("-")[2])

        if emotion not in observed_emotions:
            continue

        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(features)
        y.append(emotion)

    return train_test_split(np.array(x), y, test_size=test_size, train_size=0.75, random_state=9)

