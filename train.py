import librosa
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def extract(file):
    audio, sr = librosa.load(file)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    return np.mean(mfcc.T, axis=0)

X = []
y = []

# example dataset
files = ["audio1.wav", "audio2.wav"]
labels = [0, 1]

for f, l in zip(files, labels):
    X.append(extract(f))
    y.append(l)

X = np.array(X)

model = Sequential([
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X, y, epochs=10)

model.save("model.h5")
