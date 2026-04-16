import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model.h5")

st.title("EchoGraph – Audio Intelligence")

file = st.file_uploader("Upload Audio", type=["wav"])

def extract(file):
    audio, sr = librosa.load(file)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    return np.mean(mfcc.T, axis=0)

if file:
    features = extract(file)
    pred = model.predict(np.expand_dims(features, axis=0))

    st.write("Environment:", "Safe" if pred < 0.5 else "Alert")
