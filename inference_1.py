import numpy as np
import librosa
import tensorflow as tf
import os


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


model = tf.keras.models.load_model("model.h5")
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

emotion_mapping = {
    1: "Neutral",
    2: "Happy",
    3: "Sad",
    4: "Angry",
    5: "Fearful",
    6: "Disgust",
    7: "Surprised"
}


def extract_mfcc(file_path, sr=22050, n_mfcc=40, max_pad_len=200):
    y, sr = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Ensure consistent shape
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]

    return mfccs.T[np.newaxis, ...]


def predict_emotion(file_path):
    mfcc_features = extract_mfcc(file_path)
    prediction = model.predict(mfcc_features)
    predicted_label = np.argmax(prediction) + 1
    return emotion_mapping.get(predicted_label, "Unknown Emotion")


file_path = r"C:\Users\91797\Downloads\03-02-02-01-02-01-03.wav"
emotion = predict_emotion(file_path)
print(f"Predicted Emotion: {emotion}")
