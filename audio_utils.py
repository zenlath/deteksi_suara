import librosa
import numpy as np
from keras.models import load_model

# Load model sekali saja
model = load_model('model.h5')

labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def extract_features(file_path, sr=22050, max_pad_len=2376):
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)

        # Ambil hanya 1 MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=1)

        # Padding atau potong agar panjang waktu = 2376
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

        # Transpose ke (2376, 1)
        features = mfcc.T  # (timesteps, 1)

        # Tambahkan batch dimensi → (1, 2376, 1)
        features = np.expand_dims(features, axis=0)

        print("Shape input ke model:", features.shape)  # Debug

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def predict_emotion(file_path):
    features = extract_features(file_path)
    if features is None:
        return None, None

    prediction = model.predict(features)[0]
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    predicted_label = labels[predicted_index]
    return predicted_label, confidence
