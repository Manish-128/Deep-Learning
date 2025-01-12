import numpy as np 
import pandas as pd
import librosa
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spectral_contrast, axis=1)
    ])

fold_path = r"data"
labels = ['sad', 'happy']

X = []
y = []

for label in labels:
    modified_path = os.path.join(fold_path, label)
    print(modified_path)
    for file in os.listdir(modified_path):
        print(file)
        file_path = os.path.join(modified_path, file)
        features = extract_features(file_path)
        X.append(features)
        y.append(label)
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

test_path = r"P:\My Work\Music\Indian_Idol_Main\Test\original\aaj_jane_ki_zidd_na_karo.wav"

featured_file = extract_features(test_path)
featured_file = featured_file.reshape(1,-1)
predicted = model.predict(featured_file)
print(predicted,"    ", test_path)

test_path = r"C:\Users\manis\Downloads\Morni Banke.mp3"

featured_file = extract_features(test_path)
featured_file = featured_file.reshape(1,-1)
predicted = model.predict(featured_file)
print(predicted,"    ", test_path)

joblib.dump(model, 'happy_sad.pkl')