#### Dependencies ####

import IPython.display as ipd
import joblib
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from glob import glob
from sklearn.model_selection import train_test_split
import json

filepath1 = "/Users/norbulama/Desktop/CS-4000/audios/lie/"
filepath2 = "/Users/norbulama/Desktop/CS-4000/audios/truth"

audiopath1 = glob(filepath1 + "/*.wav")
audiopath2 = glob(filepath2 + "/*.wav")


def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)

    return mfccs_processed


features = []

# Iterate through each sound file and extract the features
for i in audiopath1:
    data = extract_features(i)
    features.append([data, "lie"])
for i in audiopath2:
    data = extract_features(i)
    features.append([data, "truth"])


# Convert into a Panda dataframe
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
featuresdf.head()


X = np.array(featuresdf['feature'].tolist())
y = np.array(featuresdf['class_label'].tolist())


# Train test saplit
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=4)


from sklearn import svm

classifier = svm.SVC(kernel = 'linear', gamma='auto', C=2)
classifier.fit(X_train, y_train)

filename = 'classifier_model.pkl'
joblib.dump(classifier, filename)

print(classifier.fit(X_train, y_train))

y_predict = classifier.predict(X_test)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))