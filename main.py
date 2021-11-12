#### Dependencies ####

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from glob import glob
import joblib
import sounddevice as sd
from scipy.io.wavfile import write

filepath1 = "/Users/norbulama/Desktop/CS-4000/audios/lie/"
filepath2 = "/Users/norbulama/Desktop/CS-4000/audios/truth"

audiopath1 = glob(filepath1 + "/*.wav")
audiopath2 = glob(filepath2 + "/*.wav")

fs = 48000  # Sample rate
seconds = 5  # Duration of recording
print("Record the first sample")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('/Users/norbulama/Desktop/CS-4000/TruthOrLieAutomation/userInput/output1.wav', fs, myrecording)  # Save as WAV file

print("Record the second sample")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('/Users/norbulama/Desktop/CS-4000/TruthOrLieAutomation/userInput/output2.wav', fs, myrecording)  # Save as WAV file

print("Record the third sample")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('/Users/norbulama/Desktop/CS-4000/TruthOrLieAutomation/userInput/output3.wav', fs, myrecording)  # Save as WAV file

userInput_path = '/Users/norbulama/Desktop/CS-4000/TruthOrLieAutomation/userInput/'
useraudiopath1 = glob(userInput_path + "/*.wav")
user_Features = []

for i in audiopath1:
    audio, sample_rate = librosa.load(i, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    plt.figure(figsize=(10, 5))
    # librosa.display.specshow(mfccs,
    #                          x_axis="time",
    #                          sr=sample_rate)
    # plt.colorbar(format="%+2f")
    # plt.show()
    mfccs_processed = np.mean(mfccs.T, axis=0)
    user_Features.append(mfccs_processed)
print(user_Features)

model = joblib.load('classifier_model.pkl')

y_predict = model.predict(user_Features)

print(y_predict)