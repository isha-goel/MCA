''' Reference : https://towardsdatascience.com/understanding-audio-data-fourier-transform-fft-spectrogram-and-speech-recognition-a4072d228520'''

import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import librosa
from sklearn import svm
from sklearn import metrics
from math import ceil as ceil
from math import log as log
import cmath
import scipy
import random 
import scipy.io.wavfile
import pickle

def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def spectrogram(samples, sample_rate, stride_ms = 10.0, 
                          window_ms = 20.0, max_freq = None, eps = 1e-14):

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, 
                                          shape = nshape, strides = nstrides)
    
    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    weighting = np.hanning(window_size)[:, None]
    
    inp = windows * weighting
    fft = dft(inp)
    fft = np.absolute(fft)
    fft = fft**2
    
    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    
    max_freq = np.max(freqs)
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)
    return specgram

if __name__ == '__main__':

  array = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
  X_train = []
  Y_train = []

  path = '/content/drive/My Drive/MCA/ass2/Dataset/_background_noise_/'
  noise_files = listdir(path)

  for i in array:
    ctr = 0
    audios = listdir('/content/drive/My Drive/MCA/ass2/Dataset/training/'+ i + '/')
    for a in audios:
      print(i,ctr)
      ctr+=1
      audio = '/content/drive/My Drive/MCA/ass2/Dataset/training/' + i + '/' + a
      #samples, sampling_rate = librosa.load(audio, sr = None, mono = True, offset = 0.0, duration = None)
      rate, signal = scipy.io.wavfile.read(audio)
      #spectrogram_found = np.array(spectrogram(samples, sampling_rate))

      l1 = len(noise_files)
      l2 = len(noise_signal)
      l3 = len(signal)
      
      index = random.randint(0, l1-1)
      noise = path + noise_files[index]
      noise_sample_rate, noise_signal = scipy.io.wavfile.read(noise)
      s = random.randint(0, l2-l3-1)
      noise_signal= noise_signal[s:s + l3]

      spectrogram_found = np.array(spectrogram(signal + 0.0001*noise_signal, rate))

      if spectrogram_found.shape[1] != 99:
        spectrogram_found = np.concatenate((spectrogram_found, np.zeros((320, 99-spectrogram_found.shape[1]))), axis = 1)

      X_train.append(np.ravel(spectrogram_found))
      Y_train.append(i)

  X_val = []
  Y_val = []
  for i in array:
    c = 0
    audios = listdir('/content/drive/My Drive/MCA/ass2/Dataset/validation/'+ i + '/')
    for a in audios:
      print(i,c)
      c+=1
      audio = '/content/drive/My Drive/MCA/ass2/Dataset/validation/' + i + '/' + a
      rate, signal = scipy.io.wavfile.read(audio)
      #samples, sampling_rate = librosa.load(audio, sr = None, mono = True, offset = 0.0, duration = None)
      spectrogram_found = np.array(spectrogram(signal, rate))

      if spectrogram_found.shape[1] != 99:
        spectrogram_found = np.concatenate((spectrogram_found, np.zeros((320, 99-spectrogram_found.shape[1]))), axis = 1)


      X_val.append(np.ravel(spectrogram_found))
      Y_val.append(i)

  
  print("Saving")

  out = '/content/drive/My Drive/MCA/ass2/noise_spec_train_X_dft.pkl'
  outfile = open(out, 'wb')
  np.save(outfile, X_train)

  print('Saved')

  print("Training")
  clf = svm.SVC(kernel='rbf')
  clf.fit(X_train, Y_train)

  print("Eval")

  y_true = Y_val
  y_pred = clf.predict(X_val)

  print(metrics.f1_score(y_true, y_pred, average='weighted'))
  print(metrics.precision_score(y_true, y_pred, average='weighted'))
  print(metrics.recall_score(y_true, y_pred, average='weighted'))

  '''print("Saving model")
  filename = '/content/drive/My Drive/MCA/ass2/SVM1_spec_noise.sav'
  pickle.dump(clf, open(filename, 'wb'))'''
