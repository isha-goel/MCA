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

  for i in array:
    ctr = 0
    audios = listdir('/content/drive/My Drive/MCA/ass2/Dataset/training/'+ i + '/')
    for a in audios:
      print(i,ctr)
      ctr+=1
      audio = '/content/drive/My Drive/MCA/ass2/Dataset/training/' + i + '/' + a
      samples, sampling_rate = librosa.load(audio, sr = None, mono = True, offset = 0.0, duration = None)
      spectrogram_found = np.array(spectrogram(samples, sampling_rate))

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
      samples, sampling_rate = librosa.load(audio, sr = None, mono = True, offset = 0.0, duration = None)
      spectrogram_found = np.array(spectrogram(samples, sampling_rate))

      if spectrogram_found.shape[1] != 99:
        spectrogram_found = np.concatenate((spectrogram_found, np.zeros((320, 99-spectrogram_found.shape[1]))), axis = 1)


      X_val.append(np.ravel(spectrogram_found))
      Y_val.append(i)

  
  print("Saving")

  out = '/content/drive/My Drive/MCA/ass2/spec_train_X_dft.pkl'
  outfile = open(out, 'wb')
  np.save(outfile, X_train)

  out = '/content/drive/My Drive/MCA/ass2/spec_train_Y_dft.pkl'
  outfile = open(out, 'wb')
  np.save(outfile, Y_train)

  out = '/content/drive/My Drive/MCA/ass2/spec_val_X_dft.pkl'
  outfile = open(out, 'wb')
  np.save(outfile, X_val)

  out = '/content/drive/My Drive/MCA/ass2/spec_val_Y_dft.pkl'
  outfile = open(out, 'wb')
  np.save(outfile, Y_val)

  print('Saved')

  print("Training")
  clf = svm.SVC(kernel='rbf')
  clf.fit(X_train, Y_train)

  print("Eval")

  y_true = Y_val
  y_pred = clf.predict(X_val)

  print(metrics.f1_score(y_true, y_pred, average='weighted'))
  print(metrics.f1_score(y_true, y_pred, average='macro'))
  print(metrics.f1_score(y_true, y_pred, average='micro'))
