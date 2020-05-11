'''Reference : https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html'''

import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
from os import listdir
from sklearn import svm
from sklearn import metrics

def get_mfcc(sample_rate, signal):
  emphasized_signal = numpy.append(signal[0], signal[1:] - 0.97 * signal[:-1])

  frame_size = 0.025
  frame_stride = 0.01
  frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate 
  signal_length = len(emphasized_signal)
  frame_length = int(round(frame_length))
  frame_step = int(round(frame_step))
  num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  

  pad_signal_length = num_frames * frame_step + frame_length
  z = numpy.zeros((pad_signal_length - signal_length))
  pad_signal = numpy.append(emphasized_signal, z) 

  indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
  frames = pad_signal[indices.astype(numpy.int32, copy=False)] 

  frames *= numpy.hamming(frame_length)

  NFFT = 512
  nfilt = 40
  num_ceps = 12
  cep_lifter = 22
  mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT)) 
  pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2)) 

  low_freq_mel = 0
  high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  
  mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  
  hz_points = (700 * (10**(mel_points / 2595) - 1)) 
  bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

  fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
  for m in range(1, nfilt + 1):
      f_m_minus = int(bin[m - 1])   
      f_m = int(bin[m])            
      f_m_plus = int(bin[m + 1])   

      for k in range(f_m_minus, f_m):
          fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
      for k in range(f_m, f_m_plus):
          fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
  filter_banks = numpy.dot(pow_frames, fbank.T)
  filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  
  filter_banks = 20 * numpy.log10(filter_banks) 

  mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
  
  (nframes, ncoeff) = mfcc.shape
  n = numpy.arange(ncoeff)
  lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
  mfcc *= lift  

  return mfcc

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
      ctr+=1
      #print(i,ctr)
      audio = '/content/drive/My Drive/MCA/ass2/Dataset/training/' + i + '/' + a

      rate, signal = scipy.io.wavfile.read(audio)

      l1 = len(noise_files)
      l2 = len(noise_signal)
      l3 = len(signal)
      
      index = random.randint(0, l1-1)
      noise = path + noise_files[index]
      noise_sample_rate, noise_signal = scipy.io.wavfile.read(noise)
      s = random.randint(0, l2-l3-1)
      noise_signal= noise_signal[s:s + l3]
      
      mfcc = get_mfcc(rate, signal + 0.0001*noise_signal)

      if mfcc.shape[0] != 98:
        mfcc = np.concatenate((mfcc, np.zeros((98-mfcc.shape[0], 12))), axis = 0)

      X_train.append(np.ravel(mfcc))
      Y_train.append(i)

  X_val = []
  Y_val = []

  for i in array:
    c = 0
    audios = listdir('/content/drive/My Drive/MCA/ass2/Dataset/validation/'+ i + '/')
    for a in audios:
      #print(i,c)
      audio = '/content/drive/My Drive/MCA/ass2/Dataset/validation/' + i + '/' + a
      sample_rate, signal = scipy.io.wavfile.read(audio)
      mfcc = get_mfcc(sample_rate, signal)

      if mfcc.shape[0] != 98:
        mfcc = np.concatenate((mfcc, np.zeros((98-mfcc.shape[0], 12))), axis = 0)

      X_val.append(np.ravel(mfcc))
      Y_val.append(i)


  print("Saving")

  out = '/content/drive/My Drive/MCA/ass2/noise_mfcc_train_X.pkl'
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
  filename = '/content/drive/My Drive/MCA/ass2/extra/SVM1_mfcc_noise.sav'
  pickle.dump(clf, open(filename, 'wb'))'''
