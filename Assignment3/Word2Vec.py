from nltk.corpus import abc
from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
from keras.layers import dot
from keras.layers import Dot
from keras import backend as K
from keras.layers import Lambda

import collections
import os
import zipfile

import numpy as np
import tensorflow as tf

def build_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def collect_data(vocabulary_size=10000):
    v1 = abc.raw("rural.txt").split()
    v2 =  abc.raw("science.txt").split()
    vocabulary = v1 + v2
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
    del vocabulary  
    return data, count, dictionary, reverse_dictionary

class SimilarityCallback:

    def run_sim(self):
      check_words = ['Iraq', 'wheat', 'letters', 'government', 'UNK', 'Australia', 'Federal', 'AWB', 'Western', 'million']
      for i in range(len(check_words)):
          valid_word = check_words[i]
          top_k = 12 
          sim = self.get_sim(dictionary[check_words[i]])
          nearest = (-sim).argsort()[1:top_k + 1]
          log_str = 'Nearest to %s:' % valid_word
          for k in range(top_k):
              close_word = reverse_dictionary[nearest[k]]
              log_str = '%s %s,' % (log_str, close_word)
          print(log_str)

    @staticmethod
    def get_sim(valid_word_idx):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = valid_word_idx
        for i in range(vocab_size):
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim

if __name__ == "__main__":

  vocab_size = 20000
  window_size = 3
  vector_dim = 300
  epochs = 30000
  valid_size = 16   
  valid_window = 100 

  data, count, dictionary, reverse_dictionary = collect_data(vocabulary_size=vocab_size)
  
  valid_examples = np.random.choice(valid_window, valid_size, replace=False)

  sampling_table = sequence.make_sampling_table(vocab_size)
  couples, labels = skipgrams(data, vocab_size, window_size=window_size, sampling_table=sampling_table)
  word_target, word_context = zip(*couples)
  word_target = np.array(word_target, dtype="int32")
  word_context = np.array(word_context, dtype="int32")

  input_target = Input((1,))
  input_context = Input((1,))

  embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
  target = embedding(input_target)
  target = Reshape((vector_dim, 1))(target)
  context = embedding(input_context)
  context = Reshape((vector_dim, 1))(context)

  similarity = dot([target, context], axes=1, normalize=True)

  dot_product = Dot(axes = 1)([target, context])
  dot_product = Reshape((1,))(dot_product)

  output = Dense(1, activation='sigmoid')(dot_product)
  model = Model(input=[input_target, input_context], output=output)
  model.compile(loss='binary_crossentropy', optimizer='rmsprop')
  validation_model = Model(input=[input_target, input_context], output=similarity)

  sim_cb = SimilarityCallback()

  arr_1 = np.zeros((1,))
  arr_2 = np.zeros((1,))
  arr_3 = np.zeros((1,))
  for cnt in range(epochs):
    idx = np.random.randint(0, len(labels)-1)
    arr_1[0,] = word_target[idx]
    arr_2[0,] = word_context[idx]
    arr_3[0,] = labels[idx]
    loss = model.train_on_batch([arr_1, arr_2], arr_3)
    if cnt % 100 == 0:
        print("Iteration {}, loss={}".format(cnt, loss))
    if cnt % 10000 == 0:
        sim_cb.run_sim()
