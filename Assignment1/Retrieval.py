import cv2
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import filters
from scipy import spatial
from os import listdir
from operator import itemgetter
import datetime

def check_ground_truth(distances, query):
  files = listdir("/content/drive/My Drive/MCA/HW-1/train/ground_truth/")
  ctr = 0
  good = ""
  ok = ""
  junk = ""
  for f in files:
    if(f[:f.rfind("_")] == query[:query.rfind("_")]):
      if(f[f.rfind("_")+1:f.find(".")] == 'good'):
        good = "/content/drive/My Drive/MCA/HW-1/train/ground_truth/"+ f
        ctr+=1
      if(f[f.rfind("_")+1:f.find(".")] == 'ok'):
        ok = "/content/drive/My Drive/MCA/HW-1/train/ground_truth/" + f
        ctr+=1
      if(f[f.rfind("_")+1:f.find(".")] == 'junk'):
        junk = "/content/drive/My Drive/MCA/HW-1/train/ground_truth/" + f
        ctr+=1
    if(ctr == 3):
      break
  #print(good,ok,junk)
  good_ctr = 0
  ok_ctr = 0
  junk_ctr = 0
  len_good = 0
  len_ok = 0
  len_junk = 0
  with open(good,'r') as f:
    for line in f:
      len_good+=1
      for d in distances:
        if(d[1] == line[:-1]):
          good_ctr+=1

  with open(ok,'r') as f:
    for line in f:
      len_ok+=1
      for d in distances:
        if(d[1] == line[:-1]):
          #print('hi')
          ok_ctr+=1

  with open(junk,'r') as f:
    for line in f:
      len_junk+=1
      for d in distances:
        if(d[1] == line[:-1]):
          #print('hi2')
          junk_ctr+=1

  Precision = (good_ctr + ok_ctr + junk_ctr)/len(distances)
  Recall = (good_ctr + ok_ctr + junk_ctr)/(len_good + len_ok + len_junk)
  if(Precision+Recall != 0):
    F1 = (2*Precision*Recall)/(Precision + Recall)
  else:
    F1 = 0
  print("Good: " + str(good_ctr) + " Ok: " + str(ok_ctr) + " Junk: " + str(junk_ctr))
  print("Precision is "+ str(Precision) + ", Recall is "+ str(Recall) + ", F1-Score is "+ str(F1))


if __name__ == '__main__':
  all_queries = listdir("/content/drive/My Drive/MCA/HW-1/train/query/")
  for query in all_queries:
    print(query)
    t1 = datetime.datetime.now()
    q = "/content/drive/My Drive/MCA/HW-1/train/query/" + query
    with open(q, 'r') as file:
      for image in file:
        index1 = image.find('_') + 1
        index2 = image.find(' ')
        image_to_be_matched = image[index1:index2]
        #print(image_to_be_matched)
        similarity_matching(image_to_be_matched, query)
        t2 = datetime.datetime.now()
        t = t2 - t1
        print("The time taken for the query to execute is: " + str(t))

from os import listdir
from operator import itemgetter
import numpy as np

def load_corellograms():
  all_corellogram_files = listdir("/content/drive/My Drive/MCA/HW-1/corellograms_new/")
  ctr = 0
  corellograms = list()
  for corellogram_file in all_corellogram_files:
    ctr+=1
    corellogram = "/content/drive/My Drive/MCA/HW-1/corellograms_new/" + corellogram_file
    with open(corellogram, 'rb') as file:
      c = np.load(file)
    corellograms.append((c, corellogram_file))
    print(ctr)
  return corellograms

c = np.load("/content/drive/My Drive/MCA/HW-1/corellogram_load.pkl", allow_pickle = 'TRUE')

'''c = load_corellograms() 
corellogram_load_file = "/content/drive/My Drive/MCA/HW-1/corellogram_load.pkl" 
outfile = open(corellogram_load_file, 'wb')
np.save(outfile, c)'''


def similarity_matching(image, query):
  pickle_file = "/content/drive/My Drive/MCA/HW-1/corellograms_new/" + image + ".pkl"
  corellogram_to_be_matched = []
  with open(pickle_file, 'rb') as file:
    corellogram_to_be_matched = np.load(file) 
  
  all_distances = list()
  for corellogram in c:
    if(corellogram[1] == image + '.pkl'):
      continue
    
    sum_distance = 0
    for i in range(corellogram[0].shape[0]):
      for j in range(corellogram[0].shape[1]):
        sum_distance += (np.absolute(corellogram[0][i][j] - corellogram_to_be_matched[i][j]))/(1 + corellogram[0][i][j] + corellogram_to_be_matched[i][j])
    all_distances.append((sum_distance, corellogram[1][:corellogram[1].find(".")]))
  distances = sorted(all_distances, key = itemgetter(0))
  distances = distances[:100]
  check_ground_truth(distances, query)
