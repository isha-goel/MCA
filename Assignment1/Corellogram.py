import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
from google.colab.patches import cv2_imshow

def get_neighbours(x,y,length,height,d):

  n1 = (x + d, y + d)
  n2 = (x + d, y)
  n3 = (x + d, y - d)
  n4 = (x, y - d)
  n5 = (x - d, y - d)
  n6 = (x - d, y)
  n7 = (x - d, y + d)
  n8 = (x, y + d)
  points = (n1, n2, n3, n4, n5, n6, n7, n8)
  neighbours = []
  for point in points:
    if(0 <= point[0] and point[0] < length and 0 <= point[1] and point[1] < height):
      neighbours.append(point)
  
  return neighbours

def get_corellogram(image):
  distance = [1,3,5,7]
  correlogram = np.zeros((64,4))
  length = image.shape[0];
  height = image.shape[1];

  for dnum, d in enumerate(distance):
    count = 0
    color_ctr = np.zeros((64,1))
    for x in range(0, length, int(round(length / 10))):
      for y in range(0, height, int(round(height / 10))):
        c = image[x][y]

        neighbours = get_neighbours(x,y,length,height,d)

        for neighbour in neighbours:
          c_ = image[neighbour[0]][neighbour[1]]
          if(c == c_):
            count+=1
            color_ctr[c]+=1

    for i in range(len(color_ctr)):
      color_ctr[i] /= count
    corellogram[:,dnum] = color_ctr.reshape(1,-1)
  return corellogram

def quantize(pixel):
  if(pixel<64):
    return 0
  if(pixel<128):
    return 1
  if(pixel<192):
    return 2
  if(pixel<256):
    return 3

if __name__ == '__main__':

  images = listdir('/content/drive/My Drive/MCA/HW-1/resized_images/')
  #loaded_images = list()
  path = '/content/drive/My Drive/MCA/HW-1/corellograms/'
  for img in images:
    i = '/content/drive/My Drive/MCA/HW-1/resized_images/' + img
    image = cv2.imread(i, 1)
    image = np.asarray(image)
    image1 = image
    image_2D = np.zeros((256,256))
    for k in range(image1.shape[0]):
      for j in range(image1.shape[1]):
        for l in range(image1.shape[2]):
          image1[k][j][l] = quantize(image1[k][j][l])
        image_2D[k][j] = image1[k][j][0]*16 + image1[k][j][1]*4 + image1[k][j][2]
    
    image_2D = image_2D.astype('int')
    corellogram = get_corellogram(image_2D) 
    #cv2_imshow(image1)
    print(corellogram)
    #pickle_file = path+img[:img.find('.')]+'.pkl'
    #outfile = open(pickle_file, 'wb')

    #np.save(outfile, corellogram)
