import cv2
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import filters
from scipy import spatial
from os import listdir
from operator import itemgetter

def get_pixel_values(co_ordinates, image):
  pixels = list()
  for blob in co_ordinates:
    y,x,r,result = blob
    pixels_of_blob = list()
    if(x>=0 and x<=255 and y>=0 and y<=255):
      pixels_of_blob.append(image[x][y])
    if(x>=0 and x<=255 and y-1>=0 and y-1<=255):
      pixels_of_blob.append(image[x][y-1])
    if(x-1>=0 and x-1<=255 and y+1>=0 and y+1<=255):
      pixels_of_blob.append(image[x-1][y+1])
    if(x+1>=0 and x+1<=255 and y-1>=0 and y-1<=255):
      pixels_of_blob.append(image[x+1][y-1])
    if(x>=0 and x<=255 and y+1>=0 and y+1<=255):
      pixels_of_blob.append(image[x][y+1])
    if(x-1>=0 and x-1<=255 and y>=0 and y<=255):
      pixels_of_blob.append(image[x-1][y])
    if(x+1>=0 and x+1<=255 and y>=0 and y<=255):
      pixels_of_blob.append(image[x+1][y])
    if(x+1>=0 and x+1<=255 and y+1>=0 and y+1<=255):
      pixels_of_blob.append(image[x+1][y+1])
    if(x-1>=0 and x-1<=255 and y-1>=0 and y-1<=255):
      pixels_of_blob.append(image[x-1][y-1])

    '''if(x-r > 0):
      x1 = int(x-r)
    else:
      x1 = 0

    if(x+r < 255):
      x2 = int(x+r)
    else:
      x2 = 255
    
    if(y-r > 0):
      y1 = int(y-r)
    else:
      y1 = 0

    if(y+r < 255):
      y2 = int(y+r)
    else:
      y2 = 255
    
    for x_ in range(x1, x2+1):
      for y_ in range(y1, y2+1):
        if(np.power((x - x_),2) + np.power((y - y_),2) <= np.power(r,2)):
          pixels_of_blob.append(image[x_][y_])'''

    pixels.append(pixels_of_blob)
  return pixels

def LoG(sigma): 
    n = np.ceil(sigma*6)
    y = np.zeros((int(n+1),1))
    x = np.zeros((1,int(n+1)))
    a = -n//2
    b= -n//2
    for i in range(y.shape[0]):
      for j in range(y.shape[1]):
        y[i][j] = a
        a+=1
    for i in range(x.shape[0]):
      for j in range(x.shape[1]):
        x[i][j] = b
        b+=1
    final_filter = (-(2*np.power(sigma,2)) + (x*x + y*y) ) *  (np.exp(-(x*x/(2.*np.power(sigma,2))))* np.exp(-(y*y/(2.*np.power(sigma,2))))) * (1/(2*np.pi*np.power(sigma,4)))
    return final_filter
  
def LoG_convolve(img, k, sigma):
    LoG_images = [] 
    for i in range(0,9):
        y = k**i 
        sigma_ = sigma*y  
        LoG_filter = LoG(sigma_) 
        image = cv2.filter2D(img,-1,LoG_filter) 
        image = np.square(np.pad(image,((1,1),(1,1)),'constant')) 
        LoG_images.append(image)
    log_image = np.zeros((9,258,258))
    for ind, image1 in enumerate(LoG_images):
      log_image[ind, :, :] = image1
    return log_image


def detect_blob(log_image, img):
    co_ordinates = [] 
    height = img.shape[0]
    width = img.shape[1]
    for i in range(1,height):
        for j in range(1,width):
            slice_img = log_image[:,i-1:i+2,j-1:j+2] 
            result = np.amax(slice_img)
            if result >= 0.025: 
                z,x,y = np.unravel_index(slice_img.argmax(),slice_img.shape)
                co_ordinates.append((i+x-1,j+y-1,k**z*sigma, result)) 
    #coordinates = sorted(co_ordinates, reverse = True, key = itemgetter(3))
    #coordinates = coordinates[:750]
    return co_ordinates

if __name__ == '__main__':
  k = 1.414
  sigma = 1.0

  path = '/content/drive/My Drive/MCA/HW-1/blobs_mine/'
  images = listdir("/content/drive/My Drive/MCA/HW-1/resized_images/")
  for n, image in enumerate(images):
    print(n+1)
    i = "/content/drive/My Drive/MCA/HW-1/resized_images/" + image  
    img = cv2.imread(i,0) 
    img = img/255.0  
    log_image = LoG_convolve(img, k, sigma)
    co_ordinates = list(set(detect_blob(log_image, img)))
    #pixels = get_pixel_values(co_ordinates, img)
    #print(pixels)
    #break

    pickle_file = path+image[:image.find('.')]+'.pkl'
    outfile = open(pickle_file, 'wb')

    np.save(outfile, co_ordinates)


