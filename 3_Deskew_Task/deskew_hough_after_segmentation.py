# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 22:44:20 2023

@author: Allwin
"""


import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import rotate
from skimage.transform import (hough_line, hough_line_peaks)
from scipy.stats import mode
from skimage import io
from skimage.filters import threshold_otsu, sobel
from matplotlib import cm
from skimage.io import imread,imshow
from skimage.transform import resize
import numpy as np
import cv2


def crop_with_mask(img,mask):
    
    
    # Find the bounding box of the mask
    x, y, w, h = cv2.boundingRect(mask)
    
    # Crop the image
    crop_img = img[y:y+h, x:x+w]
    
    # Resize the cropped image
    resized_img = cv2.resize(crop_img, (600, 600))
    
    # Create a new blank image with the desired size
    output_img = np.zeros((1200,1200, 3), np.uint8)
    
    # Paste the resized cropped image into the new image
    output_img[y:y+600, x:x+600] = resized_img
    
    # Save the output image
    cv2.imwrite('output.jpg', output_img)

#Binarize image based on threshhold
def binarizeImage(RGB_image):

  image = rgb2gray(RGB_image)
  threshold = threshold_otsu(image)
  bina_image = image < threshold
  return bina_image


def findEdges(bina_image):
  
  image_edges = sobel(bina_image)

  #plt.imshow(bina_image, cmap='gray')
  #plt.axis('off')
  #plt.title('Binary Image Edges')
  #plt.savefig('binary_image.png')
  return image_edges

def findTiltAngle(image_edges):
  h, theta, d = hough_line(image_edges)
  accum, angles, dists = hough_line_peaks(h, theta, d)
  angle = np.rad2deg(mode(angles)[0][0])
  
  if (angle < 0):
    
    r_angle = angle + 90
    
  else:
    
    r_angle = angle - 90

  # Plot Image and Lines    
  fig, ax = plt.subplots()
  

  #ax.imshow(image_edges, cmap='gray')

  origin = np.array((0, image_edges.shape[1]))

  for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):

    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    #ax.plot(origin, (y0, y1), '-r')

  #ax.set_xlim(origin)
  #ax.set_ylim((image_edges.shape[0], 0))
  #ax.set_axis_off()
  #ax.set_title('Detected lines')

  #plt.savefig('hough_lines.png')

  #plt.show()
    
  return r_angle
  
def rotateImage(RGB_image, angle):

  fixed_image = rotate(RGB_image, angle)

  #plt.imshow(fixed_image)
  #plt.axis('off')
  #plt.title('Fixed Image')
  #plt.savefig('fixed_image.png')
  #plt.show()

  return fixed_image

def generalPipeline_hough_transform(cropped_image):

  img =cropped_image

  img_h=512
  img_w=512
  img_c=3
  image = imread(img,as_gray=True)
  #plt.imshow(image)
  image=resize(image,(img_h,img_w,img_c),mode='constant',preserve_range=True)
  #plt.imshow(image)
  bina_image = binarizeImage(image)
  image_edges = findEdges(bina_image)
  angle = findTiltAngle(image_edges)
  fixed_image=rotateImage(io.imread(img), angle)
  plt.imshow(fixed_image)
  
# Load the image and mask
img = cv2.imread('fake_id_8__front.png')   # take any image from task 2
mask = cv2.imread('fake_id_8__front_seg.png', 0) #take corresponding mask

# function to crop image based on corresponding mask
crop_with_mask(img,mask)

#function for hough transform and rotation
generalPipeline_hough_transform('output.jpg')
