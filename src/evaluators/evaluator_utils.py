import cv2
import numpy as np


def crop_with_mask(img_path,mask_path,img_size):
    img = cv2.imread(img_path)   # take any image from task 2
    img = cv2.resize(img,img_size)
    mask = cv2.imread(mask_path,0) #take corresponding mask

    
    # Find the bounding box of the mask
    x, y, w, h = cv2.boundingRect(mask)
    
    # Crop the image
    crop_img = img[y:y+h, x:x+w]
    
    # Resize the cropped image
    resized_img = cv2.resize(crop_img, (600, 600))
    
    return resized_img
    