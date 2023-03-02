import tensorflow as tf
from keras.layers import Conv2D, Dropout, BatchNormalization
from keras import backend as K


from skimage.color import rgb2gray
from skimage.transform import rotate
from skimage.transform import (hough_line, hough_line_peaks)
from skimage.filters import threshold_otsu, sobel
from scipy.stats import mode
import numpy as np


def callback_earlystop(checkpoint_path):
    checkpointer=tf.keras.callbacks.ModelCheckpoint(checkpoint_path,verbose=2,save_best_only=True)
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10,monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs'),
        checkpointer
        ]
    return callbacks


def get_unet_block(input, channel,kernel_size=3,drop_out=0.2):
    c = Conv2D(channel, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(input)
    if drop_out == 0:
        c = BatchNormalization()(c)
    else:
        c = Dropout(drop_out)(c)  # Original 0.1
    return Conv2D(channel, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(c)


def binarizeImage(RGB_image):
    """Binarize image based on threshhold"""
    image = rgb2gray(RGB_image)
    threshold = threshold_otsu(image)
    return image < threshold


def findEdges(bina_image):
   return sobel(bina_image)


def findTiltAngle(image_edges):
    h, theta, d = hough_line(image_edges)
    _, angles, _ = hough_line_peaks(h, theta, d)
    angle = np.rad2deg(mode(angles)[0][0])
    
    if (angle < 0):
       r_angle = angle + 90
    else:
       r_angle = angle - 90
    return r_angle


def rotateImage(RGB_image, angle):
    return rotate(RGB_image, angle)


def IOU(y_true,y_pred,smooth=1e-6,round_off=False):
    
    print(tf.shape(y_true))
    print(tf.shape(y_pred))
    y_true_ = tf.cast(K.flatten(y_true),tf.float32)
    y_pred_ = K.flatten(y_pred)
    if round_off:
        y_pred_ = tf.cast(y_pred_>=0.5, tf.float32)
    print(tf.shape(y_true_))
    print(tf.shape(y_pred_))
    intersection = K.sum(K.dot(y_true_,y_pred_))
    total = K.sum(y_true_) + K.sum(y_pred_)
    union = total-intersection
    return (intersection+smooth)/(union+smooth)

def BinaryIOULoss(y_true,y_pred):
    """Loss function for segmentation tasks
    reference: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
    """
    iou = IOU(y_true,y_pred,round_off=False)
    return -iou

    
