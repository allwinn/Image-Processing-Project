# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 12:13:13 2023

@author: Allwin
"""

import tensorflow as tf
import os 
from tqdm import tqdm
import numpy as np
import random

from skimage.io import imread,imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

train_path_ids='Train/Ids/'
train_path_gt='Train/GroundTruth/'
test_path_ids='Test/Ids/'
test_path_gt='Test/GroundTruth/'

img_h=128
img_w=128
img_c=3

seed=42
np.random.seed=seed
train_ids=next(os.walk(train_path_ids))[2]
train_gt=next(os.walk(train_path_gt))[2]
test_ids=next(os.walk(test_path))[2]

X_train=np.zeros((len(train_ids),img_h,img_w,img_c),dtype=np.uint8)
Y_train=np.zeros((len(train_ids),img_h,img_w,1),dtype=bool)
print("resizing training images")
for n,id_ in tqdm(enumerate(train_ids),total=len(train_ids)):
    path=train_path_ids+id_
    img=imread(path)[:,:,:img_c]
    img=resize(img,(img_h,img_w),mode='constant',preserve_range=True)
    X_train[n]=img
    
    
"""
path=train_path_gt+id_
mask_=imread('Train/GroundTruth/fake_id_101_back_seg.png',as_gray=True)
mask=np.expand_dims(resize(mask_,(img_h,img_w),mode='constant',preserve_range=True),axis=-1)
mask=resize(mask_,(img_h,img_w),mode='constant',preserve_range=True)
Y_train[0]=mask
"""    
for n,id_ in tqdm(enumerate(train_gt),total=len(train_gt)):
    path=train_path_gt+id_
    #mask=np.zeros((img_h,img_w,1),dtype=bool)
    mask_=imread(path,as_gray=True)
    mask_=np.expand_dims(resize(mask_,(img_h,img_w),mode='constant',preserve_range=True),axis=-1)
    Y_train[n]=mask_
    #print('added image',n,'with path',path)
    
from tensorflow.keras import layers

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras import backend as K
from tensorflow import keras
    
def multi_unet_model(num_classes, img_size):
#Build the model
    inputs = keras.Input(shape=img_size + (3,))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.2)(c1)  # Original 0.1
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)  # Original 0.1
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)  # Original 0.1
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)  # Original 0.1
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    #model.summary()
    
    return model

num_classes=1
img_size = (128, 128)
model = multi_unet_model(num_classes,img_size)
model.summary()

checkpointer=tf.keras.callbacks.ModelCheckpoint('model_segmentation_scratch.h5',verbose=1,save_best_only=True)

callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=10,monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs'),checkpointer]

results=model.fit(X_train,Y_train,validation_split=0.1,batch_size=16,epochs=25,callbacks=callbacks)


#Check results

#resizing test images
X_test=np.zeros((len(test_ids),img_h,img_w,img_c),dtype=np.uint8)
for n,id_ in tqdm(enumerate(test_ids),total=len(test_ids)):
    path=test_path_ids+id_
    img=imread(path)[:,:,:img_c]
    img=resize(img,(img_h,img_w),mode='constant',preserve_range=True)
    X_test[n]=img
    


preds_train=model.predict(X_train[:int(X_train.shape[0]*0.9)],verbose=1)
preds_val=model.predict(X_train[int(X_train.shape[0]*0.9):],verbose=1)
preds_test=model.predict(X_test,verbose=1)

preds_train_t=(preds_train>0.5).astype(np.uint8)
preds_val_t=(preds_val>0.5).astype(np.uint8)
preds_test_t=(preds_test>0.5).astype(np.uint8)

ix=random.randint(0,len(preds_train_t))
fig = plt.figure(figsize=(10, 7))
rows = 2
columns = 2
fig.add_subplot(rows, columns, 1)
plt.imshow(X_train[ix])
plt.axis('off')
plt.title("First")

fig.add_subplot(rows, columns, 2)
plt.imshow(np.squeeze(Y_train[ix]))
plt.axis('off')
plt.title("Second")

plt.imshow(Image3)
plt.imshow(np.squeeze(preds_train[ix]))
plt.axis('off')
plt.title("Third")

plt.imshow(Image4)
plt.imshow(np.squeeze(preds_train_t[ix]))
plt.axis('off')
plt.title("Fourth")
