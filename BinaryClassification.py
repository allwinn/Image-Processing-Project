# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 15:17:44 2022

@author: Allwin
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

base_dir='1_classification'

#Folder with ids and nonid images
train_dir=os.path.join(base_dir,'Train')
test_dir=os.path.join(base_dir,'Test')
#id images for training

train_ids_yes=os.path.join(train_dir,'Ids')
train_ids_no=os.path.join(train_dir,'Non_Ids')

test_ids_yes=os.path.join(test_dir,'Ids')
test_ids_no=os.path.join(test_dir,'Non_Ids')

train_datagen = ImageDataGenerator(rescale = 1.0/255,validation_split=0.2)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150),
                                                    subset='training'
                                                    )
validation_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150),
                                                    subset='validation'
                                                    )
model = tf.keras.models.Sequential([
    # since Conv2D is the first layer of the neural network, we should also specify the size of the input
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    # apply pooling
    tf.keras.layers.MaxPooling2D(2,2),
    # and repeat the process
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    # flatten the result to feed it to the dense layer
    tf.keras.layers.Flatten(), 
    # and define 512 neurons for processing the output coming by the previous layers
    tf.keras.layers.Dense(512, activation='relu'), 
    # a single output neuron. The result will be 0 if the image is a cat, 1 if it is a dog
    tf.keras.layers.Dense(1, activation='sigmoid')  
])


model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics = ['accuracy'])


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']



history = model.fit(
            train_generator, # pass in the training generator
            steps_per_epoch=10,
            epochs=10,
            validation_data=validation_generator, # pass in the validation generator
            validation_steps=5,
            verbose=2
            )


model.predict(train_dir)
model.save_weights("modelv1.h5")
[layer.output for layer in model.layers]


import pandas as pd
#Displaying output
test_filenames = os.listdir(test_ids_yes)
test_df = pd.DataFrame({
    'filename': test_filenames[0]+test_filenames[1]
})
nb_samples = test_df.shape[0]

test_datagen = ImageDataGenerator(rescale = 1.0/255)
test_generator = train_datagen.flow_from_directory(test_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150)
                                                
                                                    )

predict = model.predict(test_generator, steps=np.ceil(nb_samples/20))

test_df['id_present'] = np.argmax(predict, axis=-1)



test_filenames=[]
path = [test_ids_yes, test_ids_no]
for i in path:
    test_filenames.append(os.listdir(i))
    
    for filename in os.listdir(i):
        with open(os.path.join(i, filename), 'r') as filedata:
            string = "".join(filedata.read().split())

os.listdir(test_ids_yes)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['acc'], color='b', label="Training accuracy")
ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()