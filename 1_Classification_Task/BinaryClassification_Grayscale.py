
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import pandas as pd

#from keras.models import load_model
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
                                                    color_mode="grayscale",
                                                    batch_size=64,
                                                    class_mode='binary',
                                                    target_size=(150, 150),
                                                    subset='training'
                                                    )
validation_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=10,
                                                    color_mode="grayscale",
                                                    class_mode='binary',
                                                    target_size=(150, 150),
                                                    subset='validation'
                                                    )

model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dense(1, activation='sigmoid')  
])


model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics = ['accuracy'])


history = model.fit(
            train_generator, # pass in the training generator
            epochs=10,
            steps_per_epoch=10,
            validation_data=validation_generator, # pass in the validation generator
            #validation_steps=5,
            verbose=2
            )


model.save_weights("modelv2_grayscale_4conv.h5")


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


#Displaying output
test_filenames=[]
path = [test_ids_yes, test_ids_no]

for i in path:
    test_filenames.append(os.listdir(i))
    
    
test_df = pd.DataFrame({
    'filename': test_filenames[0]+test_filenames[1]})
nb_samples = test_df.shape[0]

test_datagen = ImageDataGenerator(rescale = 1.0/255)
test_generator = train_datagen.flow_from_directory(test_dir,
                                                    color_mode="grayscale",
                                                    class_mode=None,
                                                    target_size=(150, 150),
                                                    shuffle=False
                                                    )


model.load_weights('modelv2_grayscale_4conv.h5')
#new_model=tf.keras.models.load_model('modelv1.h5')
predict = model.predict(test_generator)#, steps=np.ceil(nb_samples/20))

test_df['id_present'] = [0 if x<=0.75 else 1 for x in predict]
test_df['id_present'] = [x for x in predict]
test_df.to_csv("labels_test.csv")


#Visualization
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc)) 

# plot accuracy with matplotlib
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Accuracy in training and validation')
plt.figure()

# plot loss with matplotlib
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Loss in training and validation')

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('./1_classification/Test/Ids/fake_id_216__front.png')
plt.imshow(img)


