import tensorflow as tf
from keras.layers import Conv2D, Dropout


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
    c = Dropout(drop_out)(c)  # Original 0.1
    return Conv2D(channel, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(c)