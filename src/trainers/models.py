from tensorflow import keras
from keras.models import Model
from keras.layers import Input, RandomBrightness,Resizing,Rescaling, RandomFlip, RandomRotation, Conv2D, MaxPooling2D, Dense, Flatten, concatenate, Conv2DTranspose

from src.trainers.trainer_utils import get_unet_block
from src.utils import get_config

CONFIG = get_config()
seed = CONFIG["Constants"]["seed"]


def standard_conv2d(h,w,data_augmentation,kernel_size=3,num_block=1,num_classes=2):
    """
    standard convolution architecture

    repeat_block: integer, architecture follow structure like conv --> maxpool.
                Use this parameter to increase the number of block. 
    
    """

    if kernel_size  not in (3,5):
        print("Kernel size not recommended.")
    if h % 2 != 0:
        print("input resolution not recommended")

    inp = Input(shape=(None,None,3))

    x = Resizing(h,w)(inp)
    x = Rescaling(1./255)(x)

    if data_augmentation:
        # x = RandomBrightness((-0.1,0.1),seed=seed)(x)
        x = RandomFlip("vertical",seed=seed)(x)
        x = RandomRotation(0.2,seed=seed)(x)
    

    channels = 32
    ks = 5
    for _ in range(num_block):
        x = Conv2D(channels,ks, padding="same",activation="relu")(x)
        x = MaxPooling2D(pool_size=2,strides=2)(x)
        channels = channels*2
        ks = kernel_size

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    out = Dense(num_classes, activation='sigmoid')(x)
    return Model(inputs=inp, outputs=out, name="StandardConv2D")


def multi_unet_model(num_classes, img_size, down_scaling_channels=[16,32,64,128,256], drop_out=0.2):
    #Build the model
    inputs = keras.Input(shape=img_size + (3,))
    c = Rescaling(1./255)(inputs)

    down_scaling_nodes = []
    for idx, channel in enumerate(down_scaling_channels):
        if idx+1 == len(down_scaling_channels):
            c = get_unet_block(input=c,channel=channel,drop_out=round(drop_out+0.1,2))
        else:
            c = get_unet_block(input=c,channel=channel,drop_out=drop_out)
            down_scaling_nodes.append(c)
            c = MaxPooling2D((2, 2))(c)

    down_scaling_nodes = reversed(down_scaling_nodes)
    up_scaling_channels = reversed(down_scaling_channels[:-1])
    u = c
    for channel,downscaling_node in zip(up_scaling_channels,down_scaling_nodes):
        u = Conv2DTranspose(channel, (2, 2), strides=(2, 2), padding='same')(u)
        u = concatenate([u, downscaling_node])
        u = get_unet_block(input=u,channel=channel)
     
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(u)
    return Model(inputs=[inputs], outputs=[outputs])