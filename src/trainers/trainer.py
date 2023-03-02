import tensorflow as tf

from src.trainers.models import standard_conv2d, multi_unet_model
from src.trainers.trainer_utils import callback_earlystop, BinaryIOULoss, IOU
from src.data_loaders import load_classification_ds, load_segmentation_ds, load_cleaning_ds
from src.utils import get_config

from functools import partial

CONFIG = CONFIG = get_config()


def classification_trainer(model_name,img_size,num_block,epochs=5,bs=32,data_augmentation=False,lr=0.001):
    
    print(f"{'*'*10} loading dataset for classification task {'*'*10}")
    train_ds, val_ds = load_classification_ds(bs=bs)

    print(f"{'*'*10} loading the standard_conv2d model {'*'*10}")
    model = standard_conv2d(img_size,data_augmentation,num_block=num_block,num_classes=1)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=['binary_accuracy']) #1 is white

    results = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
        callbacks=callback_earlystop(model_name)
        )

    score = results.history
    print(f"{'*'*10} Training completed with train accuracy {score['binary_accuracy'][-1]} and train loss {score['loss'][-1]} {'*'*10}")

    print(f"{'*'*10} Validation accuracy {score['val_binary_accuracy'][-1]} and validation loss {score['val_loss'][-1]} {'*'*10}")


def segmentation_trainer(model_name,img_size,ds_channels,drop_out=0.2,epochs=5,bs=32,lr=0.001):
    
    print(f"{'*'*10} loading dataset for segmentation task {'*'*10}")
    train_ds, val_ds = load_segmentation_ds(bs=bs,img_size=img_size,test=False)

    print(f"{'*'*10} loading the segmentation model {'*'*10}")
    model = multi_unet_model(num_classes=1, img_size=img_size,drop_out=drop_out,down_scaling_channels=ds_channels)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
                 loss='binary_crossentropy',
                 metrics=['binary_accuracy'])#,tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)])

    results = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
        callbacks=callback_earlystop(model_name)
        )

    score = results.history
    # print(f"{'*'*10} Training completed with train accuracy {score['binary_accuracy'][-1]}, iou {score['binary_io_u'][-1]} and train loss {score['loss'][-1]} {'*'*10}")

    # print(f"{'*'*10} Validation accuracy {score['val_binary_accuracy'][-1]}, iou {score['binary_io_u'][-1]} and validation loss {score['val_loss'][-1]} {'*'*10}")


def cleaner_trainer(model_name,img_size,ds_channels,drop_out=0.2,epochs=5,bs=32,lr=0.001,focal_gamma=2,class_1_weight=0.25):
    
    print(f"{'*'*10} loading dataset for cleaning task {'*'*10}")
    train_ds, val_ds = load_cleaning_ds(bs=bs,img_size=img_size,test=False)

    print(f"{'*'*10} loading the cleaning model {'*'*10}")
    model = multi_unet_model(num_classes=1, img_size=img_size,drop_out=drop_out,down_scaling_channels=ds_channels)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
                 loss=tf.losses.BinaryFocalCrossentropy(apply_class_balancing=True,gamma=focal_gamma,alpha=class_1_weight),
                 metrics=['binary_accuracy'])#,tf.keras.metrics.BinaryIoU(target_class_ids=[0], threshold=0.5)])

    results = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
        callbacks=callback_earlystop(model_name)
        )

    score = results.history



# optimizing the mse loss doesn't work. everything is black.


    # print(f"{'*'*10} Training completed with train accuracy {score['binary_accuracy'][-1]}, iou {score['binary_io_u'][-1]} and train loss {score['loss'][-1]} {'*'*10}")

    # print(f"{'*'*10} Validation accuracy {score['val_binary_accuracy'][-1]}, iou {score['binary_io_u'][-1]} and validation loss {score['val_loss'][-1]} {'*'*10}")

#     Epoch 11: val_loss improved from 0.04885 to 0.04429, saving model to models/cleaning/iou_max_channel_512_dout_0.1_e20_bs8_sz256_lr0.0001.h5
# 40/40 - 10s - loss: 0.0224 - binary_accuracy: 0.9904 - val_loss: 0.0443 - val_binary_accuracy: 0.9756 - 10s/epoch - 238ms/step
# Epoch 12/20

# Epoch 16: val_loss improved from 0.08983 to 0.08816, saving model to models/cleaning/iou_max_channel_512_dout_0.3_e20_bs8_sz256_lr0.0001.h5
# 40/40 - 10s - loss: 0.0162 - binary_accuracy: 0.9938 - val_loss: 0.0882 - val_binary_accuracy: 0.9554 - 10s/epoch - 244ms/


# Epoch 12: val_loss improved from 0.06462 to 0.06279, saving model to models/cleaning/iou_max_channel_256_dout_0.2_e40_bs8_sz256_lr0.0001.h5
# 40/40 - 9s - loss: 0.0273 - binary_accuracy: 0.9887 - val_loss: 0.0628 - val_binary_accuracy: 0.9679 - 9s/epoch - 232ms/step
# Epoch 13/40


#from logits, reduce the training and val loss little bit. But result image is not good


# Focal loss with gamma = 2

# Epoch 38: val_loss improved from 0.03128 to 0.03119, saving model to models/cleaning/iou_max_channel_256_dout_0.2_e40_bs8_sz256_lr1e-05.h5
# 40/40 - 10s - loss: 0.0132 - binary_accuracy: 0.9824 - val_loss: 0.0312 - val_binary_accuracy: 0.9635 - 10s/epoch - 238ms/step
# Epoch 39/40

## Focal loss with gamma = 2 and weight balance of 0.25 for class 1


# Epoch 27: val_loss improved from 0.01292 to 0.01273, saving model to models/cleaning/iou_max_channel_256_dout_0.2_e40_bs8_sz256_lr1e-05.h5

# Epoch 27: val_loss improved from 0.01291 to 0.01273, saving model to models/cleaning/iou_max_channel_gamma2_cw0.25_256_dout_0.2_e40_bs8_sz256_lr1e-05.h5
# 40/40 - 10s - loss: 0.0066 - binary_accuracy: 0.9759 - val_loss: 0.0127 - val_binary_accuracy: 0.9453 - 10s/epoch - 238ms/step
# Epoch 28/40


# Epoch 3: val_loss improved from 0.01208 to 0.01053, saving model to models/cleaning/iou_max_channel_gamma2_cw0.25_256_dout_0.2_e40_bs8_sz256_lr0.0001.h5
# 40/40 - 9s - loss: 0.0094 - binary_accuracy: 0.9727 - val_loss: 0.0105 - val_binary_accuracy: 0.9681 - 9s/epoch - 233ms/step
# Epoch 4/40
# not good

#trying lr 0.00005
# Epoch 5: val_loss improved from 0.01109 to 0.01073, saving model to models/cleaning/iou_max_channel_gamma2_cw0.25_256_dout_0.2_e40_bs8_sz256_lr5e-05.h5
# 40/40 - 9s - loss: 0.0087 - binary_accuracy: 0.9732 - val_loss: 0.0107 - val_binary_accuracy: 0.9644 - 9s/epoch - 233ms/step
# Epoch 6/40
# not much difference. I believe lr of 0.00001 works betters

#lets experiment with gamma value of 3
#lr 0.00001 --> model trains for more epochs
# Epoch 22: val_loss improved from 0.00643 to 0.00639, saving model to models/cleaning/iou_max_channel_gamma3_cw0.25_256_dout_0.2_e40_bs8_sz256_lr1e-05.h5
# 40/40 - 10s - loss: 0.0041 - binary_accuracy: 0.9701 - val_loss: 0.0064 - val_binary_accuracy: 0.9514 - 10s/epoch - 244ms/step
# Epoch 23/40
#model trainn for long time and loss is also less but image is not clear, its even messier.

#Trying BCE for lr 0.00001
# Epoch 40: val_loss improved from 0.14438 to 0.14330, saving model to models/cleaning/iou_max_channel_gamma0_cw0_256_dout_0.2_e40_bs8_sz256_lr1e-05.h5
# 40/40 - 9s - loss: 0.0505 - binary_accuracy: 0.9826 - val_loss: 0.1433 - val_binary_accuracy: 0.9682 - 9s/epoch - 236ms/step


#Trying BCE for lr 0.0001

# Epoch 12: val_loss improved from 0.06414 to 0.06302, saving model to models/cleaning/iou_max_channel_gamma0_cw0_256_dout_0.2_e40_bs8_sz256_lr0.0001.h5
# 40/40 - 10s - loss: 0.0271 - binary_accuracy: 0.9888 - val_loss: 0.0630 - val_binary_accuracy: 0.9675 - 10s/epoch - 243ms/step
# Epoch 13/40

## BCE loss with lr 0.0001 is best so far specially back side.

## try focal loss with gamma 2 and lr 0.0001
# Epoch 4: val_loss improved from 0.01059 to 0.01048, saving model to models/cleaning/iou_max_channel_gamma2_cw0.25_256_dout_0.2_e40_bs8_sz256_lr0.0001.h5
# 40/40 - 9s - loss: 0.0070 - binary_accuracy: 0.9772 - val_loss: 0.0105 - val_binary_accuracy: 0.9637 - 9s/epoch - 232ms/step
# Epoch 5/40
# Problem is with this setting model doesn't train for long time and image is also not that readable.

# gamma with 2 and lr 0.00001 try to recreate everything. signature also
# gamma with 2 and lr 0.00005 is not good.

# Now trying gamma of 1 and lr 0.0001
# Epoch 20: val_loss improved from 0.01714 to 0.01496, saving model to models/cleaning/iou_max_channel_gamma1_cw0.25_256_dout_0.2_e40_bs8_sz256_lr0.0001.h5
# 40/40 - 9s - loss: 0.0035 - binary_accuracy: 0.9926 - val_loss: 0.0150 - val_binary_accuracy: 0.9595 - 9s/epoch - 235ms/step
# Epoch 21/40


# gamma 0.5 and lr 0.0001

# Epoch 1: val_loss improved from inf to 0.06444, saving model to models/cleaning/iou_max_channel_gamma0.5_cw0.25_256_dout_0.2_e40_bs8_sz256_lr0.0001.h5
# 40/40 - 12s - loss: 0.0784 - binary_accuracy: 0.8359 - val_loss: 0.0644 - val_binary_accuracy: 0.9664 - 12s/epoch - 288ms/step
# Epoch 2/40 --> worst

# gamma 1 lr 0.0001 and alpha 0.6 meaning 60% of pixel is 
# Epoch 16: val_loss improved from 0.01433 to 0.01422, saving model to models/cleaning/iou_max_channel_gamma1_cw0.6_256_dout_0.2_e40_bs8_sz256_lr0.0001.h5
# 40/40 - 10s - loss: 0.0052 - binary_accuracy: 0.9902 - val_loss: 0.0142 - val_binary_accuracy: 0.9689 - 10s/epoch - 245ms/step
# Epoch 17/40



# gamma 1 lr 0.0001 and alpha 0.2 meaning 60% of pixel is 
# Epoch 4: val_loss improved from 0.01849 to 0.01769, saving model to models/cleaning/iou_max_channel_gamma1_cw0.2_256_dout_0.2_e40_bs8_sz256_lr0.0001.h5
# 40/40 - 9s - loss: 0.0126 - binary_accuracy: 0.9767 - val_loss: 0.0177 - val_binary_accuracy: 0.9561 - 9s/epoch - 232ms/step
# Epoch 5/40


#gamma lr 0001 apha 0.6 gamma2
# Epoch 15: val_loss improved from 0.00847 to 0.00847, saving model to models/cleaning/iou_max_channel_gamma2_cw0.6_256_dout_0.2_e40_bs8_sz256_lr0.0001.h5
# 40/40 - 9s - loss: 0.0031 - binary_accuracy: 0.9891 - val_loss: 0.0085 - val_binary_accuracy: 0.9687 - 9s/epoch - 236ms/step
# Epoch 16/40


# Epoch 3: val_loss improved from 0.00828 to 0.00778, saving model to models/cleaning/iou_max_channel_gamma2_cw0.1_256_dout_0.2_e40_bs8_sz256_lr0.0001.h5
# 40/40 - 9s - loss: 0.0060 - binary_accuracy: 0.9464 - val_loss: 0.0078 - val_binary_accuracy: 0.9494 - 9s/epoch - 233ms/step
# Epoch 4/40


# Epoch 15: val_loss improved from 0.01737 to 0.01697, saving model to models/cleaning/iou_max_channel_gamma1_cw0.3_256_dout_0.2_e40_bs8_sz256_lr0.0001.h5
# 40/40 - 10s - loss: 0.0039 - binary_accuracy: 0.9927 - val_loss: 0.0170 - val_binary_accuracy: 0.9592 - 10s/epoch - 239ms/step
# Epoch 16/40



# Epoch 14: val_loss improved from 0.01810 to 0.01774, saving model to models/cleaning/iou_max_channel_gamma1_cw0.4_256_dout_0.2_e40_bs8_sz256_lr0.0001.h5
# 40/40 - 9s - loss: 0.0048 - binary_accuracy: 0.9919 - val_loss: 0.0177 - val_binary_accuracy: 0.9613 - 9s/epoch - 235ms/step
# Epoch 15/40