import tensorflow as tf

from src.trainers.models import standard_conv2d, multi_unet_model
from src.trainers.trainer_utils import callback_earlystop
from src.data_loaders import load_classification_ds, load_segmentation_ds
from src.utils import get_config

CONFIG = CONFIG = get_config()
CONSTANTS = CONFIG["Constants"]


def classification_trainer(num_block,epochs=5,bs=32):
    
    print(f"{'*'*10} loading dataset for classification task {'*'*10}")
    train_ds, val_ds = load_classification_ds(bs=bs)

    w = CONSTANTS["classification"]["img_size"]["width"]
    h = CONSTANTS["classification"]["img_size"]["height"]

    print(f"{'*'*10} loading the standard_conv2d model {'*'*10}")
    model = standard_conv2d(h,w,num_block=num_block)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    results = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
        callbacks=callback_earlystop(f"models/classification_e{epochs}_bs{bs}.h5")
        )

    score = results.history
    print(f"{'*'*10} Training completed with train accuracy {score['accuracy'][-1]} and train loss {score['loss'][-1]} {'*'*10}")

    print(f"{'*'*10} Validation accuracy {score['val_accuracy'][-1]} and validation loss {score['val_loss'][-1]} {'*'*10}")


def segmentation_trainer(epochs=5,bs=32):
    
    print(f"{'*'*10} loading dataset for segmentation task {'*'*10}")

    w = CONSTANTS["segmentation"]["img_size"]["width"]
    h = CONSTANTS["segmentation"]["img_size"]["height"]
    img_size = (h,w)
    train_ds, val_ds = load_segmentation_ds(bs=bs,img_size=img_size,test=False)

    print(f"{'*'*10} loading the segmentation model {'*'*10}")
    model = multi_unet_model(num_classes=1, img_size=img_size)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

    results = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
        callbacks=callback_earlystop(f"models/segmentation_e{epochs}_bs{bs}.h5")
        )

    score = results.history
    print(f"{'*'*10} Training completed with train accuracy {score['accuracy'][-1]} and train loss {score['loss'][-1]} {'*'*10}")

    print(f"{'*'*10} Validation accuracy {score['val_accuracy'][-1]} and validation loss {score['val_loss'][-1]} {'*'*10}")