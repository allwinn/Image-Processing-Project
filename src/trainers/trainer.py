import tensorflow as tf

from src.trainers.models import standard_conv2d, multi_unet_model
from src.trainers.trainer_utils import callback_earlystop, BinaryIOULoss, IOU
from src.data_loaders import load_classification_ds, load_segmentation_ds, load_cleaning_ds
from src.utils import get_config


CONFIG = get_config()


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
