import tensorflow as tf


def callback_earlystop(checkpoint_path):
    checkpointer=tf.keras.callbacks.ModelCheckpoint(checkpoint_path,verbose=2,save_best_only=True)
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10,monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs'),
        checkpointer
        ]
    return callbacks