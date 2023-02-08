
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from keras.utils import load_img, array_to_img
from PIL import Image, ImageDraw, ImageFilter

from src.data_loaders import load_classification_ds, load_segmentation_ds
from src.utils import get_config

import os
from pathlib import Path

CONFIG = get_config()
CONSTANTS = CONFIG["Constants"]
PATHS = CONFIG["Paths"]["prediction_path"]


def evaluate_classifier(model_name,bs,predict=False):

    print(f"{'*'*10} loading dataset for evaluating classification task {'*'*10}")
    test_ds, _ = load_classification_ds(test=True, bs=bs)
    model = tf.keras.models.load_model(model_name)
    score = model.evaluate(test_ds)

    print(f"{'*'*10} Evaluation completed with test accuracy {score[1]} and test loss {score[0]} {'*'*10}")

    if predict:
        predictions = model.predict(test_ds)
        predictions = np.argmax(predictions,axis=1)
        pred_fn = os.path.join(PATHS["root"],"classification",f"model_name_{PATHS['classification_fn']}")
        
        fnames = [path.split('/')[-1] for path in test_ds.file_paths]
        labels = [ 0 if name.split('_')[-1].split('.')[0] in ["front","back"] else 1 for name in fnames ]
        test_df = pd.DataFrame({"file_name":fnames,"label":labels})
        
        test_df.loc[:,'predict_label'] = predictions
        test_df.to_csv(pred_fn)

        print(classification_report(test_df["label"],test_df["predict_label"],target_names=["id","non_id"]))


def evaluate_segmentation(model_name,bs,img_size,predict=False):
    
    print(f"{'*'*10} loading dataset for evaluating segmentation task {'*'*10}")
    test_ds, _ = load_segmentation_ds(bs=bs,img_size=img_size,test=True)
    model = tf.keras.models.load_model(model_name)
    score = model.evaluate(test_ds)

    print(f"{'*'*10} Evaluation completed with test accuracy {score[1]} and test loss {score[0]} {'*'*10}")

    if predict:
        predictions = model.predict(test_ds)
        # predictions = np.argmax(predictions,axis=-1)
        # predictions = np.expand_dims(predictions,axis=-1)
        predictions = (predictions>0.5).astype(np.uint8)
        

        for idx, path in enumerate(test_ds.input_paths):
            
            pred_path = os.path.join(PATHS["root"],"segmentation",f"{model_name.split('/')[-1]}_pred")
            Path(pred_path).mkdir(parents=True,exist_ok=True)

            fname = f"{path.split('/')[-1].split('.')[0]}_seg.png"
            pred_fname = os.path.join(pred_path,fname)
            img = load_img(path,target_size=img_size)
            target_mask_img = load_img(test_ds.target_paths[idx],target_size=img_size)
            pred_mask_img = array_to_img(predictions[idx]).resize(img.size).convert(mode='RGB')


            im = Image.blend(img, pred_mask_img,0.5)
            im.save(pred_fname)
