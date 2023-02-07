
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

from src.data_loaders import load_classification_ds, load_segmentation_ds
from src.utils import get_config

import os

CONFIG = get_config()
CONSTANTS = CONFIG["Constants"]
PATHS = CONFIG["Paths"]["data_path"]


def evaluate_classifier(model_name,bs,predict=False):

    print(f"{'*'*10} loading dataset for evaluating classification task {'*'*10}")
    test_ds, _ = load_classification_ds(test=True, bs=bs)
    model = tf.keras.models.load_model(model_name)
    score = model.evaluate(test_ds)

    print(f"{'*'*10} Evaluation completed with test accuracy {score[1]} and test loss {score[0]} {'*'*10}")

    if predict:
        predictions = model.predict(test_ds)
        predictions = np.argmax(predictions,axis=1)
        
        fnames = [path.split('/')[-1] for path in test_ds.file_paths]
        labels = [ 0 if name.split('_')[-1].split('.')[0] in ["front","back"] else 1 for name in fnames ]
        test_df = pd.DataFrame({"file_name":fnames,"label":labels})
        
        test_df.loc[:,'predict_label'] = predictions
        test_df.to_csv(os.path.join(PATHS["prediction_dir"],"classification_test_pred.csv"))

        print(classification_report(test_df["label"],test_df["predict_label"],target_names=["id","non_id"]))


def evaluate_segmentation(model_name,bs,img_size,predict=False):
    
    print(f"{'*'*10} loading dataset for evaluating segmentation task {'*'*10}")
    test_ds, _ = load_segmentation_ds(bs=bs,img_size=img_size,test=True)
    model = tf.keras.models.load_model(model_name)
    score = model.evaluate(test_ds)

    print(f"{'*'*10} Evaluation completed with test accuracy {score[1]} and test loss {score[0]} {'*'*10}")

    # if predict:
    #     predictions = model.predict(test_ds)
    #     predictions = np.argmax(predictions,axis=1)
        
    #     fnames = [path.split('/')[-1] for path in test_ds.file_paths]
    #     labels = [ 0 if name.split('_')[-1].split('.')[0] in ["front","back"] else 1 for name in fnames ]
    #     test_df = pd.DataFrame({"file_name":fnames,"label":labels})
        
    #     test_df.loc[:,'predict_label'] = predictions
    #     test_df.to_csv(os.path.join(PATHS["prediction_dir"],"classification_test_pred.csv"))

    #     print(classification_report(test_df["label"],test_df["predict_label"],target_names=["id","non_id"]))
