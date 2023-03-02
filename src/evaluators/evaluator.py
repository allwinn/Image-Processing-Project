
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from keras.utils import load_img, array_to_img
from PIL import Image, ImageDraw, ImageFilter

from src.data_loaders import load_classification_ds, load_segmentation_ds, load_deskewing_ds, load_cleaning_ds
from src.utils import get_config
from src.trainers.models import classical_model_deskew_hough

import os
from pathlib import Path

CONFIG = get_config()
PATHS = CONFIG["Paths"]["prediction_path"]


def evaluate_classifier(model_name,bs,predict=False):

    print(f"{'*'*10} loading dataset for evaluating classification task {'*'*10}")
    test_ds, _ = load_classification_ds(test=True, bs=bs)
    model = tf.keras.models.load_model(model_name)
    score = model.evaluate(test_ds)

    print(f"{'*'*10} Evaluation completed with test accuracy {score[1]} and test loss {score[0]} {'*'*10}")

    if predict:
        predictions = model.predict(test_ds)
        predictions = (predictions<=0.5).astype(np.uint8)
        # predictions = np.argmax(predictions,axis=1)
        Path(os.path.join(PATHS["root"],"classification")).mkdir(parents=True,exist_ok=True)
        pred_fn = os.path.join(PATHS["root"],"classification",f"{model_name.split('/')[-1]}_{PATHS['classification_fn']}")
        
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
    # print(f"{'*'*10} Evaluation completed with test accuracy {score[1]} , iou {score[2]} and test loss {score[0]} {'*'*10}")

    if predict:
        predictions = model.predict(test_ds)
        # predictions = np.argmax(predictions,axis=-1)
        # predictions = np.expand_dims(predictions,axis=-1)
        predictions = (predictions<=0.5).astype(np.uint8)
        

        for idx, path in enumerate(test_ds.input_paths):
            #path where predicted img will be save
            pred_path = os.path.join(PATHS["root"],"segmentation",f"{model_name.split('/')[-1][:-3]}_pred")
            #path where blended img will be save
            blend_path = os.path.join(pred_path,"blended")
            #creating path if not exist
            Path(pred_path).mkdir(parents=True,exist_ok=True)
            Path(blend_path).mkdir(parents=True,exist_ok=True)
            # extracting file name of the input img. Output img name will be based on input img.
            fname = path.split('/')[-1].split('.')[0]
            # complete path and name of the predicted img
            pred_fname = os.path.join(pred_path,f"{fname}_seg.png")
            blend_fname = os.path.join(blend_path,f"{fname}_blend.png")
            
            # load input image
            img = load_img(path,target_size=img_size)
            pred_mask_img = array_to_img(predictions[idx]).resize(img.size).convert(mode='RGB')

            #save the predicted mask image
            pred_mask_img.save(pred_fname)
            #Blend input image and mask
            im = Image.blend(img, pred_mask_img,0.5)
            im.save(blend_fname)


def predict_deskewing(img_size,classical_model):
    print(f"{'*'*10} loading dataset for deskewing task {'*'*10}")
    gray_images, rgb_images, inp_paths = load_deskewing_ds()
    pred_path = os.path.join(PATHS["root"],"deskewing")
    Path(pred_path).mkdir(parents=True,exist_ok=True)

    print("Deskewing process started...")
    if classical_model == "houg_transform":
        for gray_img, rgb_img, path in zip(gray_images,rgb_images,inp_paths):
            deskewed_img = classical_model_deskew_hough(img_size,gray_img,rgb_img)
            pred_fname = os.path.join(pred_path,f"{path.split('/')[-1].split('.')[0]}_deskew.png")
            deskew_img = array_to_img(deskewed_img)
            deskew_img.save(pred_fname)
    elif classical_model == "sift":
        pass
    else:
        raise Exception("Unsupported classical model")
    print(f"{'*'*10} Deskewing completed. Please the data here {pred_path} {'*'*10}")


from skimage.io import imread, imsave
from skimage.transform import resize
import skimage



def evaluate_cleaner(model_name,bs,img_size,predict=False):
    
    print(f"{'*'*10} loading dataset for evaluating cleaning task {'*'*10}")
    test_ds, _ = load_cleaning_ds(bs=bs,img_size=img_size,test=True)
    model = tf.keras.models.load_model(model_name)
    score = model.evaluate(test_ds)
    # print(f"{'*'*10} Evaluation completed with test accuracy {score[1]} , iou {score[2]} and test loss {score[0]} {'*'*10}")

    if predict:
        predictions = model.predict(test_ds)
        # predictions = np.argmax(predictions,axis=-1)
        # predictions = np.expand_dims(predictions,axis=-1)
        predictions = (predictions>0.5).astype(np.uint8)

        

        for idx, path in enumerate(zip(test_ds.input_paths,test_ds.target_paths)):
            #path where predicted img will be save
            pred_path = os.path.join(PATHS["root"],"cleaning",f"{model_name.split('/')[-1][:-3]}_pred")
            data_path = os.path.join(pred_path,"data","inputs")
            data_path1 = os.path.join(pred_path,"data","targets")
            #creating path if not exist
            Path(pred_path).mkdir(parents=True,exist_ok=True)
            #creating path if not exist

            Path(data_path1).mkdir(parents=True,exist_ok=True)
            Path(data_path).mkdir(parents=True,exist_ok=True)

            # extracting file name of the input img. Output img name will be based on input img.
            fname = path[0].split('/')[-1].split('.')[0]
            # complete path and name of the predicted img
            pred_fname = os.path.join(pred_path,f"{fname}.png")
            pred_img = array_to_img(predictions[idx])

            inp_img = imread(path[0])
            inp_img = resize(inp_img,img_size,mode="constant",preserve_range=True)
            
            targ_img = imread(path[1],as_gray=True)
            targ_img = np.expand_dims(resize(targ_img,img_size,mode="constant",preserve_range=True),axis=-1)
            
            
            imsave(os.path.join(data_path,path[0].split('/')[-1]),inp_img.astype(np.uint8))
            imsave(os.path.join(data_path1,path[1].split('/')[-1]),targ_img)

            #save the predicted  image
            pred_img.save(pred_fname)