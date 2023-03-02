from src.evaluators.evaluator import evaluate_classifier, evaluate_segmentation, predict_deskewing, evaluate_cleaner
from src.utils import load_images
from src.utils import get_config
from glob import glob
import tensorflow as tf

import numpy as np
from keras.utils import load_img, array_to_img

import os
from argparse import ArgumentParser

from skimage.io import imread, imsave
from skimage.transform import resize
from pathlib import Path

from src.trainers.models import classical_model_deskew_hough
from src.evaluators.evaluator_utils import crop_with_mask
import cv2


CONFIG = get_config()
CONSTANTS = CONFIG["Constants"]
PATHS = CONFIG["Paths"]["data_path"]
TASK_CHOICES = ["classification","segmentation","deskewing","cleaning","ocr"]
CLASSICAL_MODEL_CHOICES = ["houg_transform","sift"]


DEFAULT_MODELS = {"classification":"",
                  "segmentation":"",
                  "cleaning":""}
DEFAULT_BS = 10


def classifier(model_name=None):
    if model_name is None:
        model_name = DEFAULT_MODELS["classification"]
    pipeline_dir=os.path.join(PATHS["root"],PATHS["classification"]["pipeline_input"])
    paths = glob(os.path.join(pipeline_dir,"*.png"))
    data = load_images(paths,return_array=True,color_mode="rgb")
    model = tf.keras.models.load_model(model_name)
    predictions = model.predict(data)
    predictions = (predictions<=0.5).astype(np.uint8)
    return paths,predictions


def segmentation(input_paths,model_name):
    w = CONSTANTS["img_size"]["width"]
    h = CONSTANTS["img_size"]["height"]
    img_size = (h,w)
    inputs = np.zeros((len(input_paths),h,w,3),dtype=np.uint8)

    for index,path in enumerate(input_paths):
        img = imread(path)[:,:,3]
        img = [resize(img,(img_size),mode='constant',preserve_range=True)]
        inputs[index] = img
    model = tf.keras.models.load_model(model_name)
    predictions = model.predict(inputs)
    predictions = (predictions>0.5).astype(np.uint8)

    for idx, path in enumerate(input_paths):
        pipeline_out_dir=os.path.join(PATHS["root"],PATHS["segmentation"]["pipeline_output"],"pred")
        crop_dir=os.path.join(PATHS["root"],PATHS["segmentation"]["pipeline_output"])
        Path(pipeline_out_dir).mkdir(parents=True,exist_ok=True)
        Path(crop_dir).mkdir(parents=True,exist_ok=True)
        # extracting file name of the input img. Output img name will be based on input img.
        fname = path.split('/')[-1].split('.')[0]
        # complete path and name of the predicted img
        crop_fname = os.path.join(crop_dir,f"{fname}_seg.png")
        pred_fname = os.path.join(pipeline_out_dir,f"{fname}_seg.png")
        
        pred_mask_img = array_to_img(predictions[idx])
        
        #save the predicted mask image
        pred_mask_img.save(pred_fname)
        #crop input image and mask
        croped_img = crop_with_mask(path,pred_fname)
        cv2.imwrite(crop_fname, croped_img)


def deskew(classical_model="houg_transform"):
    w = CONSTANTS["img_size"]["width"]
    h = CONSTANTS["img_size"]["height"]
    img_size = (h,w,3)

    deskew_dir=os.path.join(PATHS["root"],PATHS["deskewing"]["pipeline_output"])
    Path(deskew_dir).mkdir(parents=True,exist_ok=True)
    data_dir=os.path.join(PATHS["root"],PATHS["deskewing"]["pipeline_input"])
    input_paths = glob(os.path.join(data_dir,"*.png"))
    gray_images = load_images(input_paths,return_array=True,color_mode="grayscale")
    rgb_images = load_images(input_paths,return_array=True,color_mode="rgb")
    
    print("Deskewing process started...")
    for gray_img, rgb_img, path in zip(gray_images,rgb_images,input_paths):
        deskewed_img = classical_model_deskew_hough(img_size,gray_img,rgb_img)
        pred_fname = os.path.join(deskew_dir,f"{path.split('/')[-1].split('.')[0]}_deskew.png")
        deskew_img = array_to_img(deskewed_img)
        deskew_img.save(pred_fname)
    print(f"{'*'*10} Deskewing completed. Please the data here {deskew_dir} {'*'*10}")


def cleaning(model_name):
    w = CONSTANTS["img_size"]["width"]
    h = CONSTANTS["img_size"]["height"]
    img_size = (h,w)
    if model_name is None:
        model_name = DEFAULT_MODELS["classification"]
    evaluate_cleaner(model_name,DEFAULT_BS,img_size,True)


def ocr(train,evaluate,model_name,predict,classical_model):
    pass

def main():
    paths,cls_pred= classifier()

    new_paths = []
    for path, is_id in zip(paths,cls_pred):
        if is_id==0:
            print(f'File at {path} is Id. Starting document processing')
            new_paths.append(path)
        else:
            print(f"File at {path} is Not Id. Can't process it.")
    
    segmentation(new_paths)


    for 



 
if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Pipeline for Document Processing")
    arg_parser.add_argument("-t", "--task", required=True, choices = TASK_CHOICES, type = str.lower, help="Task name to train/evaluate.")
    _args = arg_parser.parse_args()
    
    main(_args.train)
