from src.trainers.trainer import classification_trainer, segmentation_trainer, cleaner_trainer
from src.evaluators.evaluator import evaluate_classifier, evaluate_segmentation, predict_deskewing, evaluate_cleaner
from src.utils import get_config
from src.data_loaders import load_ocr_ds

import os
from argparse import ArgumentParser

import pytesseract
from PIL import Image
import json
from pathlib import Path

CONFIG = get_config()
CONSTANTS = CONFIG["Constants"]
TASK_CHOICES = ["classification","segmentation","deskewing","cleaning","ocr"]
CLASSICAL_MODEL_CHOICES = ["houg_transform","sift"]


def classifier(train,evaluate,model_name,predict,_):
    bs = CONSTANTS["classification"]["batch_size"]
    e = CONSTANTS["classification"]["epoch"]
    lr = CONSTANTS["classification"]["lr"]
    w = CONSTANTS["classification"]["img_size"]["width"]
    h = CONSTANTS["classification"]["img_size"]["height"]
    num_block = CONSTANTS["classification"]["num_block"]
    data_augmentation = CONSTANTS["classification"]["data_aug"]

    img_size = (h,w)
    
    if train:
        if data_augmentation:
            print("Data augmentation enabled")
            model_name = os.path.join(model_name,f"data_aug_e{e}_bs{bs}_sz{h}_lr{lr}.h5")
        else:
            model_name = os.path.join(model_name,f"e{e}_bs{bs}.h5")
        classification_trainer(model_name,img_size,num_block,e,bs,data_augmentation,lr)
    if evaluate:
        evaluate_classifier(model_name,bs,predict)


def segmentation(train,evaluate,model_name,predict,classical_model):
    bs = CONSTANTS["segmentation"]["batch_size"]
    e = CONSTANTS["segmentation"]["epoch"]
    lr = CONSTANTS["segmentation"]["lr"]
    w = CONSTANTS["img_size"]["width"]
    h = CONSTANTS["img_size"]["height"]
    ds_channels = CONSTANTS["segmentation"]["down_scaling_channels"]
    d_out = CONSTANTS["segmentation"]["drop_out"]
    
    img_size = (h,w)
    
    if train:
        model_name = os.path.join(model_name,f"iou_max_channel_{ds_channels[-1]}_dout_{d_out}_e{e}_bs{bs}_sz{h}_lr{lr}.h5")
        segmentation_trainer(model_name,img_size,ds_channels,d_out,e,bs,lr)
    if evaluate:
        evaluate_segmentation(model_name,bs,img_size,predict)


def deskew(train,evaluate,_,predict,classical_model):
    w = CONSTANTS["img_size"]["width"]
    h = CONSTANTS["img_size"]["height"]
    img_size = (h,w,3)
    print(img_size)
    if train:
        print("Deskewing Task doesn't support training & evaluation. Its only possible to get prediction. remove --train flag")
    elif evaluate and not predict:
        print("Deskewing Task doesn't support evaluation. Getting prediction. use --predict flag")
    else:
        predict_deskewing(img_size,classical_model)


def cleaning(train,evaluate,model_name,predict,classical_model):
    bs = CONSTANTS["cleaning"]["batch_size"]
    e = CONSTANTS["cleaning"]["epoch"]
    lr = CONSTANTS["cleaning"]["lr"]
    focal_gamma = CONSTANTS["cleaning"]["focal_gamma"]
    class_1_weight = CONSTANTS["cleaning"]["class_1_weight"]
    w = CONSTANTS["img_size"]["width"]
    h = CONSTANTS["img_size"]["height"]
    ds_channels = CONSTANTS["cleaning"]["down_scaling_channels"]
    d_out = CONSTANTS["cleaning"]["drop_out"]
    
    img_size = (h,w)
    
    if train:
        model_name = os.path.join(model_name,f"iou_max_channel_gamma{focal_gamma}_cw{class_1_weight}_{ds_channels[-1]}_dout_{d_out}_e{e}_bs{bs}_sz{h}_lr{lr}.h5")
        cleaner_trainer(model_name,img_size,ds_channels,d_out,e,bs,lr,focal_gamma,class_1_weight)
    if evaluate:
        evaluate_cleaner(model_name,bs,img_size,predict)


def ocr(train,evaluate,_,predict,classical_model):
    output={}
    w = CONSTANTS["img_size"]["width"]
    h = CONSTANTS["img_size"]["height"]
    if train:
        print("OCR Task doesn't support training & evaluation. Its only possible to get prediction. remove --train flag")
    elif evaluate and not predict:
        print("OCR Task doesn't support evaluation. Getting prediction. use --predict flag")
    else:
        images,input_paths = load_ocr_ds()
        for image,path in zip(images,input_paths):
            fname = path.split('/')[-1].split('.')[0]
            text = pytesseract.image_to_string(image)
            output[fname]=text.split()

    Path("data/5_ocr/").mkdir(parents=True,exist_ok=True)
    with open("data/5_ocr/final_output.json", "w") as outfile:
            json.dump(output, outfile)
        
        
def main(train,evaluate,predict,task,model_name,classical_model):
    default_model_path = os.path.join('models',task)
    if task in ["deskewing","ocr"]:
        TASK_FUNC[task](train,evaluate,default_model_path,predict,classical_model)
        return
    if not train and evaluate:
        assert model_name, "Please provide model name for evaluation or train a model first."
        model_name = os.path.join(default_model_path,model_name)
        assert os.path.exists(model_name), f"Model {model_name} doesn't exist."
        print(f'Evaluating the model {model_name} on task {task}')
        default_model_path = model_name
    TASK_FUNC[task](train,evaluate,default_model_path,predict,classical_model)


TASK_FUNC = {"classification":classifier,
            "segmentation":segmentation,
            "deskewing":deskew,
            "cleaning":cleaning,
            "ocr":ocr}
 
if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Train/evaluate document processing tasks.")
    arg_parser.add_argument("-t", "--task", required=True, choices = TASK_CHOICES, type = str.lower, help="Task name to train/evaluate.")
    arg_parser.add_argument("--no_evaluation", default=False, action="store_true", help="Don't Evaluate.")
    arg_parser.add_argument("--train", default=False, action="store_true", help="Start Training.")
    arg_parser.add_argument("-p","--predict", default=False, action="store_true", help="Save predictions.")
    arg_parser.add_argument("-m","--model_name", default=None, help="Model to use for evaluation/prediction. This argument is not used during training!")
    arg_parser.add_argument("--classical_model", choices=CLASSICAL_MODEL_CHOICES, help="name of classical model to use in Tasks like segmentation,deskewing,cleaning. This argument is mandatory for deskewing task")
    _args = arg_parser.parse_args()
    
    main(_args.train,not(_args.no_evaluation),_args.predict, _args.task,_args.model_name,_args.classical_model)
