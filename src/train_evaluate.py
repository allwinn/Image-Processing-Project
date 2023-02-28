from src.trainers.trainer import classification_trainer, segmentation_trainer
from src.evaluators.evaluator import evaluate_classifier, evaluate_segmentation
from src.utils import get_config

import os
from argparse import ArgumentParser

CONFIG = get_config()
CONSTANTS = CONFIG["Constants"]
TASK_CHOICES = ["classification","segmentation","deskewing","cleaning","ocr"]


def classifier(train,evaluate,model_name,predict):
    bs = CONSTANTS["classification"]["batch_size"]
    e = CONSTANTS["classification"]["epoch"]
    num_block = CONSTANTS["classification"]["num_block"]
    data_augmentation = CONSTANTS["classification"]["data_aug"]
    
    if train:
        if data_augmentation:
            print("Data augmentation enabled")
            model_name = os.path.join(model_name,f"data_aug_e{e}_bs{bs}.h5")
        else:
            model_name = os.path.join(model_name,f"e{e}_bs{bs}.h5")
        classification_trainer(model_name,num_block,e,bs,data_augmentation)
    if evaluate:
        evaluate_classifier(model_name,bs,predict)


def segmentation(train,evaluate,model_name,predict):
    bs = CONSTANTS["segmentation"]["batch_size"]
    e = CONSTANTS["segmentation"]["epoch"]
    w = CONSTANTS["segmentation"]["img_size"]["width"]
    h = CONSTANTS["segmentation"]["img_size"]["height"]
    ds_channels = CONSTANTS["segmentation"]["down_scaling_channels"]
    d_out = CONSTANTS["segmentation"]["drop_out"]
    
    img_size = (h,w)
    
    if train:
        model_name = os.path.join(model_name,f"iou_max_channel_{ds_channels[-1]}_dout_{d_out}_e{e}_bs{bs}.h5")
        segmentation_trainer(model_name,ds_channels,d_out,e,bs)
    if evaluate:
        evaluate_segmentation(model_name,bs,img_size,predict)


def main(train,evaluate,predict,task,model_name):
    try:
        default_model_path = os.path.join('models',task)
        if not train and evaluate:
            assert model_name, "Please provide model name for evaluation or train a model first."
            model_name = os.path.join(default_model_path,model_name)
            assert os.path.exists(model_name), f"Model {model_name} doesn't exist."
            print(f'Evaluating the model {model_name} on task {task}')
            default_model_path = model_name
        
        TASK_FUNC[task](train,evaluate,default_model_path,predict)
    except Exception as e:
        print(e)


TASK_FUNC = {"classification":classifier, "segmentation":segmentation}
 
if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Train/evaluate document processing tasks.")
    arg_parser.add_argument("-t", "--task", required=True, choices = TASK_CHOICES, type = str.lower, help="Task name to train/evaluate.")
    arg_parser.add_argument("--no_evaluation", default=False, action="store_true", help="Don't Evaluate.")
    arg_parser.add_argument("--train", default=False, action="store_true", help="Start Training.")
    arg_parser.add_argument("-p","--predict", default=False, action="store_true", help="Save predictions.")
    arg_parser.add_argument("-m","--model_name", default=None, help="Model to use for evaluation/prediction. This argument is not used during training!")
    _args = arg_parser.parse_args()
    
    main(_args.train,not(_args.no_evaluation),_args.predict, _args.task,_args.model_name)
