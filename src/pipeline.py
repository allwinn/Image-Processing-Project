from src.trainers.trainer import classification_trainer, segmentation_trainer
from src.evaluators.evaluator import evaluate_classifier, evaluate_segmentation
from src.utils import get_config

CONFIG = get_config()
CONSTANTS = CONFIG["Constants"]


def id_present(train=False,evaluate=False):
    bs = CONSTANTS["classification"]["batch_size"]
    e = CONSTANTS["classification"]["epoch"]
    model_name = f"models/classification_e{e}_bs{bs}.h5"
    
    
    if train:
        num_block = CONSTANTS["classification"]["num_block"]
        classification_trainer(model_name,num_block,e,bs)
    if evaluate:
        evaluate_classifier(model_name,bs,predict=True)


def seg(train=False,evaluate=False):
    bs = CONSTANTS["segmentation"]["batch_size"]
    e = CONSTANTS["segmentation"]["epoch"]
    w = CONSTANTS["segmentation"]["img_size"]["width"]
    h = CONSTANTS["segmentation"]["img_size"]["height"]
    model_name = f"models/segmentation_b_acc_e{e}_bs{bs}.h5"
    img_size = (h,w)

    if train:
        segmentation_trainer(model_name,e,bs)
    if evaluate:
        evaluate_segmentation(model_name,bs,img_size,predict=True)


if __name__ == "__main__":
    # id_present(train=True,evaluate=True)
    seg(train=False,evaluate=True)