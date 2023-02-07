from src.trainers.trainer import classification_trainer, segmentation_trainer
from src.evaluators.evaluator import evaluate_classifier, evaluate_segmentation
from src.utils import get_config

CONFIG = get_config()
CONSTANTS = CONFIG["Constants"]


def id_present(train=False,evaluate=False):
    bs = CONSTANTS["classification"]["batch_size"]
    e = CONSTANTS["classification"]["epoch"]
    
    if train:
        num_block = CONSTANTS["classification"]["num_block"]
        classification_trainer(num_block,e,bs)
    if evaluate:
        evaluate_classifier("models/classification_e5_bs64.h5",bs,predict=True)


def seg(train=False,evaluate=False):
    bs = CONSTANTS["segmentation"]["batch_size"]
    e = CONSTANTS["segmentation"]["epoch"]
    w = CONSTANTS["segmentation"]["img_size"]["width"]
    h = CONSTANTS["segmentation"]["img_size"]["height"]
    img_size = (h,w)

    if train:
        segmentation_trainer(e,bs)
    if evaluate:
        evaluate_segmentation("models/segmentation_e25_bs16.h5",bs,img_size,predict=False)


if __name__ == "__main__":
    # id_present(train=True,evaluate=True)
    seg(evaluate=True)