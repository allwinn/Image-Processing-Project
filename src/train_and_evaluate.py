from src.trainers.trainer import classification_trainer, segmentation_trainer
from src.evaluators.evaluator import evaluate_classifier, evaluate_segmentation
from src.utils import get_config

CONFIG = get_config()
CONSTANTS = CONFIG["Constants"]


def classifier(train=False,evaluate=False):
    bs = CONSTANTS["classification"]["batch_size"]
    e = CONSTANTS["classification"]["epoch"]
    data_augmentation = CONSTANTS["classification"]["data_aug"]
    if data_augmentation:
        print("Data augmentation enabled")
        model_name = f"models/classification_data_aug_e{e}_bs{bs}.h5"
    else:
        model_name = f"models/classification_e{e}_bs{bs}.h5"

    
    
    if train:
        num_block = CONSTANTS["classification"]["num_block"]
        classification_trainer(model_name,num_block,e,bs,data_augmentation)
    if evaluate:
        evaluate_classifier(model_name,bs,predict=True)


def segmentation(train=False,evaluate=False):
    bs = CONSTANTS["segmentation"]["batch_size"]
    e = CONSTANTS["segmentation"]["epoch"]
    w = CONSTANTS["segmentation"]["img_size"]["width"]
    h = CONSTANTS["segmentation"]["img_size"]["height"]
    ds_channels = CONSTANTS["segmentation"]["down_scaling_channels"]
    d_out = CONSTANTS["segmentation"]["drop_out"]
    model_name = f"models/segmentation_iou_max_channel_{ds_channels[-1]}_dout_{d_out}_e{e}_bs{bs}.h5"
    img_size = (h,w)
    
    if train:
        segmentation_trainer(model_name,ds_channels,d_out,e,bs)
    if evaluate:
        evaluate_segmentation(model_name,bs,img_size,predict=True)


if __name__ == "__main__":
    # id_present(train=True,evaluate=True)
    # segmentation(train=False,evaluate=True)
    classifier(train=True,evaluate=True)


## Test accuracy for segmentation task with different level of down scaling.

# ********** Evaluation completed with test accuracy 0.9916638135910034 and test loss 0.022273050621151924 ********** 256
# ********** Evaluation completed with test accuracy 0.9759243726730347 and test loss 0.0670047253370285 **********  128
# ********** Evaluation completed with test accuracy 0.9785996079444885 and test loss 0.05778338015079498 ********** 64
# ********** Evaluation completed with test accuracy 0.9707129597663879 and test loss 0.07916072010993958 ********** 32


#iou
# loss: 0.0267 - binary_accuracy: 0.9896 - binary_io_u: 0.8900 --> 256
# loss: 0.0461 - binary_accuracy: 0.9840 - binary_io_u: 0.8412 --> 128
# loss: 0.0419 - binary_accuracy: 0.9846 - binary_io_u: 0.8473 --> 64
# loss: 0.0796 - binary_accuracy: 0.9706 - binary_io_u: 0.7026 --> 32