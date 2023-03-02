from src.utils import get_config, ds_from_directory, ds_from_dataloader,load_images
from glob import glob

import os

CONFIG = get_config()
CONSTANTS = CONFIG["Constants"]
PATHS = CONFIG["Paths"]["data_path"]

 
def load_classification_ds(bs,test=False):
    
    """
    Loading dataset from directory when label is infered
    reference:https://www.tensorflow.org/tutorials/load_data/images
    """
    
    train_dir=os.path.join(PATHS["root"],PATHS["classification"]["train"])
    test_dir=os.path.join(PATHS["root"],PATHS["classification"]["test"])

    if not test:
        print(f"{'*'*10} train dir: {train_dir} {'*'*10}")
        train_ds = ds_from_directory(train_dir
                                    ,bs=bs
                                    ,val_ratio=0.2
                                    ,subset="training")

        val_ds = ds_from_directory(train_dir
                                    ,bs=bs
                                    ,val_ratio=0.2
                                    ,subset="validation")
        
        return train_ds, val_ds
    else:
        print(f"{'*'*10} test dir: {test_dir} {'*'*10}")
        test_ds = ds_from_directory(test_dir
                                    ,bs=bs
                                    ,shuffle=False)
        return test_ds, None


def load_segmentation_ds(bs,img_size,test=False):
    """
    Loading dataset iteratively from directory
    reference: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """
    if test:
        data_dir = os.path.join(PATHS["root"],PATHS["segmentation"]["test"])
    else:
        data_dir = os.path.join(PATHS["root"],PATHS["segmentation"]["train"])

    return ds_from_dataloader(data_dir,bs,img_size,test)


def load_deskewing_ds():
    data_dir=os.path.join(PATHS["root"],PATHS["deskewing"]["data"])
    input_paths = glob(os.path.join(data_dir,"*.png"))
    gray_images = load_images(input_paths,return_array=True,color_mode="grayscale")
    rgb_images = load_images(input_paths,return_array=True,color_mode="rgb")
    return gray_images, rgb_images, input_paths


def load_cleaning_ds(bs,img_size,test=False):
    """
    Loading dataset iteratively from directory
    reference: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """
    if test:
        data_dir = os.path.join(PATHS["root"],PATHS["cleaning"]["test"])
    else:
        data_dir = os.path.join(PATHS["root"],PATHS["cleaning"]["train"])

    return ds_from_dataloader(data_dir,bs,img_size,test)