from keras.utils import load_img, Sequence, image_dataset_from_directory 
import numpy as np

import json
from glob import glob
import random
import os


def get_config():
    with open('config.json') as f:
        config = json.load(f)
    return config


CONFIG = get_config()
CONSTANTS = CONFIG["Constants"]

def get_segmentation_glob(dir,shuffle=True):

    input_paths = glob(os.path.join(dir,"Ids","*.png"))
    target_paths = glob(os.path.join(dir,"GroundTruth","*.png"))
    if shuffle:
        shuffled_inp_path = []
        shuffled_target_path = []
        idx = list(range(len(input_paths)))

        random.seed(CONSTANTS["seed"])
        random.shuffle(idx)

        for i in idx:
            shuffled_inp_path.append(input_paths[i])
            shuffled_target_path.append(target_paths[i])

        assert len(shuffled_inp_path) == len(input_paths)
        return shuffled_inp_path, shuffled_target_path
    return input_paths,target_paths



class SegmentationDataloader(Sequence):
    """
    Loading dataset iteratively from directory.
    Avoid memory error by loading one batch at a time.
    reference: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """

    def __init__(self, bs, inp_paths, target_paths,img_size):
        self.batch_size = bs
        self.input_paths = inp_paths
        self.target_paths = target_paths
        self.img_size = img_size


    def __len__(self):
        "return number of batches"
        return len(self.target_paths) // self.batch_size


    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        input_batch = self.input_paths[i: i+self.batch_size]
        target_batch = self.target_paths[i: i+self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,),dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,),dtype=bool)

        for index, path in enumerate(input_batch):
            img = load_img(path,target_size=self.img_size)
            x[index] = img
        
        for index, path in enumerate(target_batch):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[index] = np.expand_dims(img,2)
        return x, y


def ds_from_directory(data_dir,bs=32,val_ratio=None,subset=None,shuffle=True):

    return image_dataset_from_directory(
        data_dir,
        validation_split=val_ratio,
        subset = subset,
        seed = CONSTANTS["seed"],
        batch_size = bs,
        shuffle=shuffle

    )
