from keras.utils import load_img, Sequence, image_dataset_from_directory, img_to_array , array_to_img
import numpy as np

import json
from glob import glob
import random
import os
import math


from skimage.io import imread, imsave
from skimage.transform import resize


def get_config():
    with open('config.json') as f:
        config = json.load(f)
    return config


CONFIG = get_config()
CONSTANTS = CONFIG["Constants"]


def load_images(input_paths,return_array = True, color_mode="grayscale"):
    images = []
    for path in input_paths:
        images.append(load_img(path,color_mode=color_mode))
    if return_array:
        return [img_to_array(img) for img in images]
    return images


def check_and_send_paths(inp_path,target_path):

    assert len(inp_path) == len(target_path), "Error: inp & target path size is not same"
    for i_path,t_path in zip(inp_path,target_path):
        i_path = i_path.split('/')[-1]
        t_path = f"{t_path.split('/')[-1].replace('_seg','')}"
        
        assert i_path == t_path, f"Error: inp path {i_path} & target path {t_path} are not same"
    return inp_path,target_path


def get_path_glob(dir,shuffle=True):

    input_paths = sorted(glob(os.path.join(dir,"Ids","*.png")))
    target_paths = sorted(glob(os.path.join(dir,"GroundTruth","*.png")))

    if shuffle:
        shuffled_inp_path = []
        shuffled_target_path = []
        idx = list(range(len(input_paths)))

        random.seed(CONSTANTS["seed"])
        random.shuffle(idx)

        for i in idx:
            shuffled_inp_path.append(input_paths[i])
            shuffled_target_path.append(target_paths[i])

        return check_and_send_paths(shuffled_inp_path, shuffled_target_path)
    return check_and_send_paths(input_paths,target_paths)



class Dataloader(Sequence):
    """
    Loading dataset iteratively from directory.
    Avoid memory error by loading one batch at a time.
    reference: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """

    def __init__(self, bs, inp_paths, target_paths,img_size,target_dtype=bool):
        self.batch_size = bs
        self.input_paths = inp_paths
        self.target_paths = target_paths
        self.img_size = img_size
        self.target_dtype = target_dtype


    def __len__(self):
        "return number of batches"
        return math.ceil(len(self.target_paths) / self.batch_size)


    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        input_batch = self.input_paths[i: i+self.batch_size]
        target_batch = self.target_paths[i: i+self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,),dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,),dtype=self.target_dtype)
        for index, path in enumerate(input_batch):
            img = load_img(path,target_size=self.img_size)
            # img = imread(path)[:,:,:3]
            # img = resize(img,self.img_size,mode="constant",preserve_range=True)
            x[index] = img
            # imsave(f'test/inputs/{index}.png',img)

        for index, path in enumerate(target_batch):
            # img = imread(path,as_gray=True)
            # img = np.expand_dims(resize(img,self.img_size,mode="constant",preserve_range=True),2)
            img = np.expand_dims(load_img(path,target_size=self.img_size,color_mode="grayscale"),2)
            y[index] = img
            # imsave(f'test/targets/{index}.png',img)
        return x, y


def ds_from_dataloader(data_dir,bs,img_size,test,target_dtype):
    """
    Loading dataset iteratively from directory
    reference: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """

    if not test:
        input_path, target_path = get_path_glob(data_dir,shuffle=True)
        split_idx = int(len(input_path)*0.8)
        
        train_dataloader = Dataloader(bs,inp_paths=input_path[:split_idx]
                                                ,target_paths=target_path[:split_idx]
                                                ,img_size=img_size
                                                ,target_dtype=target_dtype)
        
        validation_dataloader = Dataloader(bs,inp_paths=input_path[split_idx+1:]
                                                ,target_paths=target_path[split_idx+1:]
                                                ,img_size=img_size
                                                ,target_dtype=target_dtype)
        return train_dataloader, validation_dataloader
    else:
        input_path, target_path = get_path_glob(data_dir,shuffle=False)
        test_dataloader = Dataloader(bs,inp_paths=input_path
                                                ,target_paths=target_path
                                                ,img_size=img_size
                                                ,target_dtype=target_dtype)
        return test_dataloader, None


def ds_from_directory(data_dir,bs=32,val_ratio=None,subset=None,shuffle=True):
    return image_dataset_from_directory(
        data_dir,
        validation_split=val_ratio,
        subset = subset,
        seed = CONSTANTS["seed"],
        batch_size = bs,
        shuffle=shuffle
    )
