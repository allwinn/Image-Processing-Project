from src.utils import get_config, get_segmentation_glob, ds_from_directory, SegmentationDataloader

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
    train_dir=os.path.join(PATHS["root"],PATHS["segmentation"]["train"])
    test_dir=os.path.join(PATHS["root"],PATHS["segmentation"]["test"])

    if not test:
        input_path, target_path = get_segmentation_glob(train_dir,shuffle=True)
        split_idx = int(len(input_path)*0.8)
        
        train_dataloader = SegmentationDataloader(bs,inp_paths=input_path[:split_idx]
                                                ,target_paths=target_path[:split_idx]
                                                ,img_size=img_size)
        
        validation_dataloader = SegmentationDataloader(bs,inp_paths=input_path[split_idx+1:]
                                                ,target_paths=target_path[split_idx+1:]
                                                ,img_size=img_size)
        return train_dataloader, validation_dataloader
    else:
        input_path, target_path = get_segmentation_glob(test_dir,shuffle=False)
        test_dataloader = SegmentationDataloader(bs,inp_paths=input_path
                                                ,target_paths=target_path
                                                ,img_size=img_size)
        return test_dataloader, None
