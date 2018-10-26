import argparse
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from baxter import BaxterConfig, BaxterDataset, DATASET_DIR, MODEL_DIR
from config import GeneralConfig
gconf = GeneralConfig()

from ipdb import set_trace

OLD_MODEL_PATH = gconf.OLD_MODEL_PATH
MODEL_DIR_NEW = gconf.MODEL_DIR_NEW
DATASET_DIR_NEW = gconf.DATASET_DIR_NEW
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def main(args):
    config = BaxterConfig()
    dataset_train = BaxterDataset()
    dataset_train.load_baxter(DATASET_DIR, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BaxterDataset()
    dataset_val.load_baxter(DATASET_DIR, "val")
    dataset_val.prepare()


    # Load and display random samples
    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        # visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

        # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = args.init_with  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)
    elif init_with == 'weights':
        model.load_weights(args.old_model_path, by_name=True)
    # elif init_with == 'retrain':
    #     model.load_weights(RETRAIN_MODEL_PATH, by_name=True)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    if OLD_MODEL_PATH is not None:
        print("Load old model and finetun")
        model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE / 10, 
                    epochs=int((OLD_MODEL_PATH.split('_')[-1]).split('.')[0]) + args.n_epochs, 
                    layers='heads')
                    # , augmentation=imgaug.augmenters.OneOf([
                            #imgaug.augmenters.Fliplr(0.5),
                            #imgaug.augmenters.Flipud(0.5),
    else:                     #imgaug.augmenters.Affine(rotate=(-90, 90))])
        print("Train new model (initialized from MSCOCO")
        model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE / 10, 
                    epochs=args.n_epochs, 
                    layers='heads')
                    # , augmentation=imgaug.augmenters.OneOf([
                            #imgaug.augmenters.Fliplr(0.5),
                            #imgaug.augmenters.Flipud(0.5),
                            #imgaug.augmenters.Affine(rotate=(-90, 90))])
                                         

    # # Fine tune all layers
    # # Passing layers="all" trains all layers. You can also 
    # # pass a regular expression to select which layers to
    # # train by name pattern.
    # model.train(dataset_train, dataset_val, 
    #             learning_rate=config.LEARNING_RATE / 10,
    #             epochs=2, 
    #             layers="all")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-epochs', type=int, default=100)
    parser.add_argument('-i', '--init-with', type=str, default='coco')
    parser.add_argument('-m', '--old-model-path', type=str, default=OLD_MODEL_PATH)

    args = parser.parse_args()
    main(args)        
