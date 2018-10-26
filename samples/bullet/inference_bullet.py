import argparse
import os
from os.path import join
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import torch

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
FROM_DATASET = False
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from bullet import BulletConfig, BulletDataset

from config import GeneralConfig
gconf = GeneralConfig()
DATASET_DIR, MODEL_DIR, OLD_MODEL_PATH, COCO_MODEL_PATH, WEIGHTS_FILE_PATH, EXP_DIR  \
        = gconf.DATASET_DIR, gconf.MODEL_DIR, gconf.OLD_MODEL_PATH, gconf.COCO_MODEL_PATH, gconf.WEIGHTS_FILE_PATH, gconf.EXP_DIR

from ipdb import set_trace

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
# DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

class InferenceConfig(BulletConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return fig, ax

def main(args):

    inference_config = InferenceConfig()

    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR,
                                  config=inference_config)

    model.load_weights(WEIGHTS_FILE_PATH, by_name=True)

    dataset = BulletDataset()
    dataset.load_bullet(DATASET_DIR, "train")
    dataset.prepare()
    colors = visualize.random_colors(10)


    if FROM_DATASET:
        # # Test on a random image
        # Validation dataset


        # image_id = random.choice(dataset.image_ids)
        # original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        #     modellib.load_image_gt(dataset, inference_config, 
        #                            image_id, use_mini_mask=False)

        # log("original_image", original_image)
        # log("image_meta", image_meta)
        # log("gt_class_id", gt_class_id)
        # log("gt_bbox", gt_bbox)
        # log("gt_mask", gt_mask)

        # visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
        #                             dataset_train.class_names, figsize=(8, 8))

        # Compute VOC-Style mAP @ IoU=0.5
        # Running on 10 images. Increase for better accuracy.
        # image_ids = np.random.choice(dataset.image_ids, 10)
        image_ids = dataset.image_ids
        APs = []
        for image_id in image_ids:
            # # Load image and ground truth data
            # image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            #     modellib.load_image_gt(dataset, inference_config,
            #                            image_id, use_mini_mask=False)
            # molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
            # # Run object detection
            # results = model.detect([image], verbose=0)
            # r = results[0]
            # # Compute AP
            # AP, precisions, recalls, overlaps =\
            #     utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
            #                      r["rois"], r["class_ids"], r["scores"], r['masks'])
            # APs.append(AP)

            image = dataset.load_image(image_id)

            results = model.detect([image], verbose=1)

            r = results[0]
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    dataset.class_names, r['scores'], ax=get_ax(), colors=colors)

    else:
        # for seq_name in os.listdir('/home/msieb/projects/Mask_RCNN/datasets/bullet/test'):
        #     print("Processing ", seq_name)
        # # seq_name = args.seqname
        #     dataset_dir = os.path.join(DATASET_DIR, "test", seq_name)
        #     filenames = os.listdir(dataset_dir)
        #     filenames = [file for file in filenames if '.jpg' in file]
        #     filenames = sorted(filenames, key=lambda x: x.split('.')[0])
        dataset_dir = join(EXP_DIR, 'synthetic_data', 'test')
        print("taking data from ", dataset_dir)
        time.sleep(2)

        filenames = os.listdir(dataset_dir)
        filenames = [file for file in filenames if '.jpg' in file]
        filenames = sorted(filenames, key=lambda x: x.split('.')[0])
        for ii, file in enumerate(filenames):
            # if not ii % 1 == 0:
            #     continue
            # # Load image and ground truth data
            # image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            #     modellib.load_image_gt(dataset, inference_config,
            #                            image_id, use_mini_mask=False)
            # molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
            # # Run object detection
            # results = model.detect([image], verbose=0)
            # r = results[0]
            # # Compute AP
            # AP, precisions, recalls, overlaps =\
            #     utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
            #                      r["rois"], r["class_ids"], r["scores"], r['masks'])
            # APs.append(AP)

            image = plt.imread(os.path.join(dataset_dir, file))
            results = model.detect([image], verbose=1)
            fig, ax = get_ax()
            r = results[0]
            ax = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    dataset.class_names, r['scores'], ax=ax, colors=colors)
            save_path = os.path.join(EXP_DIR, 'runs', args.runname)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig.savefig(os.path.join(save_path, '{}.jpg'.format(file.strip('.jpg'))))
            plt.close()
    print("wrote data to", save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--runname', type=str, default='test')
    args = parser.parse_args()
    main(args)

