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
import tensorflow as tf
import torch
from inception import inception_v3

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

from baxter import BaxterConfig, BaxterDataset, DATASET_DIR, MODEL_DIR

from ipdb import set_trace
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

WEIGHTS_PATH = "/home/msieb/projects/Mask_RCNN/weights/baxter20180707T1715/mask_rcnn_baxter_0010.h5"  # TODO: update this path


# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
# DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--runname', type=str, required=True)
args = parser.parse_args()

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return fig, ax

class InferenceConfig(BaxterConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()
os.environ["CUDA_VISIBLE_DEVICES"]="1, 2"

DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR,
                              config=inference_config)

model.load_weights(WEIGHTS_PATH, by_name=True)

dataset = BaxterDataset()
dataset.load_baxter(DATASET_DIR, "train")
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
    for seq_name in os.listdir('/home/msieb/projects/Mask_RCNN/datasets/baxter/test_2'):
        print("Processing ", seq_name)
    # seq_name = args.seqname
        dataset_dir = os.path.join(DATASET_DIR, "test_2" )
        filenames = os.listdir(dataset_dir)
        filenames = [file for file in filenames if '.jpg' in file]
        filenames = sorted(filenames, key=lambda x: x.split('.')[0])

        for ii, file in enumerate(filenames):
            if not ii % 1 == 0:
                continue
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
            encountered_ids = []
            filtered_inds = []
            target_ids = [2,4]
            for i, box in enumerate(r['rois']):
                # ROIs are sorted after confidence, if one was registered discard lower confidence detections to avoid double counting
                class_id = r['class_ids'][i]
                if class_id not in target_ids or class_id in encountered_ids:
                    continue
                encountered_ids.append(class_id)
                # all_visual_features_unordered.append(r['roi_features'][i])
                filtered_inds.append(i)
            filtered_inds = np.array(filtered_inds)

            ax = visualize.display_instances(image, r['rois'][filtered_inds], r['masks'][:, :, filtered_inds], r['class_ids'][filtered_inds],            
                        dataset.class_names, r['scores'][filtered_inds], ax=ax, colors=colors)
            save_path = os.path.join('/home/msieb/projects/Mask_RCNN/katerina', args.runname, seq_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, '{0:05d}.jpg'.format(ii)), transparent = True, bbox_inches = 'tight', pad_inches = 0)
            plt.close()