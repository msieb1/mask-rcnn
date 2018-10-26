from __future__ import division


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

from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50

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
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# WEIGHTS_PATH = "/home/msieb/projects/Mask_RCNN/weights/baxter20180707T1715/mask_rcnn_baxter_0019.h5"  # TODO: update this path
WEIGHTS_PATH = "/home/msieb/projects/Mask_RCNN/logs/baxter20180710T2016/mask_rcnn_baxter_0016.h5"  # TODO: update this path

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
# DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--runname', type=str, required=True)
args = parser.parse_args()


class InferenceConfig(BaxterConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 5 # background + 3 objects

    def load_baxter(self, dataset_dir, subset):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("baxter", 1, "blue_ring")
        self.add_class("baxter", 2, "green_ring")
        self.add_class("baxter", 3, "tower")
        self.add_class("baxter", 4, "hand")
        self.add_class("baxter", 5, "robot")
        # self.idx2class = {1: 'green_ring', 2: 'blue_ring'} # ,3: 'hand', 4: 'robot'}
        # self.class2idx = {val: key for key, val in self.idx2class.items()}

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        filenames = os.listdir(dataset_dir)
        filenames = [file for file in filenames if '.jpg' in file]

        # Add images
        for i, filename in enumerate(filenames):
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, filename)
            height = 480
            width = 640
            if len(filename.split('_')) == 3:
                try:
                    label_path = os.path.join(dataset_dir, '_' + filename[0] + '.txt')
                    with open(label_path, 'r') as fp:
                        line = fp.readline().strip('\n')
                    classes = [line]
                except:
                    classes =[-1]
                self.add_image(
                    "baxter",
                    image_id=filename,  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height, classes=classes)
            else:
                # # Synthetic image
                # label_path_1 = os.path.join(dataset_dir, '_' + filename[0] + '.txt')
                # label_path_2 = os.path.join(dataset_dir, '_' + filename.split('_')[3] + '.txt')
                # try:
                #     with open(label_path_1, 'r') as fp:
                #         line = fp.readline().strip('\n')
                #     classes = [line]
                #     with open(label_path_2, 'r') as fp:
                #         line = fp.readline().strip('\n')
                #     classes.append(line)
                # except:
                #     classes = [-1]
                # self.add_image(
                #     "baxter",
                #     image_id=filename,  # use file name as a unique image id
                #     path=image_path,
                #     width=width, height=height, classes=classes)  
                # Synthetic image
                _split = filename.split('_')
                n_added_objects = int((len(_split) - 2) / 3)
                label_path = os.path.join(dataset_dir, '_' + filename[0] + '.txt')
                with open(label_path, 'r') as fp:
                    line = fp.readline().strip('\n')
                classes = [line]
                for i in range(n_added_objects):
                    label_path = os.path.join(dataset_dir, '_' + filename.split('_')[3 + i*3] + '.txt')
                    with open(label_path, 'r') as fp:
                        line = fp.readline().strip('\n')
                    classes.append(line)
                self.add_image(
                    "baxter",
                    image_id=filename,  # use file name as a unique image id
                    path=image_path,
                   width=width, height=height, classes=classes)  

inference_config = InferenceConfig()

with tf.device('/device:GPU:1'):
    model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR,
                              config=inference_config)
    model.load_weights(WEIGHTS_PATH, by_name=True)

# with tf.device('/device:GPU:0'):
#     feature_extractor = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
#    feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')


dataset = BaxterDataset()
dataset.load_baxter(DATASET_DIR, "train")
dataset.prepare()
colors = visualize.random_colors(10)
#with torch.cuda.device(0):
    # device = torch.device("cuda:1")
 #   convnet = inception_v3(pretrained=True)

if FROM_DATASET:
    # # Test on a random image
    # Validation dataset

    image_ids = dataset.image_ids
    APs = []
    for image_id in image_ids:

        image = dataset.load_image(image_id)

        results = model.detect([image], verbose=1)

        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], ax=visualize.get_ax()[1], colors=colors)

else:
    for seq_name in os.listdir('/home/msieb/projects/Mask_RCNN/datasets/baxter/test'):
        print("Processing ", seq_name)
    # seq_name = args.seqname
        dataset_dir = os.path.join(DATASET_DIR, "test", seq_name)
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
            curr_time = time.time()
            results = model.detect([image], verbose=0)
            r = results[0]
            rois = r['roi_features']

            fig, ax = visualize.get_ax()
            ax = visualize.display_instances(image, r['rois_enlarged'], r['masks'], r['class_ids'], 
                                    dataset.class_names, r['scores'], ax=ax, colors=colors)
            save_path = os.path.join('/home/msieb/projects/Mask_RCNN/demos', args.runname, seq_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig.savefig(os.path.join(save_path, '{0:05d}.jpg'.format(ii)))

            # for ii in range(len(rois)):
            # fig, ax = visualize.get_ax()
            # plt.imshow(np.max(rois[0], axis=2), cmap='YlGnBu', interpolation='nearest')
            plt.close()
            # encountered_ids = []
            # all_inputs = []
            # all_features = dict()
            # for i, box in enumerate(r['rois']):
            #     class_id = r['class_ids'][i]
            #     if class_id in encountered_ids:
            #         continue
            #     encountered_ids.append(class_id)
            #     cropped = utils.crop_box(image, box, y_offset=20, x_offset=20)
            #     cropped = utils.resize_image(cropped, max_dim=299)[0]
            #     all_inputs.append(cropped)
            # all_inputs = np.asarray(all_inputs)
            # features = feature_extractor.predict(all_inputs)
            # for i in range(all_inputs.shape[0]):
            #     all_features[class_id] = np.squeeze(features[i])
            # print("run time: ", time.time() - curr_time)
