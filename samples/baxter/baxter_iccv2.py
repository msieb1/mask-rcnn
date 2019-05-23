"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import math
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage

# Root directory of the project
ROOT_DIR = os.path.abspath("/home/zhouxian/projects/Mask_RCNN/")
DATASET_DIR = '/media/data/iccv2019/synthetic_data_generation/iccv2019_flipping/synthetic_data'
MODEL_DIR = '/media/data/iccv2019/synthetic_data_generation/iccv2019_flipping/mask_rcnn_models'
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class BaxterConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy baxter dataset.
    """
    # Give the configuration a recognizable name
    NAME = "baxter"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    # CLASS_NAMES = ['BG', 'hexagon', 'rings', 'squares', 'mugs', 'cans', 'hand', 'robot']
    # TARGET_IDS = [1, 2, 3, 4, 5, 6, 7]
    CLASS_NAMES = ['BG', 'mugs', 'cans', 'hand', 'robot']
    TARGET_IDS = [1, 2, 3, 4]    


    # Number of classes (including background)
    NUM_CLASSES = 1 + 4 # background + 7 objects

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    # IMAGE_MIN_DIM = 128
    # IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000
    BACKBONE = "resnet50"
    MAX_GT_INSTANCES = 50
    POST_NMS_ROIS_INFERENCE = 500
    POST_NMS_ROIS_TRAINING = 1000
    RPN_TRAIN_ANCHORS_PER_IMAGE = 512
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
config = BaxterConfig()
config.display()

class InferenceConfig(BaxterConfig):
    MODEL_PATH = "/home/zhouxian/projects/Mask_RCNN/logs/baxter20190315T0139/mask_rcnn_baxter_0002.h5"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # CLASS_NAMES = ['BG', 'hexagon', 'rings', 'squares', 'mugs', 'cans', 'hand', 'robot']
    # TARGET_IDS = [1, 2, 3, 4, 5, 6, 7]
    CLASS_NAMES = ['BG', 'mugs', 'cans', 'hand', 'robot']
    TARGET_IDS = [1, 2, 3, 4]    

class BaxterDataset(utils.Dataset):
    """Generates the baxter synthetic dataset. The dataset consists of simple
    baxter (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_baxter(self, dataset_dir, subset):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        # self.add_class("baxter", 1, "hexagon")
        # self.add_class("baxter", 2, "rings")
        # self.add_class("baxter", 3, "squares")
        self.add_class("baxter", 1, "mugs")
        self.add_class("baxter", 2, "cans")
        self.add_class("baxter", 3, "hand")
        self.add_class("baxter", 4, "robot")

        # self.add_class("baxter", 1, "hexagon")
        # self.add_class("baxter", 2, "rings")
        # self.add_class("baxter", 3, "squares")
        # self.add_class("baxter", 4, "mugs")
        # self.add_class("baxter", 5, "cans")
        # self.add_class("baxter", 6, "hand")
        # self.add_class("baxter", 7, "robot")
        # self.idx2class = {1: 'green_ring', 2: 'blue_ring'} # ,3: 'hand', 4: 'robot'}
        # self.class2idx = {val: key for key, val in self.idx2class.items()}

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        filenames = os.listdir(dataset_dir)
        filenames = [file for file in filenames if '.jpg' in file or '.png' in file]

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
        
    
            # Synthetic image
            _split = filename.split('_')
            n_added_objects = int((len(_split) / 3))
            classes = []
            for i in range(n_added_objects):
                classes.append(_split[3 * i])
            self.add_image(
                "baxter",
                image_id=filename,  # use file name as a unique image id
                path=image_path,
                width=width, height=height, classes=classes)  

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "baxter":
            return super(self.__class__, self).load_mask(image_id)
        # print("image info: {}".format(image_info))
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        # mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
        #                 dtype=np.uint8) 
        image_path = info['path']
        image = skimage.img_as_ubyte(plt.imread(image_path))
        return image

    def image_reference(self, image_id):
        """Return the baxter data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "baxter":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


    def load_mask(self, image_id):
        """Generate instance masks for baxter of the given image ID.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "baxter":
            return super(self.__class__, self).load_mask(image_id)
        # print("image info: {}".format(image_info))
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        # mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
        #                 dtype=np.uint8) 
        image_path = info['path']
        image_ = plt.imread(image_path)
        mask_path = image_path.split('.')[0] + '.npy'
        mask = np.load(mask_path)
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
            mask_ids = np.ones([mask.shape[-1]], dtype=np.int32) * self.class_names.index(info["classes"][0])
        else:
            # Synthetic image
            mask_ids = np.ones([mask.shape[-1]], dtype=np.int32)
            for i in range(mask.shape[-1]):
                mask_ids[i] = self.class_names.index(info["classes"][i]) 
        # print(info["classes"])
        # mask_ids =  np.ones([mask.shape[-1]], dtype=np.int32)

        # print("class id: ", self.class2idx[info["classes"]])
        # print(type(self.class2idx[info["classes"]]))       # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # print ("mask: {}".format(np.where(mask.astype(np.bool) == True)))
        # print("class ids: {}".format(mask_ids))
        return mask.astype(np.bool), mask_ids
