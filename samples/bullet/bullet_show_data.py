import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
from os.path import join
ROOT_DIR = '/home/msieb/projects/Mask_RCNN/'
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from bullet import BulletConfig, BulletDataset
from ipdb import set_trace
# Root directory of the project
if ROOT_DIR.endswith("samples/bullet"):
    # Go up two levels to the repo root
    ROOT_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))

from config import GeneralConfig
gconf = GeneralConfig()
DATASET_DIR, MODEL_DIR, OLD_MODEL_PATH, COCO_MODEL_PATH, WEIGHTS_FILE_PATH, EXP_DIR  \
        = gconf.DATASET_DIR, gconf.MODEL_DIR, gconf.OLD_MODEL_PATH, gconf.COCO_MODEL_PATH, gconf.WEIGHTS_FILE_PATH, gconf.EXP_DIR

# Import Mask RCNN
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mrcnn'))
import utils
import visualize
from visualize import display_images
import model as modellib
from model import log

import bullet

config = BulletConfig()

# Load dataset
# Get the dataset from the releases page
# https://github.com/matterport/Mask_RCNN/releases
dataset = BulletDataset()
dataset.load_bullet( join(EXP_DIR, 'synthetic_data'), "train")

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))



# Load and display random samples
image_ids = np.random.choice(dataset.image_ids, 10)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)