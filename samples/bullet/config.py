import os
import sys
from os.path import join

class GeneralConfig(object):
	# Local path to trained weights file
	# Root directory of the project

	# Setup paths
	EXP_NAME = 'four_objects'

	ROOT_PATH = os.path.abspath("/home/msieb/projects/Mask_RCNN")
	EXP_DIR = join(ROOT_PATH, 'experiments', EXP_NAME)
	MODEL_DIR = os.path.join(EXP_DIR, "logs")
	DATASET_DIR = join(EXP_DIR, 'synthetic_data')
	OLD_MODEL_PATH = None
	COCO_MODEL_PATH = os.path.join(ROOT_PATH, "mask_rcnn_coco.h5")
	
	WEIGHTS_PATH = "/home/msieb/projects/Mask_RCNN/experiments/four_objects/training_logs/bullet20181009T1257/"  # TODO: update this path
	WEIGHTS_FILE = 'mask_rcnn_bullet_0012.h5'
	WEIGHTS_FILE_PATH = join(WEIGHTS_PATH, WEIGHTS_FILE)

	# Mask RCNN config specific and specific for experiment
	CLASS_NAMES = ['cube_white', 'bowl', 'cube_red', 'duck']
	CLASS_IDS = [i for i in range(1, len(CLASS_NAMES) + 1)]  # ascending indexing starting from 1
	
	# Bullet specific
	BULLET_CLASS_IDS = [5, 4, 6, 7] # To load labels correcty - comes from bullet interface not and MRCNN specific
	CLASS_ID_MAPPING = {key: val for key, val in zip(BULLET_CLASS_IDS, CLASS_NAMES)}

	# ADD BG next time!!!!
	RELEVANT_IDS = CLASS_NAMES
	CLASS_NAMES_W_BG = ['BG'] + CLASS_NAMES
