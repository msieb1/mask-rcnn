import argparse
import os
from os.path import join
import sys
import numpy as np
import cv2
from copy import deepcopy as copy
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
import random
from numpy.random import randint
import importlib
from ipdb import set_trace
import time
plt.ion()


## EXAMPLE USAGE ####
# python create_synthetic_images.py -i /home/msieb/projects/gps-lfd/demo_data/four_objects -e four_objects -m train
# SET EXPNAME IN CONFIG.PY



def main(args):
    module = importlib.import_module('experiments.' + args.experiment + '.configs')
    conf = getattr(module, 'Config')
    gen = SyntheticImageGenerator(args.input_root, args.experiment, args.mode, conf)
    gen.create_synthetic_images()

class SyntheticImageGenerator(object):

    def __init__(self, input_root, experiment, mode, conf):
        self.mask_path = join(input_root, 'masks')
        self.rgb_path = join(input_root, 'rgb')
        self.experiment = experiment
        self.conf = conf
        self.output_path = join(conf.EXP_DIR, 'synthetic_data', mode)
        print("write to: ",self.output_path)
        time.sleep(3)
        self.relevant_ids = self.conf.BULLET_CLASS_IDS

    def create_synthetic_images(self, n_iter=100):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        input_list = os.listdir(self.rgb_path)
        input_list = sorted(input_list, key=lambda x: x.split('.')[0])
        for itr in range(n_iter):
            for file_base in input_list:
                if not (file_base.endswith('.jpg') or file_base.endswith('.png')) or 'masked' in file_base:
                    continue
                curr_seqname = "".join([file_base.split('_')[0], '_'])
                added_list = [file for file in os.listdir(self.rgb_path) if (file.endswith('.jpg') or file.endswith('.png')) and not 'masked' in file]
                random.shuffle(added_list)
                print("Base image: ", file_base)

                # Overlay images to get synthetic image
                img_overlayed, mask_overlayed, mask_labels, save_name = self.make_synthetic_image(file_base, added_list, n_added_images=randint(1, 2))

                # Save to file
                print("saved as ", save_name )
                print("="*20)

                cv2.imwrite(join(self.output_path, save_name + '.jpg'), img_overlayed)
                np.save(join(self.output_path, save_name + '.npy'), mask_overlayed)
                np.save(join(self.output_path, save_name + '_labels.npy'), mask_labels)

                cv2.imshow('img_overlayed',img_overlayed)

                k = cv2.waitKey(1)

    def make_synthetic_image(self, file_base, list_of_img_paths, n_added_images=1):

        img_base = cv2.imread(join(self.rgb_path, file_base))  
        #img_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2RGB) 

        # Store mask labels for later training, i.e. stores the corresponding object label for every mask channel
        mask_labels = []
        if file_base.endswith('.jpg'):
            mask_base = np.load(join(self.mask_path, file_base.strip('.jpg') + '_mask.npy'))
            save_name = file_base.strip('.jpg') 
        elif file_base.endswith('.png'):
            mask_base = np.load(join(self.mask_path, file_base.strip('.png') + '_mask.npy'))
            save_name = file_base.strip('.png')
        else:
            print("provided invalid image format, only supports png or jpg")
            return

        mask_overlayed = np.zeros(mask_base.shape, dtype=np.uint8)[:, :, None]

        # Transform masks such that each channel is 0'1'ed for specific object (right now its one channel with multiple integers)
        for Id in self.relevant_ids:
            curr_obj_mask = np.zeros(mask_base.shape, dtype=np.uint8)
            inds = mask_base == Id
            # Break if object isnt found
            if np.sum(inds) == 0:
                continue
            mask_labels.append(Id)
            curr_obj_mask[np.where(inds)] = 1
            mask_overlayed = np.concatenate([mask_overlayed, \
                                    curr_obj_mask[:, :, np.newaxis]], axis=2) 
        # Get rid of placeholder channel entry
        mask_overlayed = mask_overlayed[:, :, 1:]

        if len(mask_overlayed.shape) < 3:
            mask_overlayed = mask_overlayed[:, :, np.newaxis]
        img_overlayed = copy(img_base)
       
        for i in range(n_added_images):
             # Read image to be added on top
            idx = randint(0, len(list_of_img_paths))
            file_added = list_of_img_paths[idx]
            print("Added image: ", file_added)

            img_added = cv2.imread(join(self.rgb_path, file_added))
            #img_added = cv2.cvtColor(img_added, cv2.COLOR_BGR2RGB) 
            if file_base.endswith('.jpg'):
                mask_added = np.load(join(self.mask_path, file_added.strip('.jpg') + '_mask.npy'))
            else:
                mask_added = np.load(join(self.mask_path, file_added.strip('.png') + '_mask.npy'))
            
            # Make binary masks
            mask_added_bin = np.zeros(mask_added.shape, dtype=np.uint8)
            while True:
                # Only add one object of loaded mask for convenience
                chosen_id = random.choice(self.relevant_ids)
                mask_added_bin[np.where(mask_added == chosen_id)] = 1
                # Check if object contained in image
                if np.sum(np.where(mask_added == chosen_id)) > 0:
                    break
            mask_labels.append(chosen_id)

            # Mask image
            img_added_masked = img_added * mask_added_bin[:,:,np.newaxis]

            # Augment masks
            img_added_masked, mask_added_bin = self.translate_mask(img_added_masked, mask_added_bin, \
                                                            row_shift=randint(-self.conf.MAX_SHIFT_ROW, self.conf.MAX_SHIFT_ROW), \
                                                            col_shift=randint(-self.conf.MAX_SHIFT_COL, self.conf.MAX_SHIFT_COL))
            img_added_masked, mask_added_bin = self.rotate_mask(img_added_masked, mask_added_bin, \
                                                            angle=randint(0,361,1), center=None, \
                                                            scale=np.random.uniform(0.4, 1.6))
            img_added_masked, mask_added_bin = self.perturb_intensity(img_added_masked, mask_added_bin, scale=np.random.uniform(0.7,1.0))

            # Apply masks
            img_overlayed[np.where(mask_added_bin == 1)] = img_added_masked[np.where(mask_added_bin == 1)]
            for j in range(mask_overlayed.shape[-1]):
                mask_overlayed[:, :, j] *= np.logical_not(mask_added_bin)
            mask_overlayed = np.concatenate([mask_overlayed, \
                                    mask_added_bin[:, :, np.newaxis]], axis=2)  
            # Save image and mask
            if file_base.endswith('.jpg'):
                save_name += '_' + file_added.strip('.jpg') 
            else:
                save_name += '_' + file_added.strip('.png') 

        save_name += '-0' 
        if os.path.exists(join(self.output_path, save_name + '.jpg')):
            index = int(save_name.split('-')[-1][0])
            save_name = save_name.split('-')[0] + '-' + str(index + 1)

        return img_overlayed, mask_overlayed, np.squeeze(np.asarray(mask_labels)), save_name


    def overlay_img(self, img_base, mask_base, img_added_masked, mask_added):
        img_overlayed = copy(img_base)
        img_overlayed[np.where(mask_added == 1)] = img_added_masked[np.where(mask_added == 1)]

        mask_overlayed = copy(mask_base)
        mask_overlayed *= np.logical_not(mask_added)
        mask_overlayed = np.concatenate([mask_overlayed[:, :, np.newaxis], \
                                        mask_added[:, :, np.newaxis]], axis=2)
        return img_overlayed, mask_overlayed

    def perturb_intensity(self, img_masked, mask, scale=0):
        img_perturbed = copy(img_masked)
        img_perturbed = (img_perturbed * scale).astype(np.uint8)
        img_perturbed[np.where(img_perturbed > 255)] = 255
        img_perturbed[np.where(img_perturbed < 0)] = 0
        return img_perturbed, mask

    def translate_mask(self, img_masked, mask, row_shift=0, col_shift=0):
        mask_shifted = shift(mask, [row_shift, col_shift, ])
        img_masked_shifted = shift(img_masked, [row_shift, col_shift, 0])
        return img_masked_shifted, mask_shifted

    def rotate_mask(self, img_masked, mask, angle=0, center=None, scale=1.0):
        # grab the dimensions of the image
        (h, w) = img_masked.shape[:2]

        # if the center is None, initialize it as the center of
        # the image
        if center is None:
            center = (w // 2, h // 2)

        # perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated_img = cv2.warpAffine(img_masked, M, (w, h))
        rotated_mask = cv2.warpAffine(mask, M, (w, h))

        # return the rotated image
        return rotated_img, rotated_mask


    # def translate(image, x, y):
    #     # define the translation matrix and perform the translation
    #     M = np.float32([[1, 0, x], [0, 1, y]])
    #     shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    #     # return the translated image
    #     return shifted




    def get_masked_img(self, img, mask):
        img_masked = img*mask[:,:,np.newaxis]
        return img_masked


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-root', type=str, required=True, help='data folder for input')
    parser.add_argument('-e', '--experiment', type=str, required=True, help='experiment name')
    parser.add_argument('-m', '--mode', type=str, default='train', help='train, valid or test')
    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        # do nothing here
        cv2.destroyAllWindows()

