import argparse
import os
from os.path import join as join
from PIL import Image
from copy import deepcopy as copy
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
import time
# plt.ion()
from ipdb import set_trace


ROOT_PATH = '/home/msieb/test_folder/'
BACKGROUND_IDX = '0'




DEPTH_TH = 5 # Minimum depth difference after BS
DEPTH_MIN = 0 # General minimum depth (to exclude erroneous depth values or out of bounds (zeroed out))
DEPTH_MAX = 80
RGB_TH = 25 # Minimum rgb difference after BS

H = 480 
W = 640

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, default='', help='train, valid or test')
parser.add_argument('-t', '--target', type=str, required=True)
args = parser.parse_args()

BACKGROUND_RGB_0 = cv2.imread(join(ROOT_PATH, args.target, 'background/videos/0_view0/00000.jpg')) 
BACKGROUND_DEPTH_0 = cv2.imread(join(ROOT_PATH, args.target, 'background/depth/0_view0/00000.jpg'))
BACKGROUND_RGB_1 = cv2.imread(join(ROOT_PATH, args.target, 'background/videos/0_view1/00000.jpg')) 
BACKGROUND_DEPTH_1 = cv2.imread(join(ROOT_PATH, args.target, 'background/depth/0_view1/00000.jpg'))
BACKGROUND_RGB_2 = cv2.imread(join(ROOT_PATH, args.target, 'background/videos/0_view2/00000.jpg')) 
BACKGROUND_DEPTH_2 = cv2.imread(join(ROOT_PATH, args.target, 'background/depth/0_view2/00000.jpg'))

def main(args):
    datapath = join(ROOT_PATH, args.target)
    # process_images(args.filepath)
    # depth_subtraction(args.filepath)
    for file in os.listdir(join(datapath, 'videos')):
        if file[0] == BACKGROUND_IDX or file.endswith('.mp4'):
            continue # skip background subtraction with its own background image
        grabcut(datapath, file)


def process_images(filepath):

    fgbg = cv2.createBackgroundSubtractorMOG2()  
    filepath = join(filepath, 'videos')
    for file in os.listdir(filepath):
        if file[-4:] != '.jpg':
            continue
        print("Processing current image {}".format(file))
        file_path = join(ROOT_PATH, 'videos', file)
        try:
            frame = cv2.imread(file_path)
        except:
            print("Image could not be opened - check path")

        fgmask = fgbg.apply(frame)
        cv2.imshow('frame',frame)
        cv2.imshow('mask',fgmask)
        k = cv2.waitKey(400)

    cv2.destroyAllWindows()

def get_mask(cond):
    return np.where(cond, np.ones([H, W]), np.zeros([H, W])).astype(np.float32)

def get_preprocessed_data(data_path, seqname, fr):
    clipping_distance = 1.5
    depth_path = join(data_path, 'depth', seqname, '{0:05d}.jpg'.format(fr))
    rgb_path = join(data_path, 'videos', seqname, '{0:05d}.jpg'.format(fr))
    rgb = cv2.imread(rgb_path)
    depth_img = cv2.imread(depth_path)
    depth_img = ndi.filters.gaussian_filter(depth_img, (7, 7, 0), order=0)
    # depth = depth_img / 255.0 * clipping_distance        
    # depth *= 0.0010000000474974513

    
    view = seqname[-1]
    if view == '0':
        background_rgb = BACKGROUND_RGB_0 
        background_depth = BACKGROUND_DEPTH_0
    elif view == '1':
        background_rgb = BACKGROUND_RGB_1 
        background_depth = BACKGROUND_DEPTH_1
    elif view == '2':
        background_rgb = BACKGROUND_RGB_2
        background_depth = BACKGROUND_DEPTH_2
    else:
        print("invalid view specified")
        return
    rgb_fg = rgb.astype(np.float) - background_rgb.astype(np.float)
    depth_fg = depth_img.astype(np.float) - background_depth.astype(np.float)

    #plt.imshow(rgb_fg)
    valid_rgb = get_mask(np.max(np.abs(rgb_fg), axis=2) > RGB_TH).astype(np.uint8)
    valid_depth = get_mask((np.abs(depth_fg[:, :, 0]) > DEPTH_TH) * \
                (np.abs(depth_img[:, :, 0]) > DEPTH_MIN) * \
                (np.abs(depth_img[:, :, 0]) < DEPTH_MAX)).astype(np.uint8)
    # valid = ndimage.binary_erosion(valid).astype(np.float32)
    # rgb = rgb.astype(np.float32)
    # rgb = rgb/255.0 - 0.5
    # rgb = np.reshape(rgb, [1, H, W, 3])

    # depth = np.expand_dims(depth, axis=0)
    # depth = depth.astype(np.float32)
    # depth = np.reshape(depth, [1, H, W, 1])
    return rgb, depth_img, valid_rgb, valid_depth

def grabcut(datapath, seqname):
    nFrames = 100
    save_path = join(datapath, 'masks')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("Segment sequence: ", seqname)

    for fr in np.arange(0, nFrames, 1):    
        try:
            rgb, depth, valid_rgb, valid_depth = get_preprocessed_data(datapath, seqname, fr)
        except:
            break
        if os.path.exists(join(save_path, '{0}_{1:05d}.jpg'.format(seqname, fr))):
            continue
        print("frame %d" % fr)

        img = rgb
        mask = np.zeros(img.shape[:2],np.uint8)
        # Make mask generation easier> 3 means possibly foreground, 1 means definitely foreground
        mask[np.where(valid_rgb)] = 3 
        mask[np.where(valid_depth)] = 3 
        # mask[np.where(valid_depth * valid_rgb)] = 1 

        # mask[np.where(valid == False)] = 2
        mask[:100, 0:30] = 0
        mask[-100:, 0:30] = 0
        # mask[:200, 0:50] = 0
        mask[-100:, -30:] = 0
        mask[:100, -30:] = 0
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (50,50, 480, 480)
        mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,7,cv2.GC_INIT_WITH_MASK)
        mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        rps = sorted((regionprops(label(mask > 0.5, background=0))), key=lambda x: x.area, reverse=True)
        mask_clean = np.zeros(mask.shape)
        mask_clean[rps[0].coords[:, 0], rps[0].coords[:,1]] = 1
        # mask_clean = ndi.binary_fill_holes(mask_clean, structure=np.ones((2,2))).astype(np.uint8)
        mask_clean_ = copy(mask_clean)
        mask_clean = ndi.binary_fill_holes(mask_clean_).astype(np.uint8)
        img_masked = img*mask_clean[:,:,np.newaxis]
        np.save(join(save_path, '{0}_{1:05d}'.format(seqname, fr)), mask_clean)
        cv2.imwrite(join(save_path, '{0}_{1:05d}.jpg'.format(seqname, fr)), img)
        cv2.imwrite(join(save_path, '{0}_{1:05d}_masked.jpg'.format(seqname, fr)), img_masked.astype(np.uint8)[:, :, :])
        # cv2.imshow('frame',rgb.astype(np.uint8))
        # vis_mask = ndi.binary_erosion(mask)
        # cv2.imshow('mask', mask_clean*255)
        # cv2.imshow('img',img_masked.astype(np.uint8)[:, :, :])
        
        # k = cv2.waitKey(400)
        # time.sleep(1)
    print("="*20)
    cv2.destroyAllWindows()


def depth_subtraction(datapath):
    nFrames = 100
    for fr in range(nFrames):
        rgb, depth, mask = get_preprocessed_data(datapath, fr)
        mask = np.stack((mask, mask, mask), axis=2)
        masked = np.multiply(rgb, mask).astype(np.uint8)
        cv2.imshow('frame',rgb.astype(np.uint8))
        cv2.imshow('mask',masked.astype(np.uint8)[:, :, :])
        k = cv2.waitKey(20)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    try:
        main(args)
    except KeyboardInterrupt:
        # do nothing here
        cv2.destroyAllWindows()
