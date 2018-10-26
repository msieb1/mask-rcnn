import numpy as np
import os
from os.path import join
import argparse
import json
from ipdb import set_trace
from copy import deepcopy as copy

# Call: python mask_id_assigner.py -d /home/msieb/projects/gps-lfd/demo_data/cube -s 0

NEW_IDS = {"bowl": "4", "duck": "7", "cube_white": "5", "cube_red": "6"}

def swap_mask_ids(mask, original_ids, new_ids):
    new_mask = copy(mask)
    for key in original_ids:
        new_mask[np.where(mask == int(original_ids[key]))] = int(new_ids[key])
    return new_mask


def main(args):
    print("load masks from ", join(args.datapath, 'masks'))
    print("write new masks to", join(args.datapath, 'masks', 'new_masks'))
    if not os.path.exists(join(args.datapath, 'masks', 'new_masks')):
        os.makedirs(join(args.datapath, 'masks', 'new_masks'))

    filenames = [file for file in os.listdir(join(args.datapath, 'masks')) if 'mask.npy' in file]
    filenames = sorted(filenames, key=lambda x: x.split('.')[0])
    with open('{}/{}_relevant_ids_names.json'.format(args.datapath, args.seqname), 'r') as f:
        original_ids_w_names = json.load(f)
    for file in filenames:
        cur_mask = np.load(join(args.datapath, 'masks', file))
        new_mask = swap_mask_ids(cur_mask, original_ids_w_names, NEW_IDS)
        np.save(join(args.datapath, 'masks','new_masks',file), new_mask)

    for key, val in original_ids_w_names.items():
        original_ids_w_names[key] = NEW_IDS[key]

    with open('{}/{}_relevant_ids_names.json'.format(join(args.datapath, 'masks', 'new_masks'),args.seqname), 'w') as f:
        json.dump(original_ids_w_names, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datapath', type=str, required=True)
    parser.add_argument('-s', '--seqname', type=str, required=True)

    args = parser.parse_args()
    main(args)

