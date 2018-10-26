import imageio
import os
from pdb import set_trace
from os.path import join
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, required=True)
parser.add_argument('-m', '--mode', type=str, default='')
parser.add_argument('-o', '--one-frame', type=bool, default=False)
args = parser.parse_args()

ROOT_PATH = '/home/msieb/test_folder/'
EXP_NAME = args.name
INPUT_PATHS = [join(ROOT_PATH, EXP_NAME, 'videos/', args.mode), join(ROOT_PATH, EXP_NAME, 'depth/', args.mode)]

def main(args):
    for path in INPUT_PATHS:
           #OUTDIR='/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/tcn_data/experiments/mask_rcnn/mask_estimation_' \
            #               + args.mode + '/' + path.split('/')[0] 
            # OUTDIR = ROOT_PATH  + path.split('/')[0] + '/' + path.split('/')[1]

            # OUTDIR='/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/tcn_data/experiments/mask_rcnn/background'  + '/' + path.split('/')[0] 
            print("extracting {}".format(path))
            for file in os.listdir(path):
                if not file.endswith('.mp4'): # or 'view1' in file:
                    continue
                reader = imageio.get_reader(join(path, file))
                dest_folder = join(path, file.split('.mp4')[0])

                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder)
                print("Extract {}".format(file))
                print("="*10)
                for i, im in enumerate(reader):
                    if args.one_frame:
                        if i > 0  and not args.mode == 'test':
                            break
                    imageio.imwrite(os.path.join(dest_folder, "{0:05d}.jpg".format(i)), im)
                else:
                    continue

if __name__ == '__main__':
    main(args)
