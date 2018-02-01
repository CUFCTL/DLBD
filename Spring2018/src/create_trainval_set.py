"""
Create training and validation sets

Usage:
    NOTE: Execute from ci-models directory
    python create_trainval_set.py \
        --out_path <ci_devkit directory (CIdevkit)> \
        --train 0.75 --val 0.25
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, os
from halo import Halo

def parse_args():
    """
    Parse command line arguments.

    Parameters:
        None
    Returns:
        parser arguments
    """
    parser = argparse.ArgumentParser(
        description='Create Train/Val Split')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required_arguments')
    required.add_argument('--out_path',
        dest='out_path',
        help='CI Devkit directory (CIdevkit)')
    required.add_argument('--train',
        dest='train',
        help='Percentage for training set (i.e. 0.75, 0.80, etc.)',
        type=float,
        default=0.75)
    required.add_argument('--val',
        dest='val',
        help='Percentage for validation set (i.e. 0.25, 0.20, etc.)',
        type=float,
        default=0.25)
    parser._action_groups.append(optional)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args() # Command line parameters

    if not (float(args.train) + float(args.val) == 1):
        print('Training and validation set percentages must equal 1 (i.e. 0.75/0.25)')
        exit(0)

    spinner = Halo(text='Converting', spinner='dots')
    spinner.start()

    if not os.path.exists('data'):
        os.makedirs('data')

    # Determine modulo for training and validation set
    train, val, idx = 20 * args.train, 20*args.val, 0
    f_train = open(os.path.join('data', 'train.txt'), 'w')
    f_val = open(os.path.join('data', 'val.txt'), 'w')

    # Iterate through all images and create training and validation sets (files)
    for img in sorted(os.listdir(os.path.join(args.out_path, 'CI2018', 'JPEGImages'))):
        img_name = img.split('.')[0]
        if idx % 20 < 15:
            f_train.write(img_name + '\n')
        else:
            f_val.write(img_name + '\n')
        idx += 1

    spinner.stop()

    # Close training and validation set files
    f_train.close()
    f_val.close()