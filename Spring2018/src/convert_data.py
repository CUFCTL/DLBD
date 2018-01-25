"""
Convert DAC dataset to VOC format for training with 
Tensorflow Object Detection API.

Usage:
	NOTE: Run from CUFCTL-Track directory
	python python/convert_data.py \
		--dac_path <path to data_training DAC directory> \
		--out_path <path to new data (DACdevkit) \
		--type <13 or 100>
"""

# Import necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import read_dac_xml, make_voc_directories
from utils import copy_images, convert_xml, write_pbtxt

import argparse
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
        description='Convert CI data to VOC format')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required_arguments')
    required.add_argument('--ci_path',
        dest='ci_path',
        help='CI data directory')
    required.add_argument('--out_path',
        dest='out_path',
        help='Path to write CI data in VOC format')
    parser._action_groups.append(optional)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args() # Command line parameters

    spinner = Halo(text='Copying Images', spinner='dots')
    spinner.start()
    make_voc_directories(args.out_path) # Create output DAC directory
    copy_images(args.ci_path, args.out_path) # Copy images to output DAC directory
    spinner.stop()

    spinner = Halo(text='Copying Labels', spinner='dots')
    spinner.start()
    class_labels, num_noname = convert_xml(args.ci_path, args.out_path, 99) # Copy xml to output DAC directory
    spinner.stop()

    print('There are {} xml files with no name field or empty name field'.format(num_noname))

    class_labels.sort()
    write_pbtxt(class_labels, 99) # Write .pbtxt for classes in dataset