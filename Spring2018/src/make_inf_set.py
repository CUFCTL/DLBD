"""
Copies files from the original dataset for use with inference.
NOTE: For inference, images need to be in the form:
        image{}.jpg where {} should be replaced with 1, 2, etc.

Assumptions:
    This code is run from the ci-models directory
    The images/labels are placed in data/<class-name>
"""

import os, random
import argparse
import shutil

def make_clean_directory(dir_name):
    """
    Remove directory (if exists) and make another (clean) directory

    Parameters:
        dir_name - directory name to "clean"
    Returns:
        None
    """
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name, ignore_errors=True)
    os.makedirs(dir_name)

def copy_rnd_images(data_dir, test_img_dir, img_count):
    idx = 1
    while idx < int(img_count)+1:
        img_name = random.choice(os.listdir(data_dir))
        if img_name.split('.')[1] == 'jpg':
            shutil.copy(os.path.join(data_dir, img_name),
                os.path.join(test_img_dir, 'image%01d.jpg' % (idx)))
            idx += 1
        print(idx)
        print(img_count)

def parse_args():
    """
    Parse command line arguments.
    
    Parameters:
        None
    Returns:
        parser arguments
    """
    parser = argparse.ArgumentParser(description='Make Inference Set')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional.add_argument('--img_count',
        dest = 'img_count',
        help = 'How many images to copy from data',
        default=10)
    required.add_argument('--class_name',
        dest = 'class_name',
        help = 'Class name for data (i.e. boat2, etc.)')
    parser._action_groups.append(optional)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args() # Command line parameters

    test_img_dir = 'data/test_imgs'
    make_clean_directory(test_img_dir) # Create directory for test images

    copy_rnd_images(os.path.join('data', args.class_name), test_img_dir,
        args.img_count)
