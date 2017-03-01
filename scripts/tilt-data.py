#### This file creates a dataset with the labels and backgrounds that are provided

## Import libraries
import numpy as np
import cv2, os, math, random
from glob import glob
from PIL import Image
from time import sleep


## Define all parameters for the dataset manipulation

desired_images = 1;
n_labels = 5 #five labels per image

desired_width = 1024
desired_height = 768
max_label_size = 300
min_label_size = 10
noise_mean = 0
noise_std = 0.5
blur_min_size = 7
blur_max_size = 19


label_format = 'kitti' # options: 'kitti', 'yolo', 'mxnet', 'txt'

background_dir = './data/'
saved_dir = './data/new/'


def tiltLabel(image, bgX, bgY, label_size_y, iteration):

    ####### Set this directory to the label images
    #######     ex: '/data/BMW_Labels/images/' 
    image_dir = './data/BMW_Labels/images/'

    # Pick a random label from the label set
    label_choice = random.choice(glob(os.path.join(image_dir, "*.PNG")))

    # Resize to keep porportionality
    label = cv2.imread(label_choice, -1)
    label_size_x = int(label_size_y * label.shape[1]/label.shape[0])
    label = cv2.resize(label, (label_size_x, label_size_y))
    

    #Manipulate label here

    
    

    
    # Create region where labels will not go outside of the image boundary
    max_x_range=desired_width-label_size_x
    max_y_range=desired_height-label_size_y

    # If first label on image, no overlap possible
    if iteration == 0:
        overlap_found = False
        label_x_location = random.randint(0, max_x_range)
        label_y_location = random.randint(0, max_y_range)
    else:
        overlap_found = True

    # Check for overlap in random location chosen
    while overlap_found:

        # pick a random x and y inside the boundary
        label_x_location = random.randint(0,max_x_range)
        label_y_location = random.randint(0,max_y_range)
        
        TimesTried =0
        TimesTried = TimesTried+1
        if TimesTried>10:
            # says if you have tried to find a random location 10 times, give up and move on
            return(image, 0, 0, 0, 0, 1)

        # now iterate through the pixels that the new label would occupy
        target_area = image[ label_y_location : label_y_location + label.shape[0]
                           , label_x_location : label_x_location + label.shape[1]
                           , 3]
                           
                           
        overlap_found = np.any(target_area < 225)

    image[label_y_location:label_y_location+label.shape[0], label_x_location:label_x_location+label.shape[1]] = label

    # Get coordinates and size of label to return
    coord = np.where(label[:]>0)
    label_size_y = max(coord[0]) - min(coord[0])
    label_size_x = max(coord[1]) - min(coord[1])
    label_y_location = min(coord[0]) + label_y_location
    label_x_location = min(coord[1]) + label_x_location

    return(image, label_x_location, label_y_location, label_x_location+label_size_x, label_y_location+label_size_y, 0)


if __name__ == '__main__':

    for i in range(0, desired_images):
            # Open a background image
            bg = None
            while bg is None:
                #Generic white background - might add randomly generated background
                bg_choice = os.path.join(background_dir, "background.PNG")
                bg = cv2.imread(bg_choice, -1)
                if bg is None:
                    print ("{} is invalid.".format(bg_choice))
                    
            # Add the alpha channel to the background
            rgb = cv2.split(bg)
            try:
                bg = cv2.merge((rgb[0], rgb[1], rgb[2], 0*rgb[0]+255))
            except IndexError:
                bg = cv2.merge((rgb[0], rgb[0], rgb[0], 0*rgb[0]+255))

            # Resize the image to uniform size, get shape of image
            bg = cv2.resize(bg, (desired_width, desired_height))
            (rows, cols, channel) = bg.shape

            # Randomly select how many labels to add to image


            # Determine size of labels
            label_y_size = int(np.floor(rows/(n_labels+1)))

            # Check to make sure labels will be correct size to detect
            label_y_size = min(label_y_size, max_label_size)
            label_y_size = max(label_y_size, min_label_size)
            



            # Create copy of background
            edit = bg.copy()
            
            # Create a text file for the bounding box (bbox) information
            with open('./new/data_{:05d}.txt'.format(i), "w") as f:
                # Loop through pasting labels on the background
                for n in range(n_labels):
                    (edit, left, top, right, bottom, skip) = tiltLabel(edit, cols, rows, label_y_size, n)
                    # If a suitable spot for label is not found in a timely manner, skip it
                    if not skip:
                        # Write information to label text file
                        if label_format == 'kitti':
                            f.write("Label 0 0 0 {} {} {} {} 0 0 0 0 0 0 0 0\n".format(left, top, right, bottom))
                        elif label_format == 'yolo':
                            f.write("Label {} {} {} {}\n".format(left, top, right, bottom))
                        elif label_format == 'mxnet':
                            f.write("Label {} {} {} {}\n".format(left, top, right, bottom))
                        elif label_format == 'txt':
                            f.write("Label {} {} {} {}\n".format(left, top, right, bottom))

            cv2.imwrite('./new/edit_{:05d}.png'.format(i), edit)
      
            
            print ("Image Created")
            
            
            