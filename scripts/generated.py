#### This file creates a dataset with the labels and backgrounds that are provided
# this creates dataset in kitti. Then convert to yolo with
#script on git. then train
## Import libraries
import numpy as np
import cv2, os, math, random
from glob import glob
from PIL import Image

#vim ~/.bashrc

## Define all parameters for the dataset manipulation
copies_in_train = 7000
copies_in_val = 3000
desired_width = 300
desired_height = 300
max_label_size = 300
min_label_size = 10
noise_mean = 0
noise_std = 0.5
blur_min_size = 7
blur_max_size = 19
max_labels = 10
label_format = 'kitti' # options: 'kitti', 'yolo', 'mxnet', 'txt'
#next week: make dataset and try to train
#backgrounds should be anything so it can detect it in different 
#surroundings. pascal dataset?
####### Add the directory of your background images are located
#######     ex: ./data/<my_backgrounds>
#./ is current directory or full path 
#pwd- print working directory
background_dir = '/scratch1/cmcgeac/datasets/bmw_labels/backgrounds/'

####### Add the directory for your resulting dataset
#######     ex: /data/<new_dataset>
saved_dir = '/scratch1/cmcgeac/datasets/bmw_labels/bmwtrain/'

# this function finds a valid location for the label within the image put it there...
def pasteLabel(image, bgX, bgY, label_size_y, iteration):

    ####### Set this directory to the label images
    #######     ex: '/data/BMW_Labels/images/' 
    image_dir = '/scratch1/cmcgeac/datasets/bmw_labels/images/'


    # Pick a random label from the label set
	#REMEMBER PNG IS CASE SENSITIVE
    label_choice = random.choice(glob(os.path.join(image_dir, "*.PNG")))

    # Resize to keep porportionality
    label = cv2.imread(label_choice, -1)
    label_size_x = int(label_size_y * label.shape[1]/label.shape[0])
    label = cv2.resize(label, (label_size_x, label_size_y))

    # Determine how much to rotate the label
    rot = random.randint(-90, 90)

    ####### Image processing techniques
    #######     ex: shrinking a side of label to simulate (following code)
    distort = random.randint(0,label_size_y/6)
    leftD = random.randint(0,1)
    rightD =  not(leftD)
    (leftD, rightD)  = (leftD*distort, rightD*distort)
    # and Distort at angle 
    oldpts = np.float32([[0,0],[label_size_x, 0],[0, label_size_y],[label_size_x, label_size_y]])#upleft, upright, lowleft, lowright
    newpts = np.float32([[0,leftD],[label_size_x, rightD],[0, label_size_y-leftD],[label_size_x, label_size_y-rightD]])
    M = cv2.getPerspectiveTransform(oldpts, newpts)
    label = cv2.warpPerspective(label, M, (label_size_x, label_size_y))

    ####### Put your image processing techniques here and comment out the previous one



    # Create region where labels will not go outside of the image boundary
    max_x_range=bgX-label.shape[1]
    max_y_range=bgY-label.shape[0]

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

        TimesTried = 0
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

    for dataset, dssize in ( ("train", copies_in_train), ("val", copies_in_val), ):
        for i in range(dssize):
            # Open a background image
            bg = None
            while bg is None:
               # print (os.path.join(background_dir))
                ####### May need to change .PNG extension to match your images
                bg_choice = random.choice(glob(os.path.join(background_dir, "*.png")))
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
			#resizing background 
            bg = cv2.resize(bg, (desired_width, desired_height))
            (rows, cols, channel) = bg.shape

            # Randomly select how many labels to add to image
            n_labels = random.randint(1, max_labels)

            # Determine size of labels
            label_y_size = int(np.floor(rows/(n_labels+1)))

            # Check to make sure labels will be correct size to detect
            label_y_size = min(label_y_size, max_label_size)
            label_y_size = max(label_y_size, min_label_size)

            # Create copy of background
            edit = bg.copy()

            # Create a text file for the bounding box (bbox) information
            #print ("{}{}/labels/data_{:05d}.txt"")
            with open("{}{}/labels/data_{:05d}.txt".format(saved_dir, dataset, i), "w") as f:
            # with open("./labels/data_{:05d}.txt".format(i), "w") as f:
                # Loop through pasting labels on the background
                for n in range(n_labels):
                    (edit, left, top, right, bottom, skip) = pasteLabel(edit, cols, rows, label_y_size, n)
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

            # Replace transparent regions of labels with background
            mask = (edit[:,:,3] < 255)
            edit[mask] = bg[mask]

            # Add noise to the image
            edit += np.random.normal(noise_mean, noise_std, size=(rows, cols, 4)).astype(np.uint8)

            # Generate a random blur value and blur the image
            blur = random.randint(blur_min_size, blur_max_size)
            if blur % 2 == 0:
                blur += 1
            newPicture = cv2.GaussianBlur(edit, (blur, blur), 1)

            # Write the resulting image to a file
            cv2.imwrite("{}{}/images/data_{:05d}.png".format(saved_dir, dataset, i), newPicture)