# Import necessary libraries
import os, sys, shutil, glob
import numpy as np
from PIL import Image

def print_usage():
	"""
	Definition: Prints the usage for the code.

	Parameters: None
	Returns: None
	"""
	print ("\nsage: python convert-kitti-to-yolo <dataset-directory> <label-file>")
	print ("	<dataset-directory>: directory with train and val split " \
		"already present in KITTI format")
	print ("	<label-file>: text file containing labels for dataset (each on new line)\n")
	exit()

def print_paths(path, label_file):
	"""
	Definition: Prints the dataset directory and label file.

	Parameters: path - dataset directory
				label_file - text file with labels for dataset
	Returns: None
	"""
	print ("\nDataset directory: " + path)
	print ("Label file: " + label_file + "\n")

def determine_label(label, labels):
	"""
	Definition: Converts label to index in label set

	Parameters: label - label from file
				labels - list of labels
	Returns: index of label in labels (in str format)
	"""
	return str(labels.index(label))

def parse_labels(label_file, labels, img_width, img_height):
	"""
	Definition: Parses label files to extract label and bounding box
		coordinates.  Converts (x1, y1, x1, y2) KITTI format to
		(x, y, width, height) normalized YOLO format.

	Parameters:
	Return: all_labels - contains a list of labels for objects in image
			all_coords - contains a list of coordinate for objects in image
	"""
	lfile = open(label_file)
	coords = []
	all_coords = []
	all_labels = []
	for line in lfile:
		l = line.split(" ")
		all_labels.append(determine_label(l[0], labels))
		coords = map(int, map(float, l[4:8]))
		x = float((float(coords[2]) + float(coords[0])) / 2.0) / float(img_width)
		y = float((float(coords[3]) + float(coords[1])) / 2.0) / float(img_height)
		width = float(float(coords[2]) - float(coords[0])) / float(img_width)
		height = float(float(coords[3]) - float(coords[1])) / float(img_height)
		tmp = [x, y, width, height]
		all_coords.append(tmp)
	lfile.close()
	return all_labels, all_coords

def copy_images(path):
	"""
	Definition: Copy all images from the training and validation image sets
		in kitti format to training and validation image sets in yolo format.
		This means converting from .png to .jpg

	Parameters: path - datasets directory
	Returns: None
	"""
	for filename in glob.glob(os.path.join(path + "train/images-kitti/", "*.*")):
		shutil.copy(filename, path + "train/images-yolo/")
	for filename in glob.glob(os.path.join(path + "val/images-kitti/", "*.*")):
		shutil.copy(filename, path + "val/images-yolo/")

	for filename in glob.glob(os.path.join(path + "train/images-yolo/", "*.*")):
		im = Image.open(filename)
		im.save(filename.split(".png")[0] + ".jpg", "jpeg")
		os.remove(filename)
	for filename in glob.glob(os.path.join(path + "val/images-yolo/", "*.*")):
		im = Image.open(filename)
		im.save(filename.split(".png")[0] + ".jpg", "jpeg")
		os.remove(filename)

def write_txt_files(path, f_train, f_val):
	"""
	Definition: Fill in a text file containing a list of all images in the
		training and validation image sets.

	Parameters: path - dataset directory
				f_train - file open for adding training examples
				f_val - file open for adding validation examples
	Returns: None
	"""
	for filename in glob.glob(os.path.join(path + "train/images-yolo/", "*.*")):
		f_train.write('%s\n' % (filename))
	for filename in glob.glob(os.path.join(path + "val/images-yolo/", "*.*")):
		f_val.write('%s\n' % (filename))

def rename_kitti_directories(path):
	"""
	Definition: Rename kitti image and label directories

	Parameters: dataset directory
	Returns: None
	"""
	os.rename(path + "train/images", path + "train/images-kitti")
	os.rename(path + "train/labels", path + "train/labels-kitti")
	os.rename(path + "val/images", path + "val/images-kitti")
	os.rename(path + "val/labels", path + "val/labels-kitti")

def make_yolo_directories(path):
	"""
	Definition: Make directories for yolo images and labels.
		Removes previously created yolo image and label directories.

	Parameters: dataset directory
	Returns: None
	"""
	if os.path.exists(path + "train/images-yolo/"):
		os.rmdir(path + "train/images-yolo/")
	if os.path.exists(path + "train/labels-yolo/"):
		os.rmdir(path + "train/labels-yolo/")
	if os.path.exists(path + "val/images-yolo/"):
		os.rmdir(path + "val/images-yolo/")
	if os.path.exists(path + "val/labels-yolo/"):
		os.rmdir(path + "val/labels-yolo/")
	os.makedirs(path + "train/images-yolo/")
	os.makedirs(path + "train/labels-yolo/")
	os.makedirs(path + "val/images-yolo/")
	os.makedirs(path + "val/labels-yolo/")

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print_usage()

	print_paths(sys.argv[1], sys.argv[2])

	# Split label file
	label_file = open(sys.argv[2])
	labels_split = label_file.read().split('\n')

	# Rename kitti directories and create yolo directoriesd
	rename_kitti_directories(sys.argv[1])
	make_yolo_directories(sys.argv[1])

	# Go through training data
	for f in os.listdir(sys.argv[1] + "train/labels-kitti/"):
		fname = (sys.argv[1] + "train/images-kitti/" + f).split(".txt")[0] + ".png"
		if os.path.isfile(fname):
			img = Image.open(fname)
			w, h = img.size
			labels, coords = parse_labels(os.path.join(sys.argv[1] + "train/labels-kitti/" + f),
				labels_split, w, h)
			yolof = open(sys.argv[1] + "train/labels-yolo/" + f, "a+")
			for l, c, in zip(labels, coords):
				yolof.write(l + " " + str(c[0]) + " " + str(c[1]) +
					" " + str(c[2]) + " " + str(c[3]) + "\n")
			yolof.close()

	# Go through validation data
	for f in os.listdir(sys.argv[1] + "val/labels-kitti/"):
		fname = (sys.argv[1] + "val/images-kitti/" + f).split(".txt")[0] + ".png"
		if os.path.isfile(fname):
			img = Image.open(fname)
			w, h = img.size
			labels, coords = parse_labels(os.path.join(sys.argv[1] + "val/labels-kitti/" + f),
				labels_split, w, h)
			yolof = open(sys.argv[1] + "val/labels-yolo/" + f, "a+")
			for l, c, in zip(labels, coords):
				yolof.write(l + " " + str(c[0]) + " " + str(c[1]) +
					" " + str(c[2]) + " " + str(c[3]) + "\n")
			yolof.close()

	# Copy images from kitti to yolo
	copy_images(sys.argv[1])
	f_train = open(sys.argv[1] + "train.txt", "a")
	f_val = open(sys.argv[1] + "val.txt", "a")
	write_txt_files(sys.argv[1], f_train, f_val)
	f_train.close()
	f_val.close()