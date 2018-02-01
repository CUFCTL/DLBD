### Import necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from xml.etree import ElementTree as et
import cv2, os, sys, shutil, glob, argparse, re, io
import numpy as np
import tensorflow as tf
import PIL, hashlib, logging
from PIL import Image
from lxml import etree
from halo import Halo

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

RED = (0, 0, 255)
LWIDTH = 2

### Visualization Utilities
def get_bb(xml_file):
    """
    Read the XML file and get ground truth bounding box
    coordinates.

    Parameters:
        xml_file - file containing label information
    Returns:
        tuple - (xmin, xmax, ymin, ymax)
    """
    # Open the XML file and parse the tree
    with open(xml_file) as xf:
        tree = et.parse(xf)

    # Look for specific nodes and place into the tuple
    xmin, xmax, ymin, ymax = None, None, None, None
    for node in tree.iter():
        if node.tag == 'xmin': xmin = int(node.text)
        if node.tag == 'xmax': xmax = int(node.text)
        if node.tag == 'ymin': ymin = int(node.text)
        if node.tag == 'ymax': ymax = int(node.text)

    return (xmin, xmax, ymin, ymax)

def draw_bb(img_path, oimg_path):
    """
    Draw bounding box on the give image and write to new image.
    NOTE: Assumes corresponding JPG and XML in same directory.

    Parameters:
        img_path - original image path
        bb - tuple containing bounding box
        oimg_path - output image path

    Returns:
        None
    """
    # Make sure output directory has been created
    out_path = oimg_path.split('/')[:-1]
    out_path = os.path.join('/', *out_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Read input image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # Check to see if corresponding XML file (label) exists
    xml_path = img_path.split('.')[0] + '.xml'
    if not os.path.isfile(xml_path):
        cv2.imwrite(oimg_path, img)
        return

    # Get bounding boxes from corresponding file
    xmin, xmax, ymin, ymax = get_bb(xml_path)

    if xmin == None or xmax == None or ymin == None or ymax == None:
        # Write output image without box
        cv2.imwrite(oimg_path, img)
        return

    # Draw bounding box on image
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), RED, LWIDTH)

    # Write output image
    cv2.imwrite(oimg_path, img)

### XML Utilities
def read_ci_xml(ci_xml, type_data):
    """
    Read XML tree structure from CI XML file

    Parameters:
        ci_xml - XML file in CI format
        type_data - 13-class or 100-class
    Returns:
        folder - directory where image is located
        filename - name of corresponding image
        database - database name
        width - image width
        height - image height
        name - class name of object
        xmax - bottom right x coordinate
        xmin - top left x coordinate
        ymax - bottom right y coordinate
        ymin - top left y coordinate
    """
    ci_tree = etree.parse(ci_xml)
    root = ci_tree.getroot()

    folder, filename, database = '', '', ''
    name, width, height = '', '', ''
    xmax, xmin, ymax, ymin = '', '', '', ''
    for child in root:
        if child.tag == 'folder':
            folder = child.text
        if child.tag == 'filename':
            if child.text.split('.') == 'jpg':
                filename = child.text
            else:
                filename = child.text + '.jpg'
        if child.tag == 'source':
            for child2 in child:
                if child2.tag == 'database':
                    database = child2.text
        if child.tag == 'size':
            for child2 in child:
                if child2.tag == 'width':
                    width = child2.text
                if child2.tag == 'height':
                    height = child2.text
        if child.tag == 'object':
            for child2 in child:
                if child2.tag == 'name':
                    if type_data == 13:
                        name = re.split(r'(\d+)', child2.text)[0]
                    else:
                        name = child2.text
                if child2.tag == 'bndbox':
                    for child3 in child2:
                        if child3.tag == 'xmax':
                            xmax = child3.text
                        if child3.tag == 'xmin':
                            xmin = child3.text
                        if child3.tag == 'ymax':
                            ymax = child3.text
                        if child3.tag == 'ymin':
                            ymin = child3.text

    return (folder, filename, database, width, height,
            name, xmax, xmin, ymax, ymin)

def make_voc_directories(voc_path):
    """
    Create directories for images and labels.
    NOTE: Removes previously created directories.

    Parameters:
        voc_path - path to voc directory to create
    Returns:
        None
    """
    if os.path.exists(voc_path):
        shutil.rmtree(voc_path)
    os.makedirs(voc_path)
    os.makedirs(os.path.join(voc_path, 'CI2018'))
    os.makedirs(os.path.join(voc_path, 'CI2018', 'Annotations'))
    os.makedirs(os.path.join(voc_path, 'CI2018', 'JPEGImages'))

def determine_prepend(vid_path):
    """
    Create the string to prepend to the file name.

    Parameters:
        vid_path - full path name to video directory
    Returns:
        prepend - string to prepend to filename.
    """
    vid_1 = vid_path.rstrip('/').split('/')[-1].split('(')[0]
    vid_2 = vid_path.rstrip('/').split('/')[-1].split('(')[-1].split(')')[0]
    vid_3 = vid_path.rstrip('/').split('/')[-1].split('(')[-1].split(')')[-1].split('-')[-1]
    if vid_1 == vid_2:
        return vid_1
    if vid_2 == vid_3:
        return vid_1 + '_' + vid_2
    else:
        return vid_1 + '_' + vid_2 + '_' + vid_3

def copy_images(ci_path, voc_path):
    """
    Copy images from the CI directory to VOC JPEGImages directory
    while renaming all images.
    Images are renamed by prepending the "video" (directory) name.
    Ex: 360(12)-2/0001.jpg -> 360_12_2_0001.jpg

    Parameters:
        ci_path - path to original CI training data
        voc_path - path to VOC JPEGImages directory to write to
    Returns:
        None
    """
    for video in sorted(os.listdir(ci_path)):
        if video == '.DS_Store':    # For MAC
            continue
        vid_path = os.path.join(ci_path, video)
        #prepend = determine_prepend(vid_path)
        prepend = video
        for filename in os.listdir(os.path.join(ci_path, video)):
            if filename.split('.')[-1] == 'jpg':
                jpg = os.path.join(ci_path, video, filename)
                shutil.copy(jpg, os.path.join(voc_path, 'CI2018', 'JPEGImages', prepend + '_' + filename))

def write_xml(fd, fn, db, w, h, name, xmax, xmin, ymax, ymin, xml, voc_path):
    """
    Write XML file in VOC format

    Parameters:
        fd - folder
        fn - filename
        db - database
        w - width
        h - height
        name - object class
        xmax - lower right x coordinate
        xmin - upper left x coordinate
        ymax - lower right y coordinate
        ymin - upper left y coordinate
        xml - xml file name
        voc_path - path to write XML file
    Returns:
        None
    """
    full_fn = xml.split('/')[-1]
    xml_fn = full_fn.split('.')[0] + '.xml'
    f = open(os.path.join(voc_path, 'CI2018', 'Annotations', xml_fn), 'w')
    line = '<annotation>\n'; f.write(line)
    line = '\t<filename>' + full_fn + '</filename>\n'; f.write(line)
    line = '\t<folder>' + fd + '</folder>\n'; f.write(line)
    line = '\t<object>\n'; f.write(line)
    line = '\t\t<name>' + name + '</name>\n'; f.write(line)
    line = '\t\t<bndbox>\n'; f.write(line)
    line = '\t\t\t<xmax>' + xmax + '</xmax>\n'; f.write(line)
    line = '\t\t\t<xmin>' + xmin + '</xmin>\n'; f.write(line)
    line = '\t\t\t<ymax>' + ymax + '</ymax>\n'; f.write(line)
    line = '\t\t\t<ymin>' + ymin + '</ymin>\n'; f.write(line)
    line = '\t\t</bndbox>\n'; f.write(line)
    if not xmax == '' or not xmin == '' or not ymax == '' or not ymin == '':
        if int(xmax)-int(xmin) < 10 or int(ymax)-int(ymin) < 10:
            line = '\t\t<difficult>1</difficult>\n'; f.write(line)
        else:
            line = '\t\t<difficult>0</difficult>\n'; f.write(line)
    else:
        line = '\t\t<difficult>0</difficult>\n'; f.write(line)
    line = '\t\t<occluded>0</occluded>\n'; f.write(line)
    line = '\t\t<pose>Unspecified</pose>\n'; f.write(line)
    line = '\t\t<truncated>0</truncated>\n'; f.write(line)
    line = '\t</object>\n'; f.write(line)
    line = '\t<segmented>0</segmented>\n'; f.write(line)
    line = '\t<size>\n'; f.write(line)
    line = '\t\t<depth>3</depth>\n'; f.write(line)
    line = '\t\t<height>' + h + '</height>\n'; f.write(line)
    line = '\t\t<width>' + w + '</width>\n'; f.write(line)
    line = '\t</size>\n'; f.write(line)
    line = '\t<source>\n'; f.write(line)
    line = '\t\t<database>' + db + '</database>\n'; f.write(line)
    line = '\t</source>\n'; f.write(line)
    line = '</annotation>'; f.write(line)
    f.close()

def write_pbtxt(class_labels, type_data):
    """
    Write .pbtxt file with a list of all classes in dataset

    Parameters:
        class_labels - list of dataset classes (sorted)
        type_data - type of data (13 or 100)
    Returns:
        None
    """
    if not os.path.exists('data'):
        os.makedirs('data')
    if type_data == 13:
        f = open(os.path.join('data', 'ci_label_map_coarse.pbtxt'), 'w')
    else:
        f = open(os.path.join('data', 'ci_label_map.pbtxt'), 'w')
    #f = open(os.path.join('data', 'ci_label_map_' + str(type_data) + '.pbtxt'), 'w')
    for idx, label in enumerate(class_labels):
        line = 'item {\n'; f.write(line)
        line = '\tid: ' + str(idx+1) + '\n'; f.write(line)
        line = '\tname: "' + label + '"\n}\n'; f.write(line)
    f.close()

def convert_xml(ci_path, voc_path, type_data):
    """
    Copy XML files from the CI directory to VOC Annotations directory
    while converting the contents of the XML file and renaming based on the
    image.
    XML files will be named by prepending the "video" (directory) name.
    Ex. 360(12)-2/0001.xml -> 360_12_2_0001.xml

    Parameters:
        ci_path - path to original CI training data
        voc_path - path to VOC Annotations directory to write to
        type_data - 13-class dataset or 100-class dataset
    Returns:
        None
    """
    class_labels = []
    num_noname = 0 
    for video in sorted(os.listdir(ci_path)):
        if video == '.DS_Store':    # For MAC
            continue
        vid_path = os.path.join(ci_path, video)
        prepend = determine_prepend(vid_path)
        for filename in os.listdir(os.path.join(ci_path, video)):
            if filename.split('.')[-1] == 'xml':
                xml = os.path.join(ci_path, video, filename)
                xml_fn = prepend + '_' + filename
                # fd = folder
                # fn = filename
                # db = database
                # w, h = image width, height
                fd, fn, db, w, h, name, xmax, xmin, ymax, ymin = read_ci_xml(xml, type_data)
                if not name == '' and not name in class_labels:
                    class_labels.append(name)
                if name == '':
                    num_noname += 1
                #write_xml(fd, fn, db, w, h, name, xmax, xmin, ymax, ymin,
                #    xml_fn, voc_path)
                jpg_fn = xml_fn.split('.')[0] + '.jpg'
                write_xml(fd, fn, db, w, h, name, xmax, xmin, ymax, ymin,
                    jpg_fn, voc_path)

    return class_labels, num_noname

### Re-ID Utilities
def reid_read_xml(xml_file):
    """
    Read XML tree structure to get bounding box coordinates

    Parameters:
        xml_file - XML file in CI format
    Returns:
        xmax - bottom right x coordinate
        xmin - top left x coordinate
        ymax - bottom right y coordinate
        ymin - top left y coordinate
    """
    tree = etree.parse(xml_file)
    root = tree.getroot()

    xmin, xmax, ymin, ymax = '', '', '', ''
    for child in root:
        if child.tag == 'object':
            for child2 in child:
                if child2.tag == 'bndbox':
                    for child3 in child2:
                        if child3.tag == 'xmax':
                            xmax = child3.text
                        if child3.tag == 'xmin':
                            xmin = child3.text
                        if child3.tag == 'ymax':
                            ymax = child3.text
                        if child3.tag == 'ymin':
                            ymin = child3.text

    return (xmax, xmin, ymax, ymin)

### TFRecord Utils
def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding CI XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding CI dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      CI dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  #img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
  img_path = os.path.join('CI2018', image_subdirectory, data['filename'])
  full_path = os.path.join(dataset_directory, img_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  for obj in data['object']:
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue

    difficult_obj.append(int(difficult))
    if obj['bndbox']['xmin'] is None or obj['bndbox']['xmax'] is None or obj['bndbox']['ymin'] is None or obj['bndbox']['ymax'] is None:
      continue

    xmin.append(float(obj['bndbox']['xmin']) / width)
    ymin.append(float(obj['bndbox']['ymin']) / height)
    xmax.append(float(obj['bndbox']['xmax']) / width)
    ymax.append(float(obj['bndbox']['ymax']) / height)
    classes_text.append(obj['name'].encode('utf8'))
    classes.append(label_map_dict[obj['name']])
    truncated.append(int(obj['truncated']))
    poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example