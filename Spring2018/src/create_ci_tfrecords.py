"""
Convert raw DAC dataset to TFRecord for object_detection.

Usage:
    NOTE: Execute from CUFCTL-Track directory
    python python/create_dac_tfrecords.py \
        --label_map_path=<path to .pbtxt> \
        --data_dir=<path to DACdevkit> \
        --year=DAC2017 \
        --set=<train or val> \
        --output_path=data/tfrecords/<name>.record
"""

# Import necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib, io, logging, os

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

from utils import dict_to_tf_example

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw DAC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('year', 'DAC2017', 'Desired challenge year.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/dac_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']
YEARS = ['CI2018']

def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))
    if FLAGS.year not in YEARS:
        raise ValueError('year must be in : {}'.format(YEARS))

    data_dir = FLAGS.data_dir
    years = ['CI2018']
    if FLAGS.year != 'merged':
        years = [FLAGS.year]

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    for year in years:
        logging.info('Reading from CI%s dataset.', year)
        examples_path = os.path.join('data', FLAGS.set + '.txt')
        annotations_dir = os.path.join(data_dir, year, FLAGS.annotations_dir)
        examples_list = dataset_util.read_examples_list(examples_path)
        for idx, example in enumerate(examples_list):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(examples_list))
            path = os.path.join(annotations_dir, example + '.xml')
            if not os.path.exists(path):
                continue
            with tf.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                      FLAGS.ignore_difficult_instances)
            writer.write(tf_example.SerializeToString())

    writer.close()

if __name__ == '__main__':
  tf.app.run()