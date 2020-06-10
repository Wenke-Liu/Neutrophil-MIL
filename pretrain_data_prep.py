import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

LAB_LST = './te_sample.csv'
IMG_DIR = '../sampled_tiles'
OUT_FILE = './tile_tfr/tst/te_sample.tfrecords'


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


tile_lab = pd.read_csv(LAB_LST)
writer = tf.python_io.TFRecordWriter(OUT_FILE)

for i in range(tile_lab.shape[0]):
    lab = tile_lab['label'][i]
    tile = os.path.basename(tile_lab['path'][i])
    fn= IMG_DIR + '/' + tile
    if (lab == 1 or lab == 0) and os.path.isfile(fn):
        lab = int(lab)
        tile = os.path.basename(tile_lab['path'][i])
        print('tile {}, label: {}'.format(tile, lab))
        img = cv2.imread(fn)  # read as uint8 array
        dim = np.shape(img)
        print(dim)
        feature = {'dim': _bytes_feature(tf.compat.as_bytes(np.asarray(dim).tostring())),
                   'image': _bytes_feature(tf.compat.as_bytes(img.tostring())),
                   'label': _int64_feature(lab)
                   }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
