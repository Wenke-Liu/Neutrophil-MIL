import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='Neutrophil MIL tile pre-train data prep')
parser.add_argument('--lab_lst', type=str, default='.', help='Label list.')
parser.add_argument('--img_dir', type=str, default=None, help='Directory of images png files.')
parser.add_argument('--out_f', type=str, default=None, help='Output file.')

LAB_LST = './tr_samples.csv'
IMG_DIR = '../sampled_tiles'
OUT_FILE = './tr_sample.tfrecords'

args = parser.parse_args()
for arg in vars(args):
    print(arg, getattr(args, arg))

LAB_LST = args.lab_lst
IMG_DIR = args.img_dir
OUT_FILE = args.out_f



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
