import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import MIL
import staintools
import utils
import data_prep


LOG_DIR = './test/log'
METAGRAPH_DIR = './test/model'
TFR_DIR = './tfr_test'
SCN_DIR = '.'
PNG_DIR = './png'
DIC_DIR = '.'
PRE_DIR = './tfr_test'
SAVE = True

TILE_SIZE = 299
OVERLAP = -10000
STD = './colorstandard.png'

ARCHITECTURE = 'I3'
N_EPOCH = 1
BATCH_SIZE = 2
TOP_K = 2


def main():
    slides_tfr = os.listdir(TFR_DIR)
    pretrain_tfr = os.listdir(PRE_DIR)
    slides_scn = os.listdir(SCN_DIR)
    slides_scn = list(filter(lambda x: (x[-4:] == '.scn'), slides_scn))




    m = MIL.MIL(mode=ARCHITECTURE, log_dir=LOG_DIR, meta_graph=None)
    #m.pre_train(pretrain_data_path=[PRE_DIR + '/' + f for f in pretrain_tfr],
     #           valid_data_path=['./tfr_test/0000026280.tfrecords'],
      #          batch_size=BATCH_SIZE, n_epoch=N_EPOCH, out_dir=METAGRAPH_DIR, save=SAVE)

    m.train(data_dir=TFR_DIR, slides=slides_tfr, top_k=TOP_K, sample_rate=0.8,
            valid_data_path=['./tfr_test/0000026280.tfrecords'],
            n_epoch=N_EPOCH, batch_size=BATCH_SIZE,
            save=SAVE, out_dir=METAGRAPH_DIR)
    print('Trained!')


if __name__ == "__main__":

    tf.reset_default_graph()

    for DIR in (LOG_DIR, METAGRAPH_DIR):
        try:
            os.mkdir(DIR)
        except FileExistsError:
            pass

    main()
    sys.exit(0)
