import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import MIL
import staintools
import utils
import data_prep


LOG_DIR = './200608/log'
METAGRAPH_DIR = './200608/model'
TFR_DIR = './tfr/trn'
SCN_DIR = '.'
PNG_DIR = './png'
DIC_DIR = '.'
PRE_DIR = './tile_tfr/trn'

TILE_SIZE = 299
OVERLAP = -10000
STD = './colorstandard.png'

ARCHITECTURE = 'I3'
N_EPOCH = 50
BATCH_SIZE = 64
TOP_K = 2


def main(to_reload=None):
    slides_tfr = os.listdir(TFR_DIR)
    pretrain_tfr = os.listdir(PRE_DIR)
    slides_scn = os.listdir(SCN_DIR)
    slides_scn = list(filter(lambda x: (x[-4:] == '.scn'), slides_scn))

    if to_reload:
        m = MIL.MIL(mode=ARCHITECTURE, log_dir=LOG_DIR, meta_graph=to_reload)
        print("Loaded!", flush=True)

        if MODE == 'test':
            std = staintools.read_image(STD)
            std = staintools.LuminosityStandardizer.standardize(std)
            try:
                os.mkdir(PNG_DIR)
            except FileExistsError:
                pass

            for scn in slides_scn:
                s_id = scn.split('.')[0]
                out_dir = PNG_DIR
                n_x, n_y, lowres, residue_x, residue_y, imglist, imlocpd, ct = \
                    data_prep.tile(scn, s_id, out_dir=out_dir, std_img=std, path_to_slide=SCN_DIR,
                                   tile_size=TILE_SIZE, overlap=OVERLAP)
                imglist = np.asarray(imglist)
                labs = np.repeat(999, imglist.shape[0])
                data = tf.data.Dataset.from_tensor_slices((imglist, labs))
                data_iter = data.batch(batch_size=BATCH_SIZE, drop_remainder=False).make_one_shot_iterator()
                pred = m.inference(data_iter)

                utils.slide_prediction(pred[:, 1], cutoff=0.5)
                utils.prob_heatmap(raw_img=lowres, n_x=n_x, n_y=n_y, pred=pred, tile_dic=imlocpd, out_dir=out_dir)
                utils.plot_example(s_id=s_id, imglist=imglist, pos_score=pred[:, 1],
                                   tile_dic=imlocpd, out_dir=out_dir, cutoff=0.5)

        else:  # 'retrain'
            print('Retraining begin...')
            m.train(data_dir=TFR_DIR, slides=slides_tfr, top_k=TOP_K, sample_rate=0.1, n_epoch=N_EPOCH, batch_size=BATCH_SIZE,
                    save=True, out_dir=METAGRAPH_DIR)

    else:
        m = MIL.MIL(mode=ARCHITECTURE, log_dir=LOG_DIR, meta_graph=None)
        m.pre_train(pretrain_data_path=[PRE_DIR + '/' + f for f in pretrain_tfr],
                    batch_size=BATCH_SIZE, n_epoch=15, out_dir=METAGRAPH_DIR, save=True)
        m.train(data_dir=TFR_DIR, slides=slides_tfr, top_k=TOP_K, sample_rate=0.1, n_epoch=N_EPOCH, batch_size=BATCH_SIZE,
                save=True, out_dir=METAGRAPH_DIR)
        print('Trained!')


if __name__ == "__main__":

    tf.reset_default_graph()

    for DIR in (LOG_DIR, METAGRAPH_DIR):
        try:
            os.mkdir(DIR)
        except FileExistsError:
            pass

    try:
        TO_RELOAD = sys.argv[1]
        main(to_reload=TO_RELOAD)
    except IndexError:
        main()

    sys.exit(0)
