import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import MIL
import data_prep


def main():
    parser = argparse.ArgumentParser(description='Neutrophil MIL testing')
    parser.add_argument('--to_reload', type=str, default=None, help='Path to trained model.')
    parser.add_argument('--scn_dir', type=str, default='.', help='directory of scn files.')
    parser.add_argument('--png_dir', type=str, default=None, help='output directory of png files.')
    parser.add_argument('--tfr_dir', type=str, default=None, help='output directory of TFRecords.')
    parser.add_argument('--dic_dir', type=str, default=None, help='output directory of tile dictionaries.')
    parser.add_argument('--tile_size', type=int, default=299, help='tile size pix')
    parser.add_argument('--overlap', type=int, default=49, help='tile size pix')

    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # create output directories
    for dirs in (args.png_dir, args.dic_dir, args.tfr_dir):
        if dirs:
            try:
                os.makedirs(dirs)
            except FileExistsError:
                pass

    if args.to_reload:
        m = MIL.MIL(meta_graph=args.to_reload, log_dir='.')
        print("model {} Loaded!".format(args.to_reload), flush=True)
    else:
        m = MIL.MIL(mode='I3', log_dir='.', meta_graph=None)
        print('Using imagenet pretrained Inception V3 model.')

    slides_scn = os.listdir(args.scn_dir)
    slides_scn = list(filter(lambda x: (x[-4:] == '.scn'), slides_scn))

    for scn in slides_scn:
        s_id = scn.split('.')[0]
        out_dir = args.png_dir
        n_x, n_y, lowres, residue_x, residue_y, imglist, imlocpd, ct = \
            data_prep.tile(scn, s_id, out_dir=out_dir, std_img=std, path_to_slide=SCN_DIR,
                           tile_size=args.tile_size, overlap=args.overlap)
        imglist = np.asarray(imglist)
        labs = np.repeat(999, imglist.shape[0])
        data = tf.data.Dataset.from_tensor_slices((imglist, labs))
        data_iter = data.batch(batch_size=BATCH_SIZE, drop_remainder=False).make_one_shot_iterator()
        pred = m.inference(data_iter)

        utils.slide_prediction(pred[:, 1], cutoff=0.5)
        utils.prob_heatmap(raw_img=lowres, n_x=n_x, n_y=n_y, pred=pred, tile_dic=imlocpd, out_dir=out_dir)
        utils.plot_example(s_id=s_id, imglist=imglist, pos_score=pred[:, 1],
                           tile_dic=imlocpd, out_dir=out_dir, cutoff=0.5)


if __name__ == "__main__":

    tf.reset_default_graph()
    main()

    sys.exit(0)
