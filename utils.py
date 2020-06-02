import numpy as np
import pandas as pd
import cv2
from PIL import Image
from openslide import OpenSlide


def slide_prediction(pos_score, cutoff=0.5):
    mean_pos_score = np.mean(pos_score)
    pos_ct = sum(s > cutoff for s in pos_score)
    print('Mean positive score for the slide: {}'.format(str(round(mean_pos_score, 5))))
    print('{} positive tiles in total of {}.'.format(str(pos_ct), str(len(pos_score))))


def prob_heatmap(raw_img, n_x, n_y, pred, tile_dic, out_dir, cutoff=0.5):
    tile_dic['pos_score'] = pred[:, 1]
    tile_dic['neg_score'] = pred[:, 0]
    opt = np.full((n_x, n_y), 0)
    hm_R = np.full((n_x, n_y), 0)
    hm_G = np.full((n_x, n_y), 0)
    hm_B = np.full((n_x, n_y), 0)

    # Positive is labeled red in output heat map
    for index, row in tile_dic.iterrows():
        opt[int(row["X_pos"]), int(row["Y_pos"])] = 255
        if row['pos_score'] >= cutoff:
            hm_R[int(row["X_pos"]), int(row["Y_pos"])] = 255
            hm_G[int(row["X_pos"]), int(row["Y_pos"])] = int((1 - (row['pos_score'] - 0.5) * 2) * 255)
            hm_B[int(row["X_pos"]), int(row["Y_pos"])] = int((1 - (row['pos_score'] - 0.5) * 2) * 255)
        else:
            hm_B[int(row["X_pos"]), int(row["Y_pos"])] = 255
            hm_G[int(row["X_pos"]), int(row["Y_pos"])] = int((1 - (row['neg_score'] - 0.5) * 2) * 255)
            hm_R[int(row["X_pos"]), int(row["Y_pos"])] = int((1 - (row['neg_score'] - 0.5) * 2) * 255)

    # expand 5 times
    opt = opt.repeat(50, axis=0).repeat(50, axis=1)

    # small-scaled original image
    ori_img = cv2.resize(raw_img, (np.shape(opt)[0], np.shape(opt)[1]))
    ori_img = ori_img[:np.shape(opt)[1], :np.shape(opt)[0], :3]
    tq = ori_img[:, :, 0]
    ori_img[:, :, 0] = ori_img[:, :, 2]
    ori_img[:, :, 2] = tq
    cv2.imwrite(out_dir + '/Original_scaled.png', ori_img)

    # binary output image
    topt = np.transpose(opt)
    opt = np.full((np.shape(topt)[0], np.shape(topt)[1], 3), 0)
    opt[:, :, 0] = topt
    opt[:, :, 1] = topt
    opt[:, :, 2] = topt
    cv2.imwrite(out_dir + '/Mask.png', opt * 255)

    # output heatmap
    hm_R = np.transpose(hm_R)
    hm_G = np.transpose(hm_G)
    hm_B = np.transpose(hm_B)
    hm_R = hm_R.repeat(50, axis=0).repeat(50, axis=1)
    hm_G = hm_G.repeat(50, axis=0).repeat(50, axis=1)
    hm_B = hm_B.repeat(50, axis=0).repeat(50, axis=1)
    hm = np.dstack([hm_B, hm_G, hm_R])
    cv2.imwrite(out_dir + '/HM.png', hm)

    # superimpose heatmap on scaled original image
    overlay = ori_img * 0.5 + hm * 0.5
    cv2.imwrite(out_dir + '/Overlay.png', overlay)


def plot_example(s_id, imglist, pos_score, tile_dic, out_dir, cutoff=0.9):
    for index, row in tile_dic.iterrows():
        prob = float(pos_score[index])
        if prob > cutoff:
            img = imglist[index]
            n_x = row['X_pos']
            n_y = row['Y_pos']
            tile_name = s_id + '_tile_nx_{}_ny_{}_prob_{}.png'.\
                format(str(n_x).zfill(6), str(n_y).zfill(6), str(round(prob, 3)))
            im = Image.fromarray(np.uint8(1 - img) * 255)
            im.save(out_dir + '/' + tile_name)
