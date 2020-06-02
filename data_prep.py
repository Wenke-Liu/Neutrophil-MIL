import os
import tensorflow as tf
import numpy as np
import pandas as pd
import multiprocessing as mp
import staintools
from PIL import Image
from openslide import OpenSlide


def bgcheck(img):

    the_imagea = np.array(img)[:, :, :3]
    the_imagea = np.nan_to_num(the_imagea)
    mask = (the_imagea[:, :, :3] > 200).astype(np.uint8)
    maskb = (the_imagea[:, :, :3] < 5).astype(np.uint8)
    mask = mask[:, :, 0] * mask[:, :, 1] * mask[:, :, 2]
    maskb = maskb[:, :, 0] * maskb[:, :, 1] * maskb[:, :, 2]
    white = (np.sum(mask) + np.sum(maskb)) / (the_imagea.shape[0]*the_imagea.shape[1])
    return white


# Tile color normalization
def normalization(img, sttd):
    img = np.array(img)[:, :, :3]
    img = staintools.LuminosityStandardizer.standardize(img)
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(sttd)
    img = normalizer.transform(img)
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    return img


def v_slide(slp, s_id, n_y, x, y, tile_size, step_size, out_dir, x0, std=np.asarray([])):
    # pid = os.getpid()
    # print('{}: start working'.format(pid))
    slide = OpenSlide(slp)
    imloc = []
    imlist = []
    y0 = 0
    target_x = x0 * step_size
    image_x = target_x + x
    while y0 < n_y:
        target_y = y0 * step_size
        image_y = target_y + y
        img = slide.read_region((image_x, image_y), 0, (tile_size, tile_size))
        wscore = bgcheck(img)
        # output png files if an output directory is given
        if wscore < 0.8:
            tile_name = s_id + '_tile_x_{}_y_{}.png'.format(str(image_x).zfill(6), str(image_y).zfill(6))
            if std.any():
                img = normalization(img, std)
            if out_dir:
                img.save(out_dir + '/' + tile_name)
        # append image data and descriptions to list
            imlist.append(np.asarray(img)[:, :, :3].astype(np.float32))
            imloc.append([s_id, str(x0).zfill(3), str(y0).zfill(3),
                          str(target_x).zfill(5), str(target_y).zfill(5), tile_name])

        y0 += 1
    slide.close()

    return imloc, imlist


def tile(scn_file, s_id, out_dir, std_img, path_to_slide="../Neutrophil/", tile_size=299, overlap=49):
    slp = str(path_to_slide + '/' + scn_file)
    slide = OpenSlide(slp)

    assert 'openslide.bounds-height' in slide.properties
    assert 'openslide.bounds-width' in slide.properties
    assert 'openslide.bounds-x' in slide.properties
    assert 'openslide.bounds-y' in slide.properties

    x = int(slide.properties['openslide.bounds-x'])
    y = int(slide.properties['openslide.bounds-y'])
    bounds_height = int(slide.properties['openslide.bounds-height'])
    bounds_width = int(slide.properties['openslide.bounds-width'])

    step_size = tile_size - overlap

    n_x = int((bounds_width - 1) / step_size)
    n_y = int((bounds_height - 1) / step_size)

    residue_x = int((bounds_width - n_x * step_size)/50)
    residue_y = int((bounds_height - n_y * step_size)/50)
    lowres = slide.read_region((x, y), 2, (int(n_x*step_size/16), int(n_y*step_size/16)))
    lowres = np.array(lowres)[:, :, :3]

    x0 = 0
    # create multiprocessing pool
    print(mp.cpu_count())
    pool = mp.Pool(processes=8)
    tasks = []
    while x0 < n_x:
        task = tuple((slp, s_id, n_y, x, y, tile_size, step_size, out_dir, x0, std_img))
        tasks.append(task)
        x0 += 1
    # slice images with multiprocessing
    temp = pool.starmap(v_slide, tasks)
    tempdict = list(zip(*temp))[0]
    tempimglist = list(zip(*temp))[1]

    pool.close()
    pool.join()
    print(list(map(len, tempdict)))
    print(list(map(len, tempimglist)))

    imloc = []
    list(map(imloc.extend, tempdict))
    imlocpd = pd.DataFrame(imloc, columns=["slide", "X_pos", "Y_pos", "X", "Y", "tile_name"])

    imglist = []
    list(map(imglist.extend, tempimglist))

    assert (len(imglist) == imlocpd.shape[0]), "Length of location file and image data count does not match!"

    ct = imlocpd.shape[0]

    return n_x, n_y, lowres, residue_x, residue_y, imglist, imlocpd, ct

# The following functions can be used to convert a value to a type compatible
# with tf.Example.


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_array_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_array_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def main():  # run as main

    import argparse
    parser = argparse.ArgumentParser(description='Neutrophil MIL data prep')
    parser.add_argument('--scn_dir', type=str, default='.', help='directory of scn files.')
    parser.add_argument('--png_dir', type=str, default=None, help='output directory of png files.')
    parser.add_argument('--tfr_dir', type=str, default=None, help='output directory of TFRecords.')
    parser.add_argument('--dic_dir', type=str, default=None, help='output directory of tile dictionaries.')
    parser.add_argument('--tile_size', type=int, default=299, help='tile size pix')
    parser.add_argument('--overlap', type=int, default=49, help='tile size pix')
    parser.add_argument('--standard', type=str, default=None, help='standard image for color normalization')
    parser.add_argument('--slide_lab', type=str, default=None, help='txt file containing slide level labels')

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

    if args.standard:  # standard image for color normalization
        std = staintools.read_image(args.standard)
        std = staintools.LuminosityStandardizer.standardize(std)
    else:
        std = np.asarray([])

    # read slide level file as dictionary, if provided
    if args.slide_lab:
        s_lab = {}
        with open(args.slide_lab) as l:
            for line in l:
                line = line.replace('"', '').strip()
                (key, val) = line.split(',')
                s_lab[key] = val

    # list all scn files
    scn_ls = os.listdir(args.scn_dir)
    scn_ls = list(filter(lambda x: (x[-4:] == '.scn'), scn_ls))

    # iterate through all scn files
    for f in scn_ls:
        f_id = f.split('.')[0]
        print('Slide id: {}'.format(f_id))

        if args.png_dir:  # whether output png files
            out_dir = args.png_dir+'/'+f_id
            try:
                os.mkdir(out_dir)
            except FileExistsError:
                pass

        else:
            out_dir = None

        # read slide and get tiles using multiple processing.
        # save png files if asked; save tile info and image arrays in lists.

        n_x, n_y, lowres, residue_x, residue_y, imglist, imlocpd, ct =\
            tile(f, f_id, out_dir=out_dir, std_img=std, path_to_slide=args.scn_dir,
                 tile_size=args.tile_size, overlap=args.overlap)
        print('number columns:{}'.format(n_x))
        print('number of rows: {}'.format(n_y))
        print('total number of tiles in slide {}: {}'.format(f_id, ct))

        dims = list(map(np.shape, imglist))

        print(dims[0])
        print(type(imglist[0]))
        print(imglist[0].shape)
        assert(all(x == dims[0] for x in dims)), "Images are of different dimensions"

        if args.dic_dir: # if save tile info in csv
            imlocpd.to_csv(args.dic_dir + '/' + f_id + "_dict.csv", index=False)
            print('Tile info saved in: ' + args.dic_dir + '/' + f_id + "_dict.csv")

        if args.tfr_dir:  # if output TFRecords files for future training

            tf_fn = args.tfr_dir + '/' + f_id + '.tfrecords'
            writer = tf.python_io.TFRecordWriter(tf_fn)
            try:
                s_lab  # if slide level labels are available
                try: # get slide level label from input dictionary
                    lab = int(s_lab[f_id])
                    print('slide {} label: {}'.format(f_id, str(lab)))
                except KeyError:  # if slide id not in the label list
                    lab = 999  # numeric code for missing value
                    print('slide {} has no label: coded as {}'.format(f_id, str(lab)))
            except NameError:  # no labels provided
                lab = 999

            for i in range(len(imglist)):
                feature = {'dim': _bytes_feature(tf.compat.as_bytes(np.asarray(dims[i]).tostring())),
                           'image': _bytes_feature(tf.compat.as_bytes(imglist[i].tostring())),
                           'label': _int64_feature(lab)
                           }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            writer.close()


if __name__ == "__main__":
    main()
