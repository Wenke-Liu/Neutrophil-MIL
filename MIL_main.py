import sys
import os
import tensorflow as tf
import data_input
import MIL

ARCHITECTURE = 'I3'
N_EPOCH = 2
BATCH_SIZE = 1
LOG_DIR = './test/log'
METAGRAPH_DIR = './test/model'
DATA_DIR = './tfr_test'
TOP_K = 2

def main(to_reload=None):

    filenames = os.listdir(DATA_DIR)
    filenames = list(filter(lambda x: (x[-10:] == '.tfrecords'), filenames))
    full_fn = [DATA_DIR + '/' + f for f in filenames]

    if to_reload:
        m = MIL.MIL(mode=ARCHITECTURE, log_dir=LOG_DIR, meta_graph=to_reload)
        print("Loaded!", flush=True)
        data = data_input.DataSet(full_fn, batch_size=BATCH_SIZE)
        print('Data size: {}'.format(data.get_size()))
        data_iter = data.batch_iter()
        new_pred = m.inference(data_iter)
        print(new_pred)

    else:
        m = MIL.MIL(mode=ARCHITECTURE, log_dir=LOG_DIR, meta_graph=None)
        m.train(data_dir=DATA_DIR, slides=filenames, top_k=TOP_K, n_epoch=N_EPOCH, batch_size=BATCH_SIZE,
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
        to_reload = sys.argv[1]
        main(to_reload=to_reload)
    except IndexError:
        main()
