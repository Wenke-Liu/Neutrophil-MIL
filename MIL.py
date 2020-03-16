import os
import sys
import tensorflow as tf
import numpy as np
from datetime import datetime
import data_input
import models

"""
Based on Campanella et al, Clinical-grade computational pathology using weakly supervised deep 
learning on whole slide images, Nature Medicine 2019
https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019
Implemented in Tensorflow 1.10
"""


class MIL:

    """
    Multiple instance learning with slide level labels
    """

    RESTORE_KEY = "to_restore"

    def __init__(self,
                 mode='I3',
                 n_class=2,
                 learning_rate=1E-3,
                 dropout=0.5,
                 save_graph_def=True,
                 meta_graph=None,
                 log_dir="./log"
                 ):

        self.architecture = mode
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.n_class = n_class
        self.epoch_trained = 0

        self.sesh = tf.Session()

        if not meta_graph:  # new model
            self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")

            # build graph
            handles = self._buildGraph()
            for handle in handles:
                tf.add_to_collection(MIL.RESTORE_KEY, handle)
            self.sesh.run(tf.global_variables_initializer())

        else:  # restore saved model
            model_datetime, model_name = os.path.basename(meta_graph).split("_MIL_")
            self.datetime = "{}_reloaded".format(model_datetime)
            self.architecture, _ = model_name.split("_lr_")

            # rebuild graph
            meta_graph = os.path.abspath(meta_graph)
            tf.train.import_meta_graph(meta_graph + ".meta").restore(
                self.sesh, meta_graph)
            handles = self.sesh.graph.get_collection(MIL.RESTORE_KEY)

        # unpack handles for tensor ops to feed or fetch
        (self.x_in, self.y_in, self.training_status,
         self.logits, self.pred, self.cost,
         self.global_step, self.train_op, self.merged_summary) = handles

        if save_graph_def:  # tensorboard
            try:
                os.mkdir(log_dir + '/training')
                #os.mkdir(log_dir + '/validation')
            except FileExistsError:
                pass
            self.train_logger = tf.summary.FileWriter(log_dir + '/training', self.sesh.graph)
            #self.valid_logger = tf.summary.FileWriter(log_dir + '/validation', self.sesh.graph)

    def _buildGraph(self):
        x_in = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
        y_in = tf.placeholder(tf.int64, shape=[None])
        onehot_labels = tf.one_hot(indices=tf.cast(y_in, tf.int32), depth=self.n_class)
        is_train = tf.placeholder_with_default(False, shape=[], name="is_train")
        global_step = tf.Variable(0, trainable=False)

        if self.architecture == 'I3':
            print('Using Inception v3 architecture.')
            logits, nett, _ = models.inceptionv3(x_in,
                                                 num_classes=self.n_class, is_training=is_train,
                                                 dropout_keep_prob=self.dropout, scope='InceptionV3')
        elif self.architecture == 'IR2':
            print('Using Inception-Resnet v2 architecture.')
            logits, nett, _ = models.inceptionresnetv2(x_in,
                                                       num_classes=self.n_class, is_training=is_train,
                                                       dropout_keep_prob=self.dropout, scope='InceptionResV2')
        else:
            print('Using default architecture: Inception V3.')
            logits, nett, _ = models.inceptionv3(x_in,
                                                 num_classes=self.n_class, is_training=is_train,
                                                 dropout_keep_prob=self.dropout, scope='InceptionV3')

        pred = tf.nn.softmax(logits, name="prediction")
        cost = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

        tf.summary.scalar("{}_cost".format(self.architecture), cost)
        tf.summary.tensor_summary("{}_pred".format(self.architecture), pred)

        # optimizer based on TensorFlow version
        if int(str(tf.__version__).split('.', 3)[0]) == 2:
            opt = tf.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        train_op = opt.minimize(loss=cost, global_step=global_step)
        merged_summary = tf.summary.merge_all()

        return (x_in, y_in, is_train,
                logits, pred, cost,
                global_step, train_op, merged_summary)

    def inference(self, inf_iter, verbose=True):
        pred = []
        next_batch = inf_iter.get_next()
        while True:
            try:
                X, _ = self.sesh.run(next_batch)
                feed = {self.x_in: X, self.training_status: False}
                batch_pred, i = self.sesh.run(feed_dict=feed, fetches=[self.pred, self.global_step])
                pred.extend(batch_pred)

            except tf.errors.OutOfRangeError:
                if verbose:
                    print('end of iteration. {} predictions'.format(str(len(pred))))
                break
        pred = np.asarray(pred)
        print(pred)
        return pred

    def train(self, data_dir, out_dir, slides, top_k=10, n_epoch=10, batch_size=128, save=True):

        if save:
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            outfile = os.path.join(os.path.abspath(out_dir), "{}_MIL_{}_lr_{}_drop_{}".format(
                str(self.datetime), str(self.architecture),
                str(self.learning_rate), str(self.dropout)))

        now = datetime.now().isoformat()[11:]
        print("------- Training begin: {} -------\n".format(now))

        try:
            for epoch in range(n_epoch):
                """
                Inference run: get top score tiles from each slide
                """
                train_subsets = []  # list of selected tiles, each element is a tf.Dataset with features: ((image, label), pred)
                for slide in slides:
                    # s_id = slide.split('.')[0]
                    slide_data = data_input.DataSet(inputs=[data_dir + '/' + slide], batch_size=batch_size)
                    slide_iter = slide_data.batch_iter()
                    pred = self.inference(slide_iter, verbose=False)
                    pred_1 = pred[:,1]
                    threshold = pred_1[pred_1.argsort()][-top_k]
                    slide_pred = tf.data.Dataset.from_tensor_slices((pred_1))
                    data_pred = tf.data.Dataset.zip((slide_data.get_data(), slide_pred))

                    def filter_fn(data, pred_value):  # nested function to subset data based on current model inference
                        keep = pred_value > threshold
                        return keep

                    filtered_data = data_pred.filter(filter_fn)
                    train_subsets.extend([filtered_data])

                train_data = train_subsets[0]
                if len(train_subsets) > 1:
                    for i in range(1, len(train_subsets)):
                        train_data = train_data.concatenate(train_subsets[i])
                train_data = train_data.shuffle(buffer_size=2000)
                train_data = train_data.batch(batch_size, drop_remainder=False)
                train_iter = train_data.make_one_shot_iterator()
                next_batch = train_iter.get_next()

                while True:
                    try:
                        ((train_X, train_Y), _) =self.sesh.run(next_batch)
                        feed = {self.x_in: train_X, self.y_in: train_Y}
                        fetches = [self.merged_summary, self.logits, self.pred,
                                   self.cost, self.global_step, self.train_op]
                        err_train = 0
                        epoch_batch = 0
                        summary, logits, pred, cost, i, _ = self.sesh.run(fetches=fetches, feed_dict=feed)
                        err_train += cost
                        epoch_batch += 1
                    except tf.errors.OutOfRangeError:
                        print('Epoch {} finished.'.format(epoch))
                        print('Global step {}: average train error {}'.format(self.global_step, err_train / epoch_batch))
                        break

                self.epoch_trained = epoch
            try:
                self.train_logger.flush()
                self.train_logger.close()
            except AttributeError:  # not logging
                print('Not logging')

        except KeyboardInterrupt:
            pass

        now = datetime.now().isoformat()[11:]
        print("------- Training end: {} -------\n".format(now), flush=True)
        print('Epochs trained: {}'.format(str(self.epoch_trained)))
        i = self.global_step.eval(session = self.sesh)
        print('Global steps: {}'.format(str(i)))

        if save:
            saver.save(self.sesh, outfile, global_step=None)
            print('Trained model saved to {}'.format(outfile))

        sys.exit(0)
