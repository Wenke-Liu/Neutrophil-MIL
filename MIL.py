import os
import re
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from datetime import datetime
import data_input
import utils
import models
import inception_v3

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
        self.epoch_pretrained = 0

        self.sesh = tf.Session()

        if not meta_graph:  # new model
            self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")

            # build graph
            handles = self._buildGraph()
            for handle in handles:
                tf.add_to_collection(MIL.RESTORE_KEY, handle)
            self.sesh.run(tf.global_variables_initializer())
            self._load_imagenet()

        else:  # restore saved model
            model_datetime, model_name = re.split("_MIL_|_preMIL_", os.path.basename(meta_graph))
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
                os.mkdir(log_dir + '/pretraining')
                os.mkdir(log_dir + '/validation')

            except FileExistsError:
                pass
            self.train_logger = tf.summary.FileWriter(log_dir + '/training', self.sesh.graph)
            self.pretrain_logger = tf.summary.FileWriter(log_dir + '/pretraining', self.sesh.graph)
            self.validation_logger = tf.summary.FileWriter(log_dir + '/validation', self.sesh.graph)

    def _buildGraph(self):
        x_in = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
        y_in = tf.placeholder(tf.int64, shape=[None])
        onehot_labels = tf.one_hot(indices=tf.cast(y_in, tf.int32), depth=self.n_class)
        is_train = tf.placeholder_with_default(False, shape=[], name="is_train")
        global_step = tf.Variable(0, trainable=False)

        if self.architecture == 'I3':
            print('Using Inception v3 architecture.')
            """
            logits, nett, _ = models.inceptionv3(x_in,
                                                 num_classes=self.n_class, is_training=is_train,
                                                 dropout_keep_prob=self.dropout, scope='InceptionV3')
            """
            with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
                logits, _ = inception_v3.inception_v3(x_in,
                                                      num_classes=self.n_class, is_training=is_train,
                                                      dropout_keep_prob=self.dropout)




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

    def _load_imagenet(self):
        checkpoint_exclude_scopes = 'InceptionV3/Logits, InceptionV3/AuxLogits'
        exclusions = [scope.strip()
                      for scope in checkpoint_exclude_scopes.split(',')]
        variables_to_restore = []
        for var in slim.get_model_variables():
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    break
            else:
                variables_to_restore.append(var)
        checkpoint_path = './pretrain_ckpt/' + self.architecture + '.ckpt'
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)
        init_fn(self.sesh)
        print('Load imagenet pretrained weights.')

    def inference(self, img_to_infer):
        img_to_infer = utils.input_preprocessing(img_to_infer, model=self.architecture)
        feed = {self.x_in: img_to_infer, self.training_status: False}
        pred, i = self.sesh.run(feed_dict=feed, fetches=[self.pred, self.global_step])
        return pred

    def iter_inference(self, inf_batch, verbose=True):
        pred = []
        while True:
            try:
                X = self.sesh.run(inf_batch)
                X = utils.input_preprocessing(X, model=self.architecture)
                batch_pred = self.inference(X)
                pred.append(batch_pred)

            except tf.errors.OutOfRangeError:
                if verbose:
                    print('end of iteration. {} predictions'.format(str(len(pred))))
                break

        pred = np.concatenate(pred, axis=0)
        pred = np.asarray(pred)  # pred is an array of n_predictions by n_class
        # print(pred)
        return pred

    def pre_train(self, pretrain_data_path, out_dir, valid_data_path=None, n_epoch=10, batch_size=128, save=True):
        """
        Pretrain the model with tile level labels before MIL (the train method below)
        """
        pretrain_data = data_input.DataSet(inputs=pretrain_data_path, batch_size=batch_size)
        if save:
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            outfile = os.path.join(os.path.abspath(out_dir), "{}_preMIL_{}_lr_{}_drop_{}".format(
                str(self.datetime), str(self.architecture),
                str(self.learning_rate), str(self.dropout)))

        valid_costs = []
        pretrain_iter = pretrain_data.shuffled_iter()
        next_pretrn_batch = pretrain_iter.get_next()

        if valid_data_path:
            preval_data = data_input.DataSet(inputs=valid_data_path, batch_size=batch_size)
            preval_iter = preval_data.shuffled_iter()
            next_preval_batch = preval_iter.get_next()

        now = datetime.now().strftime(r"%y-%m-%d %H:%M:%S")

        print("------- Pre-training begin: {} -------\n".format(now))
        try:
            for epoch in range(n_epoch):

                err_train = 0
                epoch_batch = 0
                self.sesh.run(pretrain_iter.initializer)

                while True:
                    try:
                        pretrain_X, pretrain_Y = self.sesh.run(next_pretrn_batch)
                        pretrain_X = utils.input_preprocessing(pretrain_X, model=self.architecture)
                        feed = {self.x_in: pretrain_X, self.y_in: pretrain_Y}
                        fetches = [self.merged_summary, self.logits, self.pred,
                                   self.cost, self.global_step, self.train_op]
                        summary, logits, pred, cost, i, _ = self.sesh.run(fetches=fetches, feed_dict=feed)
                        self.train_logger.add_summary(summary, i)
                        err_train += cost
                        epoch_batch += 1
                    except tf.errors.OutOfRangeError:
                        i = self.global_step.eval(session=self.sesh)
                        print('Epoch {} finished.'.format(epoch))
                        print('Global step {}: average train error {}'.format(i, err_train / epoch_batch))
                        break
                    self.epoch_pretrained = epoch

                if valid_data_path:
                    self.sesh.run(preval_iter.initializer)
                    valid_X, valid_Y = self.sesh.run(next_preval_batch)
                    valid_X = utils.input_preprocessing(valid_X, model=self.architecture)
                    feed = {self.x_in: valid_X, self.y_in: valid_Y, self.training_status: False}
                    fetches = [self.merged_summary, self.pred,
                                   self.cost, self.global_step]
                    summary, pred, cost, i = self.sesh.run(fetches=fetches, feed_dict=feed)
                    self.validation_logger.add_summary(summary, i)
                    print('Tile pre-training epoch {} validation cost: {}'.format(self.epoch_pretrained, cost))
                    valid_costs.append(cost)
                    min_valid_cost = min(valid_costs)

                    if cost > min_valid_cost:
                        print('Validation cost reached plateau. Pre-training stopped.')
                        break

            try:
                self.pretrain_logger.flush()
                self.pretrain_logger.close()
                self.validation_logger.flush()
                self.validation_logger.close()

            except AttributeError:  # not logging
                print('Not logging')

        except KeyboardInterrupt:
            pass

        now = datetime.now().strftime(r"%y-%m-%d %H:%M:%S")
        print("------- Pre-training end: {} -------\n".format(now), flush=True)
        print('Epochs trained: {}'.format(str(self.epoch_pretrained)))
        i = self.global_step.eval(session=self.sesh)
        print('Global steps: {}'.format(str(i)))

        if save:
            saver.save(self.sesh, outfile, global_step=None)
            print('Pre-trained model saved to {}'.format(outfile))

    def train(self, data_dir, out_dir, slides, top_k=10,
              valid_data_path=None, sample_rate=None, n_epoch=10, batch_size=128, save=True):

        if save:
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            outfile = os.path.join(os.path.abspath(out_dir), "{}_MIL_{}_lr_{}_drop_{}".format(
                str(self.datetime), str(self.architecture),
                str(self.learning_rate), str(self.dropout)))

        now = datetime.now().strftime(r"%y-%m-%d %H:%M:%S")

        print("------- Training begin: {} -------\n".format(now))

        slide_fn = tf.placeholder(tf.string, shape=None)
        slide_dataset = data_input.DataSet(inputs=slide_fn, batch_size=64)
        rand_ph = tf.placeholder(tf.float32, shape=None)

        def sample_slide(ds, rand, rate):  # random sample from the whole slide, based on the sample_rate argument
            if not sample_rate:
                iter = ds.shuffled_iter()

            else:
                def sample_fn(data, rind):  # random sample from the whole slide, based on the sample_rate argument
                    return rind < rate

                dat = ds.get_data()
                rds = tf.data.Dataset.from_tensor_slices(rand)
                dat_sample = tf.data.Dataset.zip((dat, rds))
                dat_sample = dat_sample.filter(sample_fn)
                dat_sample = dat_sample.map(lambda dat_, rds_: dat_)
                iter = dat_sample.batch(batch_size=batch_size, drop_remainder=False).make_initializable_iterator()

            return iter

        slide_tfr_iter = sample_slide(slide_dataset, rand_ph, sample_rate)
        next_tfr_batch = slide_tfr_iter.get_next()

        trn_img_ph = tf.placeholder(tf.uint8)
        trn_lab_ph = tf.placeholder(tf.int64)
        trn_ds = tf.data.Dataset.from_tensor_slices((trn_img_ph, trn_lab_ph))
        trn_ds = trn_ds.shuffle(buffer_size=2000).batch(batch_size=batch_size)
        trn_iter = trn_ds.make_initializable_iterator()
        next_trn_batch = trn_iter.get_next()

        if valid_data_path:
            valid_data = data_input.DataSet(inputs=valid_data_path, batch_size=64)
            valid_iter = valid_data.shuffled_iter()
            next_val_batch = valid_iter.get_next()

        try:
            for epoch in range(n_epoch):
                """
                Inference run: get top score tiles from each slide
                """
                now = datetime.now().strftime(r"%y-%m-%d %H:%M:%S")
                print('----------epoch {}: {}----------'.format(epoch, now))
                trn_img_subsets = []
                trn_lab_subsets = []

                for slide in slides:
                    # s_id = slide.split('.')[0]
                    slide_prob = []
                    slide_img = []
                    slide_lab = []
                    slide_counter = 0
                    slide_path = data_dir + '/' + slide

                    self.sesh.run(slide_tfr_iter.initializer,
                                  feed_dict={slide_fn: slide_path, rand_ph: np.random.uniform(0., 1., 200000)})

                    while True:
                        try:
                            imgs, labs = self.sesh.run(next_tfr_batch)
                            batch_pred = self.inference(imgs)[:, 1]
                            batch_top_ind = batch_pred.argsort()[-top_k:]  # index of largest k probabilities
                            slide_counter += imgs.shape[0]

                            for top_ind in batch_top_ind:
                                top_prob = batch_pred[top_ind]
                                #print(top_ind)
                                #print(top_prob)
                                if (slide_counter <= top_k*batch_size or
                                        top_prob >= np.sort(np.array(slide_prob))[-top_k]):
                                    slide_prob.append(batch_pred[top_ind])
                                    slide_img.append(imgs[top_ind])
                                    slide_lab.append(labs[top_ind])
                                else:
                                    pass

                        except tf.errors.OutOfRangeError:
                            break

                        slide_prob_ind = np.asarray(slide_prob).argsort()[:-top_k]
                        for ind in sorted(slide_prob_ind, reverse=True):
                            del slide_prob[ind]
                            del slide_img[ind]
                            del slide_lab[ind]

                    print('{}: {} tiles inferred from slide.'.format(slide, slide_counter))
                    print('Top {} probabilities: '.format(top_k))
                    print(slide_prob)

                    for i in range(len(slide_prob)):
                        trn_img_subsets.append(slide_img[i])
                        trn_lab_subsets.append(slide_lab[i])

                    print('Filtered images: {}'.format(len(trn_img_subsets)))
                    #print('Filtered labels:{}'.format(len(lab_subsets)))

                trn_img_subsets = np.asarray(trn_img_subsets)
                trn_lab_subsets = np.asarray(trn_lab_subsets)
                self.sesh.run(trn_iter.initializer, feed_dict={trn_img_ph: trn_img_subsets,
                                                               trn_lab_ph: trn_lab_subsets})

                err_train = 0
                epoch_batch = 0

                while True:
                    try:
                        train_X, train_Y =self.sesh.run(next_trn_batch)
                        train_X = utils.input_preprocessing(train_X, model=self.architecture)
                        feed = {self.x_in: train_X, self.y_in: train_Y}
                        fetches = [self.merged_summary, self.logits, self.pred,
                                   self.cost, self.global_step, self.train_op]
                        summary, logits, pred, cost, i, _ = self.sesh.run(fetches=fetches, feed_dict=feed)
                        err_train += cost
                        epoch_batch += 1
                    except tf.errors.OutOfRangeError:
                        print('MIL training epoch {} finished.'.format(epoch))
                        print('Global step {}: average train error {}'.format(i, err_train / epoch_batch))
                        break

                self.epoch_trained = epoch

                if valid_data_path:
                    self.sesh.run(valid_iter.initializer)
                    valid_X, valid_Y = self.sesh.run(next_val_batch)
                    valid_X = utils.input_preprocessing(valid_X, model=self.architecture)
                    feed = {self.x_in: valid_X, self.y_in: valid_Y, self.training_status: False}
                    fetches = [self.merged_summary, self.pred,
                               self.cost, self.global_step]
                    summary, pred, cost, i = self.sesh.run(fetches=fetches, feed_dict=feed)
                    self.validation_logger.add_summary(summary, i)
                    print('MIL training epoch {} validation cost: {}'.format(self.epoch_trained, cost))
                if save:
                    saver.save(self.sesh, outfile, global_step=None)
                    print('Trained model saved to {}'.format(outfile))

            try:
                self.train_logger.flush()
                self.train_logger.close()
                self.validation_logger.flush()
                self.validation_logger.close()

            except AttributeError:  # not logging
                print('Not logging')

        except KeyboardInterrupt:
            pass

        now = datetime.now().strftime(r"%y-%m-%d %H:%M:%S")
        print("------- Training end: {} -------\n".format(now), flush=True)
        print('Epochs trained: {}'.format(str(self.epoch_trained)))
        i = self.global_step.eval(session = self.sesh)
        print('Global steps: {}'.format(str(i)))

        if save:
            saver.save(self.sesh, outfile, global_step=None)
            print('Trained model saved to {}'.format(outfile))
