import tensorflow as tf
import numpy as np

"""
Concstruct tensorflow dataset object from input TFrecord file
Return a tf.Dataset object with features: image, label
"""


class DataSet(object):

    def __init__(self, inputs, batch_size=128):
        self.batch_size = batch_size
        if all(isinstance(elem, str) for elem in inputs):  # if all inputs are filename strings
            self._dataset = tf.data.TFRecordDataset(filenames=inputs)
        else:  # if inputs are TFRecords datasets
            self._dataset = inputs[0]
            if len(inputs)>1:
                for i in range(1,len(inputs)):
                    self._dataset = self._dataset.concatenate(inputs[i])

    def decode_example(self, example):
        feature_description = {'dim': tf.FixedLenFeature([], tf.string),
                               'image': tf.FixedLenFeature([], tf.string),
                               'label': tf.FixedLenFeature([], tf.int64)}

        def _parse_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            return tf.parse_single_example(example_proto, feature_description)

        parsed_example = _parse_function(example)
        dim = tf.decode_raw(parsed_example['dim'], tf.int64)
        image = tf.decode_raw(parsed_example['image'], tf.float32)
        label = tf.cast(parsed_example['label'], tf.int32)
        image = tf.reshape(image, dim)

        return image, label

    def get_data(self):  # return decoded datasets
        parsed_dataset = self._dataset.map(self.decode_example)
        return parsed_dataset

    def get_data_batch(self):
        parsed_dataset = self._dataset.map(self.decode_example)
        batched_dataset = parsed_dataset.batch(batch_size=self.batch_size, drop_remainder=False)
        return batched_dataset

    def data_iter(self):
        parsed_dataset = self._dataset.map(self.decode_example)
        return parsed_dataset.make_one_shot_iterator()

    def batch_iter(self):
        parsed_dataset = self._dataset.map(self.decode_example)
        batched_dataset = parsed_dataset.batch(batch_size=self.batch_size, drop_remainder=False)
        return batched_dataset.make_one_shot_iterator()

    def get_raw(self):
        return self._dataset

    def get_size(self):
        i = 0
        count_iter = self._dataset.make_one_shot_iterator()
        next_element = count_iter.get_next()
        with tf.Session() as sess:
            try:
                while True:
                    sess.run(next_element)
                    i += 1
            except tf.errors.OutOfRangeError:
                pass
        return i
