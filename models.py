import tensorflow as tf
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate, add
from keras.regularizers import l2
import numpy as np


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name,
        data_format="channels_last",
        kernel_regularizer=l2(0.0002))(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def inceptionv3(input, dropout_keep_prob=0.8, num_classes=1000, is_training=True,
                scope='InceptionV3', channel_axis=3):

    with tf.name_scope(scope, "InceptionV3", [input]):

        x = conv2d_bn(input, 32, 3, 3, strides=(2, 2), padding='valid')
        x = conv2d_bn(x, 32, 3, 3, padding='valid')
        x = conv2d_bn(x, 64, 3, 3)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv2d_bn(x, 80, 1, 1, padding='valid')
        x = conv2d_bn(x, 192, 3, 3, padding='valid')
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # mixed 0: 35 x 35 x 256
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3),
                                              strides=(1, 1),
                                              padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
        x = concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed0')

        # mixed 1: 35 x 35 x 288
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3),
                                              strides=(1, 1),
                                              padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
        x = concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed1')

        # mixed 2: 35 x 35 x 288
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3),
                                              strides=(1, 1),
                                              padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
        x = concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed2')

        # mixed 3: 17 x 17 x 768
        branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(
            branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate(
            [branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed3')

        # mixed 4: 17 x 17 x 768
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 128, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 128, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3),
                                              strides=(1, 1),
                                              padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed4')

        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = conv2d_bn(x, 192, 1, 1)

            branch7x7 = conv2d_bn(x, 160, 1, 1)
            branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
            branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = conv2d_bn(x, 160, 1, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
            x = concatenate(
                [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                axis=channel_axis,
                name='mixed' + str(5 + i))

        # mixed 7: 17 x 17 x 768
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 192, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 192, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3),
                                              strides=(1, 1),
                                              padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed7')

        loss2_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss2/ave_pool')(x)

        loss2_conv_a = conv2d_bn(loss2_ave_pool, 128, 1, 1)
        loss2_conv_b = conv2d_bn(loss2_conv_a, 768, 5, 5)

        loss2_flat = Flatten()(loss2_conv_b)

        loss2_fc = Dense(1024, activation='relu', name='loss2/fc', kernel_regularizer=l2(0.0002))(loss2_flat)

        loss2_drop_fc = Dropout(dropout_keep_prob)(loss2_fc, training=is_training)

        loss2_classifier = Dense(num_classes, name='loss2/classifier', kernel_regularizer=l2(0.0002))(loss2_drop_fc)

        # mixed 8: 8 x 8 x 1280
        branch3x3 = conv2d_bn(x, 192, 1, 1)
        branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                              strides=(2, 2), padding='valid')

        branch7x7x3 = conv2d_bn(x, 192, 1, 1)
        branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
        branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
        branch7x7x3 = conv2d_bn(
            branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate(
            [branch3x3, branch7x7x3, branch_pool],
            axis=channel_axis,
            name='mixed8')

        # mixed 9: 8 x 8 x 2048
        for i in range(2):
            branch1x1 = conv2d_bn(x, 320, 1, 1)

            branch3x3 = conv2d_bn(x, 384, 1, 1)
            branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
            branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
            branch3x3 = concatenate(
                [branch3x3_1, branch3x3_2],
                axis=channel_axis,
                name='mixed9_' + str(i))

            branch3x3dbl = conv2d_bn(x, 448, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
            branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
            branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
            branch3x3dbl = concatenate(
                [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

            branch_pool = AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
            x = concatenate(
                [branch1x1, branch3x3, branch3x3dbl, branch_pool],
                axis=channel_axis,
                name='mixed' + str(9 + i))
        net = x

        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)

        pool5_drop_10x10_s1 = Dropout(dropout_keep_prob)(x, training=is_training)

        loss3_classifier_w = Dense(num_classes, name='loss3/classifier', kernel_regularizer=l2(0.0002))

        loss3_classifier = loss3_classifier_w(pool5_drop_10x10_s1)

        w_variables = loss3_classifier_w.get_weights()

        logits = tf.cond(tf.equal(is_training, tf.constant(True)),
                         lambda: tf.add(loss3_classifier, tf.scalar_mul(tf.constant(0.3), loss2_classifier)),
                         lambda: loss3_classifier)

    return logits, net, tf.convert_to_tensor(w_variables[0])

def resnet_v2_stem(input):
    '''The stem of the pure Inception-v4 and Inception-ResNet-v2 networks. This is input part of those networks.'''

    # Input shape is 299 * 299 * 3 (Tensorflow dimension ordering)
    x = Conv2D(32, (3, 3), kernel_regularizer=l2(0.0002), activation="relu", strides=(2, 2))(input)  # 149 * 149 * 32
    x = Conv2D(32, (3, 3), kernel_regularizer=l2(0.0002), activation="relu")(x)  # 147 * 147 * 32
    x = Conv2D(64, (3, 3), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(x)  # 147 * 147 * 64

    x1 = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x2 = Conv2D(96, (3, 3), kernel_regularizer=l2(0.0002), activation="relu", strides=(2, 2))(x)

    x = concatenate([x1, x2], axis=3)  # 73 * 73 * 160

    x1 = Conv2D(64, (1, 1), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(x)
    x1 = Conv2D(96, (3, 3), kernel_regularizer=l2(0.0002), activation="relu")(x1)

    x2 = Conv2D(64, (1, 1), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(x)
    x2 = Conv2D(64, (7, 1), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(x2)
    x2 = Conv2D(64, (1, 7), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(x2)
    x2 = Conv2D(96, (3, 3), kernel_regularizer=l2(0.0002), activation="relu", padding="valid")(x2)

    x = concatenate([x1, x2], axis=3)  # 71 * 71 * 192

    x1 = Conv2D(192, (3, 3), kernel_regularizer=l2(0.0002), activation="relu", strides=(2, 2))(x)

    x2 = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = concatenate([x1, x2], axis=3)  # 35 * 35 * 384

    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    return x


def inception_resnet_v2_A(input, scale_residual=True):
    '''Architecture of Inception_ResNet_A block which is a 35 * 35 grid module.'''

    ar1 = Conv2D(32, (1, 1), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(input)

    ar2 = Conv2D(32, (1, 1), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(input)
    ar2 = Conv2D(32, (3, 3), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(ar2)

    ar3 = Conv2D(32, (1, 1), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(input)
    ar3 = Conv2D(48, (3, 3), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(ar3)
    ar3 = Conv2D(64, (3, 3), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(ar3)

    merged = concatenate([ar1, ar2, ar3], axis=3)

    ar = Conv2D(384, (1, 1), kernel_regularizer=l2(0.0002), activation="linear", padding="same")(merged)
    if scale_residual: ar = Lambda(lambda a: a * 0.1)(ar)

    output = add([input, ar])
    output = BatchNormalization(axis=3)(output)
    output = Activation("relu")(output)

    return output


def inception_resnet_v2_B(input, scale_residual=True):
    '''Architecture of Inception_ResNet_B block which is a 17 * 17 grid module.'''

    br1 = Conv2D(192, (1, 1), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(input)

    br2 = Conv2D(128, (1, 1), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(input)
    br2 = Conv2D(160, (1, 7), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(br2)
    br2 = Conv2D(192, (7, 1), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(br2)

    merged = concatenate([br1, br2], axis=3)

    br = Conv2D(1152, (1, 1), kernel_regularizer=l2(0.0002), activation="linear", padding="same")(merged)
    if scale_residual: br = Lambda(lambda b: b * 0.1)(br)

    output = add([input, br])
    output = BatchNormalization(axis=3)(output)
    output = Activation("relu")(output)

    return output


def inception_resnet_v2_C(input, scale_residual=True):
    '''Architecture of Inception_ResNet_C block which is a 8 * 8 grid module.'''

    cr1 = Conv2D(192, (1, 1), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(input)

    cr2 = Conv2D(192, (1, 1), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(input)
    cr2 = Conv2D(224, (1, 3), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(cr2)
    cr2 = Conv2D(256, (3, 1), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(cr2)

    merged = concatenate([cr1, cr2], axis=3)

    cr = Conv2D(2144, (1, 1), kernel_regularizer=l2(0.0002), activation="linear", padding="same")(merged)
    if scale_residual: cr = Lambda(lambda c: c * 0.1)(cr)

    output = add([input, cr])
    output = BatchNormalization(axis=3)(output)
    output = Activation("relu")(output)

    return output


def reduction_resnet_A(input, k=192, l=224, m=256, n=384):
    '''Architecture of a 35 * 35 to 17 * 17 Reduction_ResNet_A block. It is used by both v1 and v2 Inception-ResNets.'''

    rar1 = MaxPooling2D((3, 3), strides=(2, 2))(input)

    rar2 = Conv2D(n, (3, 3), kernel_regularizer=l2(0.0002), activation="relu", strides=(2, 2))(input)

    rar3 = Conv2D(k, (1, 1), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(input)
    rar3 = Conv2D(l, (3, 3), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(rar3)
    rar3 = Conv2D(m, (3, 3), kernel_regularizer=l2(0.0002), activation="relu", strides=(2, 2))(rar3)

    merged = concatenate([rar1, rar2, rar3], axis=3)
    rar = BatchNormalization(axis=3)(merged)
    rar = Activation("relu")(rar)

    return rar


def reduction_resnet_v2_B(input):
    '''Architecture of a 17 * 17 to 8 * 8 Reduction_ResNet_B block.'''

    rbr1 = MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(input)

    rbr2 = Conv2D(256, (1, 1), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(input)
    rbr2 = Conv2D(384, (3, 3), kernel_regularizer=l2(0.0002), activation="relu", strides=(2, 2))(rbr2)

    rbr3 = Conv2D(256, (1, 1), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(input)
    rbr3 = Conv2D(288, (3, 3), kernel_regularizer=l2(0.0002), activation="relu", strides=(2, 2))(rbr3)

    rbr4 = Conv2D(256, (1, 1), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(input)
    rbr4 = Conv2D(288, (3, 3), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(rbr4)
    rbr4 = Conv2D(320, (3, 3), kernel_regularizer=l2(0.0002), activation="relu", strides=(2, 2))(rbr4)

    merged = concatenate([rbr1, rbr2, rbr3, rbr4], axis=3)
    rbr = BatchNormalization(axis=3)(merged)
    rbr = Activation("relu")(rbr)

    return rbr


def inceptionresnetv2(input, dropout_keep_prob=0.8, num_classes=1000, is_training=True,
                      scope='InceptionResnetV2'):
    '''Creates the Inception_ResNet_v2 network.'''
    with tf.variable_scope(scope, 'InceptionResnetV2', [input]):
        # Input shape is 299 * 299 * 3
        x = resnet_v2_stem(input)  # Output: 35 * 35 * 256

        # 5 x Inception A
        for i in range(5):
            x = inception_resnet_v2_A(x)
            # Output: 35 * 35 * 256

        # Reduction A
        x = reduction_resnet_A(x, k=256, l=256, m=384, n=384)  # Output: 17 * 17 * 896

        # 10 x Inception B
        for i in range(10):
            x = inception_resnet_v2_B(x)
            # Output: 17 * 17 * 896

        # auxiliary
        loss2_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss2/ave_pool')(x)

        loss2_conv_a = Conv2D(128, (1, 1), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(
            loss2_ave_pool)
        loss2_conv_b = Conv2D(768, (5, 5), kernel_regularizer=l2(0.0002), activation="relu", padding="same")(
            loss2_conv_a)

        loss2_conv_b = BatchNormalization(axis=3)(loss2_conv_b)

        loss2_conv_b = Activation('relu')(loss2_conv_b)

        loss2_flat = Flatten()(loss2_conv_b)

        loss2_fc = Dense(1024, activation='relu', name='loss2/fc', kernel_regularizer=l2(0.0002))(loss2_flat)

        loss2_drop_fc = Dropout(dropout_keep_prob)(loss2_fc, training=is_training)

        loss2_classifier = Dense(num_classes, name='loss2/classifier', kernel_regularizer=l2(0.0002))(loss2_drop_fc)

        # Reduction B
        x = reduction_resnet_v2_B(x)  # Output: 8 * 8 * 1792

        # 5 x Inception C
        for i in range(5):
            x = inception_resnet_v2_C(x)
            # Output: 8 * 8 * 1792

        net = x

        # Average Pooling
        x = GlobalAveragePooling2D(name='avg_pool')(x)  # Output: 1792

        pool5_drop_10x10_s1 = Dropout(dropout_keep_prob)(x, training=is_training)

        loss3_classifier_w = Dense(num_classes, name='loss3/classifier', kernel_regularizer=l2(0.0002))

        loss3_classifier = loss3_classifier_w(pool5_drop_10x10_s1)

        w_variables = loss3_classifier_w.get_weights()

        logits = tf.cond(tf.equal(is_training, tf.constant(True)),
                         lambda: tf.add(loss3_classifier, tf.scalar_mul(tf.constant(0.3), loss2_classifier)),
                         lambda: loss3_classifier)

        return logits, net, tf.convert_to_tensor(w_variables[0])
