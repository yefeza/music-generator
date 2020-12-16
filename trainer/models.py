# keras imports
import keras
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Input, UpSampling2D, Layer
from keras.layers import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv1D, Conv2DTranspose, Conv3DTranspose
from keras.layers.convolutional import MaxPooling2D, MaxPooling1D, AveragePooling1D, AveragePooling2D
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import LeakyReLU
from keras.layers import Add
from keras.utils.vis_utils import plot_model
from keras import backend
import tensorflow as tf
from keras.losses import categorical_crossentropy
import numpy as np

# WGAN + ACGAN  funcion loss del generador


def G_wgan_acgan(y_true, y_pred):
    cond_weight = 1.0
    fake_scores_out = y_pred
    loss = -fake_scores_out
    label_penalty_fakes = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
            labels=y_true, logits=fake_scores_out)
    print(y_true)
    print(y_pred)
    loss += label_penalty_fakes * cond_weight
    return loss

# WGANGP + ACGAN  funcion loss del discriminador


def D_wgangp_acgan(y_true, y_pred, gradient_penalty):
    wgan_lambda = 10.0      # Weight for the gradient penalty term.
    wgan_epsilon = 0.001     # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target = 1.0       # Target value for gradient magnitudes.
    cond_weight = 1.0       # Weight of the conditioning terms.
    half_batch = y_true.get_shape()[0]
    if half_batch == None:
        half_batch = 1
    else:
        half_batch = half_batch/2
    y_true_real_images = y_true[:half_batch]
    y_true_fake_images = y_true[half_batch:]
    real_scores_out = y_pred[:half_batch]
    fake_scores_out = y_pred[half_batch:]
    loss = fake_scores_out - real_scores_out
    gradient_penalty = gradient_penalty[0][0]
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))
    epsilon_penalty = tf.square(real_scores_out)
    loss += epsilon_penalty * wgan_epsilon
    label_penalty_reals = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
            labels=y_true_real_images, logits=real_scores_out)
    label_penalty_fakes = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
            labels=y_true_fake_images, logits=fake_scores_out)
    loss += (label_penalty_reals + label_penalty_fakes) * cond_weight
    return loss

# Weighted Sum Layer para el proceso de fade-in


class WeightedSum(Add):
    # init with default value
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = backend.variable(alpha, name='ws_alpha')

    # output a weighted sum of inputs
    def _merge_function(self, inputs):
        # only supports a weighted sum of two inputs
        assert (len(inputs) == 2)
        # ((1-a) * input1) + (a * input2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output

# Minibatch Standard Deviation Layer


class MinibatchStdDev(Layer):
    def call(self, inputs):
        group_size = 4
        x = inputs
        with tf.compat.v1.variable_scope('MinibatchStddev'):
            # Minibatch must be divisible by (or smaller than) group_size.
            group_size = tf.minimum(group_size, tf.shape(x)[0])
            # [NCHW]  Input shape.
            s = x.shape
            # [GMCHW] Split minibatch into M groups of size G.
            y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])
            # [GMCHW] Cast to FP32.
            y = tf.cast(y, tf.float32)
            # [GMCHW] Subtract mean over group.
            y -= tf.reduce_mean(y, axis=0, keepdims=True)
            # [MCHW]  Calc variance over group.
            y = tf.reduce_mean(tf.square(y), axis=0)
            # [MCHW]  Calc stddev over group.
            y = tf.sqrt(y + 1e-8)
            # [M111]  Take average over fmaps and pixels.
            y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)
            # [M111]  Cast back to original data type.
            y = tf.cast(y, x.dtype)
            # [N1HW]  Replicate over group and pixels.
            y = tf.tile(y, [group_size, 1, s[2], s[3]])
            # [NCHW]  Append as new fmap.
            return tf.concat([x, y], axis=1)


# agregar bloque a discriminador para escalar las dimensiones

def add_discriminator_block(old_model, n_input_layers=3):
    # getshapeofexistingmodel
    in_shape = list(old_model.input[0].shape)
    # definenewinputshapeasdoublethesize
    input_shape = (in_shape[-3]*2, in_shape[-2]*2, in_shape[-1])
    in_image = Input(shape=input_shape)
    y_true = Input(shape=(1,))
    is_weight = Input(shape=(1,))
    # definenewinputprocessinglayer
    d = Conv2D(64, (1, 1), padding='same',
               kernel_initializer='he_normal')(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # definenewblock
    d = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = AveragePooling2D()(d)
    block_new = d
    first_dense=True
    # skiptheinput,1x1andactivationfortheoldmodel
    for i in range(n_input_layers, len(old_model.layers)):
        if isinstance(old_model.layers[i], Dense):
            if first_dense:
                d = old_model.layers[i](d)
                first_dense=False
            else:
                final_layer = old_model.layers[i](d)
                break
        else:
            d = old_model.layers[i](d)
    # definestraight-throughmodel
    model1 = Model([in_image, y_true, is_weight], final_layer)
    model1.add_loss(D_wgangp_acgan(y_true, final_layer, is_weight))
    # model 1 without multiple inputs for composite
    model1_comp = Model(in_image, final_layer)
    # compilemodel
    model1.compile(loss=None, optimizer=Adam(
        lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    # downsamplethenewlargerimage
    downsample = AveragePooling2D()(in_image)
    # connectoldinputprocessingtodownsamplednewinput
    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)
    # fadeinoutputofoldmodelinputlayerwithnewinput
    d = WeightedSum()([block_old, block_new])
    # skiptheinput,1x1andactivationfortheoldmodel
    first_dense=True
    for i in range(n_input_layers, len(old_model.layers)):
        if isinstance(old_model.layers[i], Dense):
            if first_dense:
                d = old_model.layers[i](d)
                first_dense=False
            else:
                final_layer = old_model.layers[i](d)
                break
        else:
            d = old_model.layers[i](d)
    # definestraight-throughmodel
    model2 = Model([in_image, y_true, is_weight], final_layer)
    model2.add_loss(D_wgangp_acgan(y_true, final_layer, is_weight))
    # model 2 without multiple inputs for composite
    model2_comp = Model(in_image, final_layer)
    # compilemodel
    model2.compile(loss=None, optimizer=Adam(
        lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    return[model1, model2, model1_comp, model2_comp]

# definir los discriminadores


def define_discriminator(n_blocks, lstm_layer, input_shape=(4, 750, 2)):
    model_list = list()
    # base model input
    in_image = Input(shape=input_shape)
    y_true = Input(shape=(1,))
    is_weight = Input(shape=(1,))
    # conv 1x1
    d = Conv2D(64, (1, 1), padding='same',
               kernel_initializer='he_normal')(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # conv 3x3 (output block)
    d = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(d)
    d = LeakyReLU(alpha=0.2)(d)
    # conv 4x4
    d = Conv2D(128, (4, 4), padding='same', kernel_initializer='he_normal')(d)
    d = MinibatchStdDev()(d)
    d = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(d)
    d = LeakyReLU(alpha=0.2)(d)
    # lstm layer
    d = Flatten()(d)
    wls = lstm_layer(d)
    out_class = Dense(1, activation='linear')(wls)
    # define model
    model = Model([in_image, y_true, is_weight], out_class)
    model.add_loss(D_wgangp_acgan(y_true, out_class, is_weight))
    # compile model
    model.compile(loss=None, optimizer=Adam(
        lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    model_comp = Model(in_image, out_class)
    # store model
    model_list.append([model, model, model_comp, model_comp])
    # create submodels
    for i in range(1, n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_discriminator_block(old_model)
        # store model
        model_list.append(models)
    return model_list

# agregar bloque a generador para escalar las dimensiones


def add_generator_block(old_model):
    # get the end of the last block
    block_end = old_model.layers[-2].output
    # upsample, and define new block
    upsampling = UpSampling2D()(block_end)
    g = Conv2D(128, (3, 3), padding='same',
               kernel_initializer='he_normal')(upsampling)
    g = LeakyReLU(alpha=0.2)(g)
    g = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(g)
    g = LeakyReLU(alpha=0.2)(g)
    out_image = Conv2DTranspose(2, (1, 1), padding='same')(g)
    # define model
    model1 = Model(old_model.input, out_image)
    # get the output layer from old model
    out_old = old_model.layers[-1]
    # connect the upsampling to the old output layer
    out_image2 = out_old(upsampling)
    # define new output image as the weighted sum of the old and new models
    merged = WeightedSum()([out_image2, out_image])
    # define model
    model2 = Model(old_model.input, merged)
    return [model1, model2]

# definir los generadores


def define_generator(n_blocks, lstm_layer):
    model_list = list()
    # input
    ly0 = Input(shape=(1, 5, 2))
    # bloque 1 deconvolusion
    g = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='valid')(ly0)
    g = LeakyReLU(alpha=0.2)(g)
    g = Conv2DTranspose(64, (2, 5), strides=(2, 5), padding='valid')(g)
    g = LeakyReLU(alpha=0.2)(g)
    g = Conv2DTranspose(128, (1, 15), strides=(1, 15), padding='valid')(g)
    g = LeakyReLU(alpha=0.2)(g)
    g = Conv2DTranspose(160, (1, 1), padding='same')(g)
    g = LeakyReLU(alpha=0.2)(g)
    out_image = Flatten()(g)
    g_lstm_layer = lstm_layer(out_image)
    g_lstm_layer.trainable=False
    wls = Reshape(target_shape=(1, 50, 2))(g_lstm_layer)
    wls = Conv2DTranspose(2, (1, 15), strides=(1, 15), padding='valid')(wls)
    wls = LeakyReLU(alpha=0.2)(wls)
    wls = Conv2DTranspose(128, (4, 1), strides=(4, 1), padding='valid')(wls)
    wls = LeakyReLU(alpha=0.2)(wls)
    wls = Conv2DTranspose(2, (1, 1), padding='same')(wls)
    wls = LeakyReLU(alpha=0.2)(wls)
    model = Model(ly0, wls)
    # store model
    model_list.append([model, model])
    # create submodels
    for i in range(1, n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_generator_block(old_model)
        # store model
        model_list.append(models)
    return model_list

# define composite models for training generators via discriminators


def define_composite(discriminators, generators):
    model_list = list()
    # create composite models
    for i in range(len(discriminators)):
        g_models, d_models = generators[i], discriminators[i]
        # straight-through model
        d_models[2].trainable = False
        model1 = Sequential()
        model1.add(g_models[0])
        model1.add(d_models[2])
        model1.compile(loss=G_wgan_acgan, optimizer=Adam(
            lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        # fade-in model
        d_models[3].trainable = False
        model2 = Sequential()
        model2.add(g_models[1])
        model2.add(d_models[3])
        model2.compile(loss=G_wgan_acgan, optimizer=Adam(
            lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        # store
        model_list.append([model1, model2])
    return model_list
