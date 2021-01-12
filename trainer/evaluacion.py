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

# agregar bloque a evaluador para escalar las dimensiones

def add_evaluator_block(old_model, n_input_layers=3):
    # getshapeofexistingmodel
    in_shape = list(old_model.input[0].shape)
    # definenewinputshapeasdoublethesize
    input_shape = (in_shape[-3]*2, in_shape[-2]*2, in_shape[-1])
    in_image = Input(shape=input_shape)
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
    model = Model(in_image, final_layer)
    return model

# definir los evaluadores

def define_evaluator(n_blocks, input_shape=(4, 750, 2)):
    model_list = list()
    # base model input
    in_image = Input(shape=input_shape)
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
    wls = Dense(100)(d)
    out_class = Dense(9, activation='softmax')(wls)
    # define model
    model = Model(in_image, out_class)
    # store model
    model_list.append(model)
    # create submodels
    for i in range(1, n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i - 1]
        # create new model for next resolution
        new_model = add_evaluator_block(old_model)
        # store model
        model_list.append(new_model)
    return model_list

#descargar evaluadores

def load_evaluator(dimension, job_dir, download, train_dataset, epochs):
    path = job_dir + "/evaluadores/" + str(dimension[0]) + "-" + str(dimension[1])
    if download:
        model=tf.compat.v1.keras.experimental.load_from_saved_model(path)
        return model
    else:
        evaluadores=define_evaluator(7)
        if dimension[0]==4:
            model=evaluadores[0]
        if dimension[0]==8:
            model=evaluadores[1]
        if dimension[0]==16:
            model=evaluadores[2]
        if dimension[0]==32:
            model=evaluadores[3]
        if dimension[0]==64:
            model=evaluadores[4]
        if dimension[0]==128:
            model=evaluadores[5]
        if dimension[0]==256:
            model=evaluadores[6]
        X=train_dataset[0]
        y=train_dataset[1]
        model.compile(optimizer="Adam", loss='sparse_categorical_crossentropy', metrics=["accuracy", "loss"])
        model.fit(X,y,epochs=epochs, validation_split=0.2)
        export_path = tf.compat.v1.keras.experimental.export_saved_model(model, path)
        print('Model exported to: {}'.format(export_path))
        return model

# calculate the inception score for p(y|x)
def calculate_inception_score(p_yx, eps=1E-16):
    # calculate p(y)
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    # undo the logs
    is_score = np.exp(avg_kl_d)
    return is_score