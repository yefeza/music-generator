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
# import from utils
from .utils import upload_blob
from google.cloud import storage
import os

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
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
        if isinstance(old_model.layers[i], Dense):
            final_layer = old_model.layers[i](d)
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
    d = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Flatten()(d)
    out_class = Dense(9, activation='softmax')(d)
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

def load_evaluator(dimension, bucket_name, download, train_dataset, epochs):
    storage_client = storage.Client(project='ia-devs')
    bucket = storage_client.bucket(bucket_name)
    path = "evaluadores/" + str(dimension[0]) + "-" + str(dimension[1])
    file_name = "evaluadores/" + str(dimension[0]) + "-" + str(dimension[1]) + "/model.h5"
    if download:
        storage_client = storage.Client(project='ia-devs')
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        if not os.path.exists(path):
            os.makedirs(path)
        blob.download_to_filename(file_name)
        model=keras.models.load_model(file_name)
        return model
    else:
        evaluadores=define_evaluator(7)
        if dimension[0]==4:
            model=evaluadores[0]
            batch_size=16
        if dimension[0]==8:
            model=evaluadores[1]
            batch_size=8
        if dimension[0]==16:
            model=evaluadores[2]
            batch_size=4
        if dimension[0]==32:
            model=evaluadores[3]
            batch_size=2
        if dimension[0]==64:
            model=evaluadores[4]
            batch_size=2
        if dimension[0]==128:
            model=evaluadores[5]
            batch_size=2
        if dimension[0]==256:
            model=evaluadores[6]
            batch_size=2
        X=train_dataset[0]
        y=train_dataset[1]
        model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=["accuracy"])
        model.fit(X,y,epochs=epochs, batch_size=batch_size, validation_data=(X,y))
        if not os.path.exists(path):
            os.makedirs(path)
        model.save(file_name)
        upload_blob(bucket_name,file_name,file_name)
        return model
