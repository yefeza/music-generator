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
from .models import *

# agregar bloque a evaluador para escalar las dimensiones

def add_evaluator_block(old_model, n_input_layers=3):
    # getshapeofexistingmodel
    in_shape = list(old_model.input[0].shape)
    # definenewinputshapeasdoublethesize
    input_shape = (in_shape[-3]*2, in_shape[-2]*2, in_shape[-1])
    in_image = Input(shape=input_shape)
    featured_layer = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal')(in_image)
    #featured_layer = LeakyReLU(alpha=0.2)(featured_layer)
    #convolusion block 1
    d_1 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(featured_layer)
    #d_1 = LeakyReLU(alpha=0.2)(d_1)
    d_1 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(d_1)
    #d_1 = LeakyReLU(alpha=0.2)(d_1)
    #d = AveragePooling2D()(d)
    d_1 = Conv2D(128, (2, 2), padding='same', kernel_initializer='he_normal')(d_1)
    #d_1 = LeakyReLU(alpha=0.2)(d_1)
    #convolusion block 2
    d_2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(d_1)
    #d_2 = LeakyReLU(alpha=0.2)(d_2)
    d_2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(d_2)
    #d_2 = LeakyReLU(alpha=0.2)(d_2)
    d_2 = Conv2D(128, (2, 2), padding='same', kernel_initializer='he_normal')(d_2)
    #d_2 = LeakyReLU(alpha=0.2)(d_2)
    #convolusion block 3
    d_3 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(d_2)
    #d_3 = LeakyReLU(alpha=0.2)(d_3)
    d_3 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(d_3)
    #d_3 = LeakyReLU(alpha=0.2)(d_3)
    d_3 = Conv2D(128, (2, 2), strides=(2,2), padding='valid', kernel_initializer='he_normal')(d_3)
    #d_3 = LeakyReLU(alpha=0.2)(d_3)
    d_block=Conv2D(128, (1,1), padding='same', kernel_initializer='he_normal')(d_3)
    #d_block = LeakyReLU(alpha=0.2)(d_block)
    for i in range(n_input_layers, len(old_model.layers)):
        if isinstance(old_model.layers[i], Dense):
            final_layer = old_model.layers[i](d_block)
        else:
            d_block = old_model.layers[i](d_block)
    model = Model(in_image, final_layer)
    return model

# definir los evaluadores

def define_evaluator(n_blocks, input_shape=(3000, 2)):
    model_list = list()
    # base model input
    in_data = Input(shape=input_shape)
    converted_block = Reshape((4,750,2))(in_data)
    converted_block = ToMonoChannel()(converted_block)
    converted_block = FFT()(converted_block)
    converted_block = FreqChannelChange()(converted_block)
    # convolusion block 1
    d_1 = Dense(64)(converted_block)
    d_1 = TimeToEnd()(d_1)
    d_1 = Dense(1)(d_1)
    d_1 = Flatten()(d_1)
    out_class = Dense(9, activation='softmax')(d_1)
    # define model
    model = Model(in_data, out_class)
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
    custom_layers={
        "DefaultNetwork":DefaultNetwork,
        "WeightedSum":WeightedSum,
        "MinibatchStdDev":MinibatchStdDev,
        "SoftRectifier":SoftRectifier,
        "StaticOptTanh":StaticOptTanh,
        "DecisionLayer":DecisionLayer,
        "DecisionLayer2D":DecisionLayer2D,
        "SlicerLayer":SlicerLayer,
        "ToMonoChannel":ToMonoChannel,
        "FreqToTime":FreqToTime,
        "FreqChannelChange":FreqChannelChange,
        "TimeToEnd":TimeToEnd,
        "FrequencyMagnitude":FrequencyMagnitude,
        "ComplexToChannels":ComplexToChannels,
        "ChannelsToComplex":ChannelsToComplex,
        "FFT":FFT,
        "iFFT":iFFT,
        "LaplaceLayer":LaplaceLayer,
        "LaplaceLayerNonTrain":LaplaceLayerNonTrain
    }
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
        model=keras.models.load_model(file_name, custom_objects=custom_layers)
        return model
    else:
        evaluadores=define_evaluator(1)
        if dimension[0]==4:
            model=evaluadores[0]
            batch_size=64
            epochs=50
        if dimension[0]==8:
            model=evaluadores[1]
            batch_size=32
            epochs=20
        if dimension[0]==16:
            model=evaluadores[2]
            batch_size=16
            epochs=30
        if dimension[0]==32:
            model=evaluadores[3]
            batch_size=8
            epochs=40
        if dimension[0]==64:
            model=evaluadores[4]
            batch_size=4
            epochs=50
        if dimension[0]==128:
            model=evaluadores[5]
            batch_size=2
            epochs=60
        if dimension[0]==256:
            model=evaluadores[6]
            batch_size=2
            epochs=70
        X=train_dataset[0]
        y=train_dataset[1]
        model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=["accuracy"])
        model.fit(X,y,epochs=epochs, batch_size=batch_size, validation_data=(X,y))
        if not os.path.exists(path):
            os.makedirs(path)
        model.save(file_name)
        upload_blob(bucket_name,file_name,file_name)
        return model
