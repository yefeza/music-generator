# keras imports
import keras
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Input, UpSampling2D, Layer
from tensorflow.keras.layers import Conv1D, Conv2D, Conv2DTranspose, SeparableConv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Add, Multiply, Concatenate
from tensorflow.keras.layers import LayerNormalization
from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend
import tensorflow as tf
import numpy as np
from .dataset import *
from .utils import *
from google.cloud import storage
import os
from tensorflow.keras.layers import Conv1DTranspose

EQ_DIM={
    3000: (4,750,2),
    12000: (8,1500,2),
    48000: (16,3000,2),
    192000: (32,6000,2),
    768000: (64,12000,2),
    3072000: (128,24000,2),
    12288000: (256,48000,2)
    }

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

    def get_config(self):
        config = super(SoftRectifier, self).get_config()
        config.update({"alpha": self.alpha})
        return config

class GAN(keras.Model):
    def __init__(
        self,
        discriminator,
        encoder,
        generator,
        generator_default,
        latent_dim,
        fade_in=False,
        default_network_extra=4,
        gp_weight=10.0,
    ):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.encoder = encoder
        self.generator = generator
        self.generator_default = generator_default
        self.latent_dim = latent_dim
        self.def_steps = default_network_extra
        self.gp_weight = gp_weight
        self.actual_step = 0
        self.total_steps = 0
        self.fade_in=fade_in

    def compile(self, d_optimizer, enc_optimizer, g_optimizer, df_optimizer, d_loss_fn, g_loss_fn, was_loaded=False):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.enc_optimizer = enc_optimizer
        self.g_optimizer = g_optimizer
        self.df_optimizer = df_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        if not was_loaded:
            self.discriminator.compile(optimizer=d_optimizer)
            self.encoder.compile(optimizer=enc_optimizer)

    def set_train_steps(self, size):
        self.train_steps=size

    def train_step(self, real_data):
        #control actual step
        if self.fade_in:
            models=[self.discriminator, self.generator]
            # calculate current alpha (linear from 0 to 1)
            alpha = self.actual_step / float(self.train_steps - 1)
            # update the alpha for each model
            for model in models:
                for layer in model.layers:
                    if isinstance(layer, WeightedSum):
                        backend.set_value(layer.alpha, alpha)

        if isinstance(real_data, tuple):
            real_data = real_data[0]

        # Get the batch size
        batch_size = real_data.shape[0]
        gen_shape = self.generator.output_shape

        # Run on default network
        for i in range(self.def_steps):
            mini_bsize=int(batch_size/2)
            random_encoder_input = tf.random.uniform(shape=(mini_bsize, gen_shape[-2], gen_shape[-1]), minval=-1., maxval=1.)
            random_encoded_real=self.encoder(real_data[:mini_bsize], training=False)
            random_encoded_random=self.encoder(random_encoder_input, training=False)
            random_latent_vectors=tf.concat([random_encoded_real, random_encoded_random], 0)
            random_latent_vectors = tf.random.shuffle(random_latent_vectors)
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator_default(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=False)
                # Get the logits for the real images
                real_logits = self.discriminator(real_data, training=False)
                # Calculate the discriminator loss using the fake and real image logits
                def_loss = self.g_loss_fn(fake_logits, real_logits)
            # Get the gradients w.r.t the discriminator loss
            def_gradient = tape.gradient(def_loss, self.generator_default.trainable_default_network)
            # Update the weights of the discriminator using the discriminator optimizer
            self.generator_default.optimizer.apply_gradients(
                zip(def_gradient, self.generator_default.trainable_default_network)
            )

        # Train discriminator
        mini_bsize=int(batch_size/2)
        random_encoder_input = tf.random.uniform(shape=(mini_bsize, gen_shape[-2], gen_shape[-1]), minval=-1., maxval=1.)
        random_encoded_real=self.encoder(real_data[:mini_bsize], training=False)
        random_encoded_random=self.encoder(random_encoder_input, training=False)
        random_latent_vectors=tf.concat([random_encoded_real, random_encoded_random], 0)
        random_latent_vectors = tf.random.shuffle(random_latent_vectors)
        with tf.GradientTape() as tape:
            # Generate fake images from the latent vector
            fake_images = self.generator(random_latent_vectors, training=False)
            # Get the logits for the fake images
            fake_logits = self.discriminator(fake_images, training=True)
            # Get the logits for the real images
            real_logits = self.discriminator(real_data, training=True)
            # Calculate the discriminator loss using the fake and real image logits
            d_loss = self.d_loss_fn(fake_logits, real_logits)
        # Get the gradients w.r.t the discriminator loss
        d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
        # Update the weights of the discriminator using the discriminator optimizer
        self.discriminator.optimizer.apply_gradients(
            zip(d_gradient, self.discriminator.trainable_variables)
        )
        # Train the generator
        # Get the latent vector
        #random_latent_vectors = tf.random.uniform(shape=(batch_size, self.latent_dim[0], self.latent_dim[1], self.latent_dim[2]))
        with tf.GradientTape(persistent=True) as tape:
            #get noise from encoder
            mini_bsize=int(batch_size/2)
            random_encoder_input = tf.random.uniform(shape=(mini_bsize, gen_shape[-2], gen_shape[-1]), minval=-1., maxval=1.)
            random_encoded_real=self.encoder(real_data[:mini_bsize], training=True)
            random_encoded_random=self.encoder(random_encoder_input, training=True)
            random_latent_vectors=tf.concat([random_encoded_real, random_encoded_random], 0)
            #random_latent_vectors = tf.random.shuffle(random_latent_vectors)
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=False)
            # Get the logits for the real images
            real_logits = self.discriminator(real_data, training=False)
            # Calculate the generator loss using the fake and real image logits
            g_loss = self.g_loss_fn(gen_img_logits, real_logits)
        # Get the gradients w.r.t the encoder with generator loss
        enc_gradient = tape.gradient(g_loss, self.encoder.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.encoder.optimizer.apply_gradients(
            zip(enc_gradient, self.encoder.trainable_variables)
        )
        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.generator.optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        #calculate actual delta value
        ci_1=tf.reduce_mean(real_logits)
        cu_1=tf.reduce_mean(gen_img_logits)
        dif=cu_1-ci_1
        delta_1=tf.math.abs(dif)
        self.actual_step+=1
        return {"dif": dif, "delta_1": delta_1, "cu_1": cu_1, "ci_1": ci_1, "d_loss": d_loss, "g_loss": g_loss}

class DefaultNetwork(keras.Model):
    def __init__(self):
        super(DefaultNetwork, self).__init__()
    @property
    def trainable_default_network(self):
        default_vars=[]
        for i in range(0, len(self.layers)):
            if self.layers[i].name[:5]=="defly":
                default_vars+=self.layers[i].trainable_weights
        return default_vars

# Minibatch Standard Deviation Layer

class MinibatchStdDev(Layer):
    # init with default value
    def __init__(self, **kwargs):
        super(MinibatchStdDev, self).__init__(**kwargs)

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

    def get_config(self):
        config = super(MinibatchStdDev, self).get_config()
        return config

#custom activation layer (tanh(x)+(x/(alpha+0.1)))
class SoftRectifier(Layer):
    def __init__(self, start_alpha=400.0, **kwargs):
        super(SoftRectifier, self).__init__(**kwargs)
        self.start_alpha=start_alpha
        #self.w = tf.Variable(initial_value=start_alpha, trainable=True)

    def call(self, inputs):
        return tf.math.tanh(inputs) + tf.math.divide_no_nan(inputs,self.start_alpha)

    def get_config(self):
        config = super(SoftRectifier, self).get_config()
        config.update({"start_alpha": self.start_alpha})
        return config

#custom activation layer (tanh(x)+(x/(alpha+0.1)))
class StaticOptTanh(Layer):
    def __init__(self, alpha=40000.0, **kwargs):
        super(StaticOptTanh, self).__init__(**kwargs)
        self.alpha=alpha

    def call(self, inputs):
        return tf.cast((tf.math.tanh(inputs) + tf.math.divide_no_nan(inputs,self.alpha)), dtype=tf.float32)

    def get_config(self):
        config = super(StaticOptTanh, self).get_config()
        config.update({"alpha": self.alpha})
        return config

#Decision Layer
class DecisionLayer(Layer):
    def __init__(self, output_size=9, **kwargs):
        super(DecisionLayer, self).__init__(**kwargs)
        self.output_size=output_size

    def call(self, inputs):
        # only supports two inputs: values and index distribution of valid output
        assert (len(inputs) == 2)
        output_distribution=inputs[1]
        shape_data=inputs[0].shape
        output_distribution=tf.reshape(output_distribution, shape=[-1, self.output_size, 1, 1])
        #output_distribution=tf.math.round(output_distribution)
        input_data=inputs[0]
        input_data=tf.reshape(input_data, shape=[-1, 1, shape_data[1], shape_data[2]])
        input_data=tf.tile(input_data, [1, self.output_size, 1, 1])
        return tf.math.multiply(input_data, output_distribution)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size, input_shape[1], input_shape[2])

    def get_config(self):
        config = super(DecisionLayer, self).get_config()
        config.update({"output_size": self.output_size})
        return config

#Decision Layer
class DecisionLayer2D(Layer):
    def __init__(self, output_size=9, **kwargs):
        super(DecisionLayer2D, self).__init__(**kwargs)
        self.output_size=output_size

    def call(self, inputs):
        # only supports two inputs: values and index distribution of valid output
        assert (len(inputs) == 2)
        output_distribution=inputs[1]
        shape_data=inputs[0].shape
        output_distribution=tf.reshape(output_distribution, shape=[-1, self.output_size, 1, 1, 1])
        #output_distribution=tf.math.round(output_distribution)
        input_data=inputs[0]
        input_data=tf.reshape(input_data, shape=[-1, 1, shape_data[1], shape_data[2], shape_data[3]])
        input_data=tf.tile(input_data, [1, self.output_size, 1, 1, 1])
        return tf.math.multiply(input_data, output_distribution)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size, input_shape[1], input_shape[2])

    def get_config(self):
        config = super(DecisionLayer2D, self).get_config()
        config.update({"output_size": self.output_size})
        return config

#Slicer Layer
class SlicerLayer(Layer):
    def __init__(self, index_work, **kwargs):
        super(SlicerLayer, self).__init__(**kwargs)
        self.index_work=index_work

    def call(self, inputs):
        return inputs[:,self.index_work]

    def get_config(self):
        config = super(SlicerLayer, self).get_config()
        config.update({"index_work": self.index_work})
        return config

#Complex real amd imag components as channels
class FusionLayer(Layer):
    def __init__(self, **kwargs):
        super(FusionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        abs_value=tf.math.abs(inputs)
        abs_value=tf.reduce_mean(abs_value, 3, keepdims=True)
        return tf.transpose(abs_value, perm=[0,1,3,2])

    def get_config(self):
        config = super(FusionLayer, self).get_config()
        return config

class InvertTranspose1D(Layer):
    def __init__(self, **kwargs):
        super(InvertTranspose1D, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.transpose(inputs, perm=[0,2,1])

    def get_config(self):
        config = super(InvertTranspose1D, self).get_config()
        return config

class InvertTranspose(Layer):
    def __init__(self, **kwargs):
        super(InvertTranspose, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.transpose(inputs, perm=[0,1,3,2])

    def get_config(self):
        config = super(InvertTranspose, self).get_config()
        return config

class ChannelsToComplex(Layer):
    def __init__(self, **kwargs):
        super(ChannelsToComplex, self).__init__(**kwargs)

    def call(self, inputs):
        real=inputs[:,:,:,:2]
        imag=inputs[:,:,:,2:]
        return tf.complex(real, imag)

    def get_config(self):
        config = super(ChannelsToComplex, self).get_config()
        return config

#FFT Layer
class FFT(Layer):
    def __init__(self, **kwargs):
        super(FFT, self).__init__(**kwargs)

    def call(self, inputs):
        transposed=tf.transpose(inputs, perm=[0, 1, 3, 2])
        fft = tf.signal.rfft(transposed)
        return tf.transpose(fft, perm=[0, 1, 3, 2])

    def get_config(self):
        config = super(FFT, self).get_config()
        return config

#inverse FFT Layer
class iFFT(Layer):
    def __init__(self, **kwargs):
        super(iFFT, self).__init__(**kwargs)

    def call(self, inputs):
        transposed=tf.transpose(inputs, perm=[0, 1, 3, 2])
        ifft = tf.signal.irfft(transposed)
        return tf.transpose(ifft, perm=[0, 1, 3, 2])

    def get_config(self):
        config = super(iFFT, self).get_config()
        return config

# agregar bloque a discriminador para escalar las dimensiones

def add_discriminator_block(old_model, n_input_layers=3):
    # getshapeofexistingmodel
    in_shape = list(old_model.input[0].shape)
    alpha=400.0
    soft_alpha=600.0
    if in_shape[-3]==8:
        alpha=600.0
        soft_alpha=800.0
    if in_shape[-3]==16:
        alpha=800.0
        soft_alpha=1000.0
    if in_shape[-3]==32:
        alpha=1000.0
        soft_alpha=1200.0
    if in_shape[-3]==64:
        alpha=1200.0
        soft_alpha=1400.0
    if in_shape[-3]==128:
        alpha=1400.0
        soft_alpha=1600.0
    if in_shape[-3]==256:
        alpha=1600.0
        soft_alpha=1800.0
    # definenewinputshapeasdoublethesize
    input_shape = (in_shape[-3]*2, in_shape[-2]*2, in_shape[-1])
    in_image = Input(shape=input_shape)
    featured_layer = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal')(in_image)
    #convolusion block 1
    #d_1 = FFT2d()(featured_layer)
    d_1 = Conv2D(512, (1, 2), strides=(1, 2), padding='valid', kernel_initializer='he_normal')(featured_layer)
    d_1 = Conv2D(512, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(d_1)
    #d_1 = iFFT2d()(d_1)
    d_1 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal')(d_1)
    d_1 = Dropout(0.2)(d_1)
    d_block = SoftRectifier(start_alpha=soft_alpha)(d_1)
    block_new = d_block
    # skiptheinput,1x1andactivationfortheoldmodel
    for i in range(n_input_layers, len(old_model.layers)):
        if isinstance(old_model.layers[i], StaticOptTanh):
            final_layer = StaticOptTanh()(d_block)
        else:
            d_block = old_model.layers[i](d_block)
    # model 1 without multiple inputs for composite
    model1_comp = Model(in_image, final_layer)
    # downsamplethenewlargerimage
    downsample = AveragePooling2D()(in_image)
    # connectoldinputprocessingtodownsamplednewinput
    block_old = old_model.layers[1](downsample)
    # fadeinoutputofoldmodelinputlayerwithnewinput
    d = WeightedSum()([block_old, block_new])
    # skiptheinput,1x1andactivationfortheoldmodel
    for i in range(n_input_layers, len(old_model.layers)):
        if isinstance(old_model.layers[i], StaticOptTanh):
            final_layer = StaticOptTanh()(d)
        else:
            d = old_model.layers[i](d)
    model2_comp = Model(in_image, final_layer)
    return[model1_comp, model2_comp]

# definir los discriminadores

def define_discriminator(n_blocks, input_shape=(3000, 2)):
    model_list = list()
    # base model input
    in_data = Input(shape=input_shape)
    # conv 1x1
    converted_block = Reshape((4,750,2))(in_data)
    converted_block = FFT()(converted_block)
    converted_block = FusionLayer()(converted_block)
    # convolusion block 1
    d_1 = Dense(120)(converted_block)
    d_1 = Dense(64)(d_1)
    d_1 = InvertTranspose()(d_1)
    d_1 = Conv2D(32, (1, 33), padding='valid')(d_1)
    d_1 = MinibatchStdDev()(d_1)
    d_1 = Flatten()(d_1)
    d_1 = SoftRectifier()(d_1)
    d = Dense(1)(d_1)
    out_class = StaticOptTanh()(d)
    # define model
    model_comp = Model(in_data, out_class)
    # store model
    model_list.append([model_comp, model_comp])
    # create submodels
    for i in range(1, n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_discriminator_block(old_model)
        # store model
        model_list.append(models)
    return model_list

# agregar bloque al encoder para escalar las dimensiones

def add_encoder_block(old_model, n_input_layers=3):
    # getshapeofexistingmodel
    in_shape = list(old_model.input[0].shape)
    alpha=400.0
    soft_alpha=600.0
    if in_shape[-3]==8:
        alpha=600.0
        soft_alpha=800.0
    if in_shape[-3]==16:
        alpha=800.0
        soft_alpha=1000.0
    if in_shape[-3]==32:
        alpha=1000.0
        soft_alpha=1200.0
    if in_shape[-3]==64:
        alpha=1200.0
        soft_alpha=1400.0
    if in_shape[-3]==128:
        alpha=1400.0
        soft_alpha=1600.0
    if in_shape[-3]==256:
        alpha=1600.0
        soft_alpha=1800.0
    # definenewinputshapeasdoublethesize
    input_shape = (in_shape[-3]*2, in_shape[-2]*2, in_shape[-1])
    in_image = Input(shape=input_shape)
    featured_layer = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal')(in_image)
    #convolusion block 1
    d_1 = Conv2D(512, (1, 2), strides=(1, 2), padding='valid', kernel_initializer='he_normal')(featured_layer)
    d_1 = Conv2D(512, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(d_1)
    d_1 = Dense(128)(d_1)
    d_1 = Dropout(0.2)(d_1)
    d_block = SoftRectifier(start_alpha=soft_alpha)(d_1)
    block_new = d_block
    # skiptheinput,1x1andactivationfortheoldmodel
    for i in range(n_input_layers, len(old_model.layers)):
        if isinstance(old_model.layers[i], Dense):
            final_layer = old_model.layers[i](d_block)
        else:
            d_block = old_model.layers[i](d_block)
    # model 1 without multiple inputs for composite
    model1_comp = Model(in_image, final_layer)
    # downsamplethenewlargerimage
    downsample = AveragePooling2D()(in_image)
    # connectoldinputprocessingtodownsamplednewinput
    block_old = old_model.layers[1](downsample)
    # fadeinoutputofoldmodelinputlayerwithnewinput
    d = WeightedSum()([block_old, block_new])
    # skiptheinput,1x1andactivationfortheoldmodel
    for i in range(n_input_layers, len(old_model.layers)):
        if isinstance(old_model.layers[i], Dense):
            final_layer = old_model.layers[i](d)
        else:
            d = old_model.layers[i](d)
    model2_comp = Model(in_image, final_layer)
    return[model1_comp, model2_comp]

# definir los discriminadores

def define_encoder(n_blocks, input_shape=(3000, 2)):
    model_list = list()
    # base model input
    in_data = Input(shape=input_shape)
    converted_block = Reshape((4,750,2))(in_data)
    converted_block = FFT()(converted_block)
    converted_block = FusionLayer()(converted_block)
    # convolusion block 1
    d_1 = Dense(120)(converted_block)
    d_1 = InvertTranspose()(d_1)
    d_1 = Conv2D(32, (2, 11), padding='valid')(d_1)
    d_1 = Conv2D(32, (2, 6), padding='valid')(d_1)
    d_1 = Conv2D(32, (2, 6), padding='valid')(d_1)
    d_1 = Reshape((100, 32))(d_1)
    d_1 = Dropout(0.2)(d_1)
    out_class = Dense(1)(d_1)
    # define model
    model_comp = Model(in_data, out_class)
    # store model
    model_list.append([model_comp, model_comp])
    # create submodels
    for i in range(1, n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_encoder_block(old_model)
        # store model
        model_list.append(models)
    return model_list

#controlador de nombres de capas
class LayerCounter:
    def __init__(self):
        self.actual_count=0

    def get_next(self):
        self.actual_count+=1
        return str(self.actual_count)

# agregar bloque a generador para escalar las dimensiones

def add_generator_block(old_model, counter):
    out_shape = old_model.output_shape
    mult=int((out_shape[-3]*2)/4)
    sr_out=int(out_shape[-2]*2)
    k_size=int(sr_out-(out_shape[-2]-1))
    # get the end of the last block
    block_end = old_model.layers[-2].output
    # upsample, and define new block
    upsampling = UpSampling2D()(block_end)
    #selector de incice 0
    i_sel_0=Conv2D(32, (1,6), padding='valid', name="defly_"+counter.get_next())(block_end)
    i_sel_0=Conv2D(32, (1,11), padding='valid', name="defly_"+counter.get_next())(i_sel_0)
    i_sel_0=Dropout(0.3)(i_sel_0)
    i_sel_0=Flatten()(i_sel_0)
    i_sel_0=Dense(9, activation='softmax', name="defly_"+counter.get_next())(i_sel_0)
    #decision layer 0
    des_ly_0=DecisionLayer(output_size=9)([block_end, i_sel_0])
    #rama 1 bloque 0
    b0_r1 = SlicerLayer(index_work=0)(des_ly_0)
    b0_r1 = Conv2DTranspose(8, (1, k_size), padding='valid')(b0_r1)
    b0_r1 = Conv2D(8, (1, 15), padding='same', kernel_initializer='he_normal')(b0_r1)
    b0_r1 = Conv2D(8, (1, 35), padding='same', kernel_initializer='he_normal')(b0_r1)
    b0_r1 = Conv2D(8, (1, 20), padding='same', kernel_initializer='he_normal')(b0_r1)
    #rama 2 bloque 0
    b0_r2 = SlicerLayer(index_work=1)(des_ly_0)
    b0_r2 = Conv2DTranspose(8, (1, k_size), padding='valid')(b0_r2)
    b0_r2 = Conv2D(8, (1, 25), padding='same', kernel_initializer='he_normal')(b0_r2)
    b0_r2 = Conv2D(8, (1, 12), padding='same', kernel_initializer='he_normal')(b0_r2)
    #rama 3 bloque 0
    b0_r3 = SlicerLayer(index_work=2)(des_ly_0)
    b0_r3 = Conv2DTranspose(8, (1, k_size), padding='valid')(b0_r3)
    b0_r3 = Conv2D(8, (1, 15), padding='same', kernel_initializer='he_normal')(b0_r3)
    b0_r3 = Conv2D(8, (1, 7), padding='same', kernel_initializer='he_normal')(b0_r3)
    #rama 4 bloque 0
    b0_r4 = SlicerLayer(index_work=3)(des_ly_0)
    b0_r4 = Conv2DTranspose(8, (1, k_size), padding='valid')(b0_r4)
    b0_r4 = Conv2D(8, (1, 15), padding='same', kernel_initializer='he_normal')(b0_r4)
    b0_r4 = Conv2D(8, (1, 20), padding='same', kernel_initializer='he_normal')(b0_r4)
    b0_r4 = Conv2D(8, (1, 25), padding='same', kernel_initializer='he_normal')(b0_r4)
    #rama 5 bloque 0
    b0_r5 = SlicerLayer(index_work=4)(des_ly_0)
    b0_r5 = Conv2DTranspose(8, (1, k_size), padding='valid')(b0_r5)
    b0_r5 = Conv2D(8, (1, 16), padding='same', kernel_initializer='he_normal')(b0_r5)
    b0_r5 = Conv2D(8, (1, 26), padding='same', kernel_initializer='he_normal')(b0_r5)
    #rama 6 bloque 0
    b0_r6 = SlicerLayer(index_work=5)(des_ly_0)
    b0_r6 = Conv2DTranspose(8, (1, k_size), padding='valid')(b0_r6)
    b0_r6 = Conv2D(8, (1, 15), padding='same', kernel_initializer='he_normal')(b0_r6)
    b0_r6 = Conv2D(8, (1, 25), padding='same', kernel_initializer='he_normal')(b0_r6)
    #rama 7 bloque 0
    b0_r7 = SlicerLayer(index_work=6)(des_ly_0)
    b0_r7 = Conv2DTranspose(8, (1, k_size), padding='valid')(b0_r7)
    b0_r7 = Conv2D(8, (1, 15), padding='same', kernel_initializer='he_normal')(b0_r7)
    b0_r7 = Conv2D(8, (1, 35), padding='same', kernel_initializer='he_normal')(b0_r7)
    b0_r7 = Conv2D(8, (1, 11), padding='same', kernel_initializer='he_normal')(b0_r7)
    #rama 8 bloque 0
    b0_r8 = SlicerLayer(index_work=7)(des_ly_0)
    b0_r8 = Conv2DTranspose(8, (1, k_size), padding='valid')(b0_r8)
    b0_r8 = Conv2D(8, (1, 35), padding='same', kernel_initializer='he_normal')(b0_r8)
    b0_r8 = Conv2D(8, (1, 15), padding='same', kernel_initializer='he_normal')(b0_r8)
    #rama 9 bloque 0
    b0_r9 = SlicerLayer(index_work=8)(des_ly_0)
    b0_r9 = Conv2DTranspose(8, (1, k_size), padding='valid')(b0_r9)
    b0_r9 = Conv2D(8, (1, 11), padding='same', kernel_initializer='he_normal')(b0_r9)
    b0_r9 = Conv2D(8, (1, 25), padding='same', kernel_initializer='he_normal')(b0_r9)
    #sumar ramas bloque 0
    merger_b0=Add()([b0_r1, b0_r2, b0_r3, b0_r4, b0_r5, b0_r6, b0_r7, b0_r8, b0_r9])
    #index selector block 1
    i_sel_1=Conv2D(32, (1,15), padding='valid', name="defly_"+counter.get_next())(merger_b0)
    i_sel_1=Conv2D(32, (1,51), padding='valid', name="defly_"+counter.get_next())(i_sel_1)
    i_sel_1=Dropout(0.3)(i_sel_1)
    i_sel_1=Flatten()(i_sel_1)
    i_sel_1=Dense(6, activation='softmax', name="defly_"+counter.get_next())(i_sel_1)
    #decision layer
    des_ly_1=DecisionLayer(output_size=6)([merger_b0, i_sel_1])
    #rama 1
    b1_r1 = SlicerLayer(index_work=0)(des_ly_1)
    b1_r1 = UpSampling2D()(b1_r1)
    b1_r1 = UpSampling2D()(b1_r1)
    b1_r1 = Conv2D(16, (2,4), strides=(2,4), padding='valid', kernel_initializer='he_normal')(b1_r1)
    b1_r1 = Conv2D(16, (4, 15), padding='same', kernel_initializer='he_normal')(b1_r1)
    b1_r1 = Conv2D(16, (4, 20), padding='same', kernel_initializer='he_normal')(b1_r1)
    #rama 2
    b1_r2 = SlicerLayer(index_work=1)(des_ly_1)
    b1_r2 = UpSampling2D()(b1_r2)
    b1_r2 = UpSampling2D()(b1_r2)
    b1_r2 = Conv2D(16, (2,4), strides=(2,4), padding='valid', kernel_initializer='he_normal')(b1_r2)
    b1_r2 = Conv2D(16, (4, 15), padding='same', kernel_initializer='he_normal')(b1_r2)
    b1_r2 = Conv2D(16, (4, 35), padding='same', kernel_initializer='he_normal')(b1_r2)
    #rama 3
    b1_r3 = SlicerLayer(index_work=2)(des_ly_1)
    b1_r3 = UpSampling2D()(b1_r3)
    b1_r3 = UpSampling2D()(b1_r3)
    b1_r3 = Conv2D(16, (2,4), strides=(2,4), padding='valid', kernel_initializer='he_normal')(b1_r3)
    b1_r3 = Conv2D(16, (4, 20), padding='same', kernel_initializer='he_normal')(b1_r3)
    b1_r3 = Conv2D(16, (4, 15), padding='same', kernel_initializer='he_normal')(b1_r3)
    #rama 4
    b1_r4 = SlicerLayer(index_work=3)(des_ly_1)
    b1_r4 = UpSampling2D()(b1_r4)
    b1_r4 = UpSampling2D()(b1_r4)
    b1_r4 = Conv2D(16, (2,4), strides=(2,4), padding='valid', kernel_initializer='he_normal')(b1_r4)
    b1_r4 = Conv2D(16, (4, 15), padding='same', kernel_initializer='he_normal')(b1_r4)
    b1_r4 = Conv2D(16, (4, 20), padding='same', kernel_initializer='he_normal')(b1_r4)
    #rama 5
    b1_r5 = SlicerLayer(index_work=4)(des_ly_1)
    b1_r5 = UpSampling2D()(b1_r5)
    b1_r5 = UpSampling2D()(b1_r5)
    b1_r5 = Conv2D(16, (2,4), strides=(2,4), padding='valid', kernel_initializer='he_normal')(b1_r5)
    b1_r5 = Conv2D(16, (4, 15), padding='same', kernel_initializer='he_normal')(b1_r5)
    b1_r5 = Conv2D(16, (4, 35), padding='same', kernel_initializer='he_normal')(b1_r5)
    #rama 6
    b1_r6 = SlicerLayer(index_work=5)(des_ly_1)
    b1_r6 = UpSampling2D()(b1_r6)
    b1_r6 = UpSampling2D()(b1_r6)
    b1_r6 = Conv2D(16, (2,4), strides=(2,4), padding='valid', kernel_initializer='he_normal')(b1_r6)
    b1_r6 = Conv2D(16, (4, 20), padding='same', kernel_initializer='he_normal')(b1_r6)
    b1_r6 = Conv2D(16, (4, 15), padding='same', kernel_initializer='he_normal')(b1_r6)
    #sumar ramas
    merger_b1=Add()([b1_r1, b1_r2, b1_r3, b1_r4, b1_r5, b1_r6])
    #index selector block 3
    i_sel_2=Conv2D(8, (1,26), padding='valid', name="defly_"+counter.get_next())(merger_b1)
    i_sel_2=Conv2D(16, (1,26), padding='valid', name="defly_"+counter.get_next())(i_sel_2)
    i_sel_2=Conv2D(32, (1,26), padding='valid', name="defly_"+counter.get_next())(i_sel_2)
    i_sel_2=Dropout(0.3)(i_sel_2)
    i_sel_2=Flatten()(i_sel_2)
    i_sel_2=Dense(32, activation='softmax', name="defly_"+counter.get_next())(i_sel_2)
    #decision layer block 3
    des_ly_3=DecisionLayer(output_size=32)([merger_b1, i_sel_2])
    outputs_list=[]
    #Convolutional outputs
    for i in range(16):
        new_output_block = SlicerLayer(index_work=i)(des_ly_3)
        new_output_block = Conv2D(2, (1,1), padding='valid')(new_output_block)
        outputs_list.append(new_output_block)
    #Dense outputs
    for i in range(16):
        new_output_block = SlicerLayer(index_work=(16+i))(des_ly_3)
        new_output_block = Dense(2)(new_output_block)
        outputs_list.append(new_output_block)
    merger_b2=Add()(outputs_list)
    out_image = LayerNormalization(axis=[1,2])(merger_b2)
    # define model
    model1_normal = Model(old_model.input, out_image)
    model1_normal.compile(optimizer=Adamax(learning_rate=0.0005))
    #define default
    model1_default = Model(old_model.input, out_image)
    for i in range(0, len(model1_default.layers)):
        if model1_default.layers[i].name[:5]!="defly":
            model1_default.layers[i].trainable=False
    model1_default.compile(optimizer=Adamax(learning_rate=0.0005))
    #return to trainable dflayers for fadein
    model2_normal = Model(old_model.input, out_image)
    for i in range(0, len(model2_normal.layers)):
        if model2_normal.layers[i].name[:5]!="defly":
            model2_normal.layers[i].trainable=True
    # define new output image as the weighted sum of the old and new models
    merged = WeightedSum()([upsampling, merger_b2])
    output_2 = LayerNormalization(axis=[1,2])(merged)
    # define model
    model2_normal = Model(old_model.input, output_2)
    model2_normal.compile(optimizer=Adamax(learning_rate=0.0005))
    #define default
    model2_default = Model(old_model.input, output_2)
    for i in range(0, len(model2_default.layers)):
        if model2_default.layers[i].name[:5]!="defly":
            model2_default.layers[i].trainable=False
    model2_default.compile(optimizer=Adamax(learning_rate=0.0005))
    return [model1_normal, model1_default, model2_normal, model2_default]

# definir los generadores

def define_generator(n_blocks, latent_dim):
    counter=LayerCounter()
    model_list = list()
    # input
    ly0 = Input(shape=latent_dim)
    #selector de incice 0
    i_sel_0=InvertTranspose1D()(ly0)
    i_sel_0=Dense(32, name="defly_"+counter.get_next())(i_sel_0)
    i_sel_0=InvertTranspose1D()(i_sel_0)
    i_sel_0=Conv1D(64, 17, padding='valid', name="defly_"+counter.get_next())(i_sel_0)
    i_sel_0=Dropout(0.2)(i_sel_0)
    i_sel_0=Flatten()(i_sel_0)
    i_sel_0=Dense(6, activation='softmax', name="defly_"+counter.get_next())(i_sel_0)
    #decision layer 0
    des_ly_0=DecisionLayer(output_size=6)([ly0, i_sel_0])
    #bloque 0 salidas de (376,16)
    #rama 1 bloque 0
    b0_r1 = SlicerLayer(index_work=0)(des_ly_0)
    b0_r1 = Conv1D(16, 2, strides=2, padding='valid')(b0_r1)
    b0_r1 = Conv1DTranspose(16, 5, strides=5, padding='valid')(b0_r1)
    b0_r1 = Conv1DTranspose(16, 127, padding='valid')(b0_r1)
    #rama 2 bloque 0
    b0_r2 = SlicerLayer(index_work=1)(des_ly_0)
    b0_r2 = Conv1D(16, 2, strides=2, padding='valid')(b0_r2)
    b0_r2 = Conv1DTranspose(8, 3, strides=2, padding='valid')(b0_r2)
    b0_r2 = Conv1DTranspose(16, 5, strides=2, padding='valid')(b0_r2)
    b0_r2 = Conv1DTranspose(16, 46, padding='valid')(b0_r2)
    b0_r2 = Conv1DTranspose(16, 51, padding='valid')(b0_r2)
    b0_r2 = Conv1DTranspose(16, 77, padding='valid')(b0_r2)
    #rama 3 bloque 0
    b0_r3 = SlicerLayer(index_work=2)(des_ly_0)
    b0_r3 = Conv1D(16, 51, padding='valid')(b0_r3)
    b0_r3 = Conv1DTranspose(8, 23, strides=3, padding='valid')(b0_r3)
    b0_r3 = Conv1DTranspose(8, 81, padding='valid')(b0_r3)
    b0_r3 = Conv1DTranspose(16, 26, padding='valid')(b0_r3)
    b0_r3 = Conv1DTranspose(16, 26, padding='valid')(b0_r3)
    b0_r3 = Conv1DTranspose(16, 26, padding='valid')(b0_r3)
    b0_r3 = Conv1DTranspose(16, 26, padding='valid')(b0_r3)
    b0_r3 = Conv1DTranspose(16, 27, padding='valid')(b0_r3)
    #rama 4 bloque 0
    b0_r4 = SlicerLayer(index_work=3)(des_ly_0)
    b0_r4 = Conv1D(16, 51, padding='valid')(b0_r4)
    b0_r4 = Conv1DTranspose(8, 51, padding='valid')(b0_r4)
    b0_r4 = Conv1DTranspose(8, 101, padding='valid')(b0_r4)
    b0_r4 = Conv1DTranspose(16, 51, padding='valid')(b0_r4)
    b0_r4 = Conv1DTranspose(8, 51, padding='valid')(b0_r4)
    b0_r4 = Conv1DTranspose(16, 51, padding='valid')(b0_r4)
    b0_r4 = Conv1DTranspose(16, 27, padding='valid')(b0_r4)
    #rama 5 bloque 0
    b0_r5 = SlicerLayer(index_work=4)(des_ly_0)
    b0_r5 = Conv1D(16, 51, padding='valid')(b0_r5)
    b0_r5 = Conv1DTranspose(8, 5, strides=5, padding='valid')(b0_r5)
    b0_r5 = Conv1DTranspose(8, 26, padding='valid')(b0_r5)
    b0_r5 = Conv1DTranspose(16, 26, padding='valid')(b0_r5)
    b0_r5 = Conv1DTranspose(8, 26, padding='valid')(b0_r5)
    b0_r5 = Conv1DTranspose(8, 26, padding='valid')(b0_r5)
    b0_r5 = Conv1DTranspose(16, 27, padding='valid')(b0_r5)
    #rama 6 bloque 0
    b0_r6 = SlicerLayer(index_work=5)(des_ly_0)
    b0_r6 = Conv1D(16, 51, padding='valid')(b0_r6)
    b0_r6 = Conv1DTranspose(16, 3, strides=2, padding='valid')(b0_r6)
    b0_r6 = Conv1DTranspose(8, 5, strides=2, padding='valid')(b0_r6)
    b0_r6 = Conv1DTranspose(16, 46, padding='valid')(b0_r6)
    b0_r6 = Conv1DTranspose(8, 51, padding='valid')(b0_r6)
    b0_r6 = Conv1DTranspose(16, 51, padding='valid')(b0_r6)
    b0_r6 = Conv1DTranspose(16, 27, padding='valid')(b0_r6)
    #sumar ramas bloque 0
    to_connect_0=Add()([b0_r1, b0_r2, b0_r3, b0_r4, b0_r5, b0_r6])
    #selector de incice 1
    i_sel_1=InvertTranspose1D()(to_connect_0)
    i_sel_1=Dense(120, name="defly_"+counter.get_next())(i_sel_1)
    i_sel_1=InvertTranspose1D()(i_sel_1)
    i_sel_1=Conv1D(8, 31, padding='valid', name="defly_"+counter.get_next())(i_sel_1)
    i_sel_1=Dropout(0.3)(i_sel_1)
    i_sel_1=Flatten()(i_sel_1)
    i_sel_1=Dense(6, activation='softmax', name="defly_"+counter.get_next())(i_sel_1)
    #decision layer
    des_ly_1=DecisionLayer(output_size=6)([to_connect_0, i_sel_1])
    #bloque 1 salida (376, 8)
    #rama 1
    b1_r1 = SlicerLayer(index_work=0)(des_ly_1)
    b1_r1 = InvertTranspose1D()(b1_r1)
    b1_r1 = Dense(120)(b1_r1)
    b1_r1 = InvertTranspose1D()(b1_r1)
    b1_r1 = Conv1DTranspose(16, 121, padding='valid')(b1_r1)
    b1_r1 = Conv1DTranspose(16, 137, padding='valid')(b1_r1)
    #rama 2
    b1_r2 = SlicerLayer(index_work=1)(des_ly_1)
    b1_r2 = InvertTranspose1D()(b1_r2)
    b1_r2 = Dense(60)(b1_r2)
    b1_r2 = InvertTranspose1D()(b1_r2)
    b1_r2 = Conv1DTranspose(16, 2, strides=2, padding='valid')(b1_r2)
    b1_r2 = Conv1DTranspose(16, 3, strides=3, padding='valid')(b1_r2)
    b1_r2 = Conv1DTranspose(16, 17, padding='valid')(b1_r2)
    #rama 3
    b1_r3 = SlicerLayer(index_work=2)(des_ly_1)
    b1_r3 = InvertTranspose1D()(b1_r3)
    b1_r3 = Dense(30)(b1_r3)
    b1_r3 = InvertTranspose1D()(b1_r3)
    b1_r3 = Conv1DTranspose(16, 2, strides=2, padding='valid')(b1_r3)
    b1_r3 = Conv1DTranspose(16, 2, strides=2, padding='valid')(b1_r3)
    b1_r3 = Conv1DTranspose(16, 3, strides=3, padding='valid')(b1_r3)
    b1_r3 = Conv1DTranspose(16, 17, padding='valid')(b1_r3)
    #rama 4
    b1_r4 = SlicerLayer(index_work=3)(des_ly_1)
    b1_r4 = InvertTranspose1D()(b1_r4)
    b1_r4 = Dense(170)(b1_r4)
    b1_r4 = InvertTranspose1D()(b1_r4)
    b1_r4 = Conv1DTranspose(16, 2, strides=2, padding='valid')(b1_r4)
    b1_r4 = Conv1DTranspose(16, 37, padding='valid')(b1_r4)
    #rama 5
    b1_r5 = SlicerLayer(index_work=4)(des_ly_1)
    b1_r5 = InvertTranspose1D()(b1_r5)
    b1_r5 = Dense(160)(b1_r5)
    b1_r5 = InvertTranspose1D()(b1_r5)
    b1_r5 = Conv1DTranspose(16, 2, strides=2, padding='valid')(b1_r5)
    b1_r5 = Conv1DTranspose(16, 26, padding='valid')(b1_r5)
    b1_r5 = Conv1DTranspose(16, 32, padding='valid')(b1_r5)
    #rama 6
    b1_r6 = SlicerLayer(index_work=5)(des_ly_1)
    b1_r6 = InvertTranspose1D()(b1_r6)
    b1_r6 = Dense(50)(b1_r6)
    b1_r6 = InvertTranspose1D()(b1_r6)
    b1_r6 = Conv1DTranspose(16, 7, strides=7, padding='valid')(b1_r6)
    b1_r6 = Conv1DTranspose(16, 27, padding='valid')(b1_r6)
    #sumar ramas
    to_connect_1=Add()([b1_r1, b1_r2, b1_r3, b1_r4, b1_r5, b1_r6])
    #bloque 2
    #selector de incice 1
    i_sel_2=InvertTranspose1D()(to_connect_1)
    i_sel_2=Dense(120, name="defly_"+counter.get_next())(i_sel_2)
    i_sel_2=InvertTranspose1D()(i_sel_2)
    i_sel_2=Conv1D(8, 31, padding='valid', name="defly_"+counter.get_next())(i_sel_2)
    i_sel_2=Dropout(0.3)(i_sel_2)
    i_sel_2=Flatten()(i_sel_2)
    i_sel_2=Dense(6, activation='softmax', name="defly_"+counter.get_next())(i_sel_2)
    #decision layer
    des_ly_2=DecisionLayer(output_size=6)([to_connect_1, i_sel_2])
    #bloque 1 salida (376, 8)
    #rama 1
    b2_r1 = SlicerLayer(index_work=0)(des_ly_2)
    b2_r1 = InvertTranspose1D()(b2_r1)
    b2_r1 = Dense(120)(b2_r1)
    b2_r1 = InvertTranspose1D()(b2_r1)
    b2_r1 = Conv1DTranspose(16, 121, padding='valid')(b2_r1)
    b2_r1 = Conv1DTranspose(16, 137, padding='valid')(b2_r1)
    #rama 2
    b2_r2 = SlicerLayer(index_work=1)(des_ly_2)
    b2_r2 = InvertTranspose1D()(b2_r2)
    b2_r2 = Dense(60)(b2_r2)
    b2_r2 = InvertTranspose1D()(b2_r2)
    b2_r2 = Conv1DTranspose(16, 2, strides=2, padding='valid')(b2_r2)
    b2_r2 = Conv1DTranspose(16, 3, strides=3, padding='valid')(b2_r2)
    b2_r2 = Conv1DTranspose(16, 17, padding='valid')(b2_r2)
    #rama 3
    b2_r3 = SlicerLayer(index_work=2)(des_ly_2)
    b2_r3 = InvertTranspose1D()(b2_r3)
    b2_r3 = Dense(30)(b2_r3)
    b2_r3 = InvertTranspose1D()(b2_r3)
    b2_r3 = Conv1DTranspose(16, 2, strides=2, padding='valid')(b2_r3)
    b2_r3 = Conv1DTranspose(16, 2, strides=2, padding='valid')(b2_r3)
    b2_r3 = Conv1DTranspose(16, 3, strides=3, padding='valid')(b2_r3)
    b2_r3 = Conv1DTranspose(16, 17, padding='valid')(b2_r3)
    #rama 4
    b2_r4 = SlicerLayer(index_work=3)(des_ly_2)
    b2_r4 = InvertTranspose1D()(b2_r4)
    b2_r4 = Dense(170)(b2_r4)
    b2_r4 = InvertTranspose1D()(b2_r4)
    b2_r4 = Conv1DTranspose(16, 2, strides=2, padding='valid')(b2_r4)
    b2_r4 = Conv1DTranspose(16, 37, padding='valid')(b2_r4)
    #rama 5
    b2_r5 = SlicerLayer(index_work=4)(des_ly_2)
    b2_r5 = InvertTranspose1D()(b2_r5)
    b2_r5 = Dense(160)(b2_r5)
    b2_r5 = InvertTranspose1D()(b2_r5)
    b2_r5 = Conv1DTranspose(16, 2, strides=2, padding='valid')(b2_r5)
    b2_r5 = Conv1DTranspose(16, 26, padding='valid')(b2_r5)
    b2_r5 = Conv1DTranspose(16, 32, padding='valid')(b2_r5)
    #rama 6
    b2_r6 = SlicerLayer(index_work=5)(des_ly_2)
    b2_r6 = InvertTranspose1D()(b2_r6)
    b2_r6 = Dense(50)(b2_r6)
    b2_r6 = InvertTranspose1D()(b2_r6)
    b2_r6 = Conv1DTranspose(16, 7, strides=7, padding='valid')(b2_r6)
    b2_r6 = Conv1DTranspose(16, 27, padding='valid')(b2_r6)
    #sumar ramas
    to_connect_2=Add()([b2_r1, b2_r2, b2_r3, b2_r4, b2_r5, b2_r6])
    #bloque 3
    #selector de incice 1
    i_sel_3=InvertTranspose1D()(to_connect_2)
    i_sel_3=Dense(120, name="defly_"+counter.get_next())(i_sel_3)
    i_sel_3=InvertTranspose1D()(i_sel_3)
    i_sel_3=Conv1D(8, 31, padding='valid', name="defly_"+counter.get_next())(i_sel_3)
    i_sel_3=Dropout(0.3)(i_sel_3)
    i_sel_3=Flatten()(i_sel_3)
    i_sel_3=Dense(6, activation='softmax', name="defly_"+counter.get_next())(i_sel_3)
    #decision layer
    des_ly_3=DecisionLayer(output_size=6)([to_connect_2, i_sel_3])
    #bloque 1 salida (376, 8)
    #rama 1
    b3_r1 = SlicerLayer(index_work=0)(des_ly_3)
    b3_r1 = InvertTranspose1D()(b3_r1)
    b3_r1 = Dense(120)(b3_r1)
    b3_r1 = InvertTranspose1D()(b3_r1)
    b3_r1 = Conv1DTranspose(16, 121, padding='valid')(b3_r1)
    b3_r1 = Conv1DTranspose(16, 137, padding='valid')(b3_r1)
    #rama 2
    b3_r2 = SlicerLayer(index_work=1)(des_ly_3)
    b3_r2 = InvertTranspose1D()(b3_r2)
    b3_r2 = Dense(60)(b3_r2)
    b3_r2 = InvertTranspose1D()(b3_r2)
    b3_r2 = Conv1DTranspose(16, 2, strides=2, padding='valid')(b3_r2)
    b3_r2 = Conv1DTranspose(16, 3, strides=3, padding='valid')(b3_r2)
    b3_r2 = Conv1DTranspose(16, 17, padding='valid')(b3_r2)
    #rama 3
    b3_r3 = SlicerLayer(index_work=2)(des_ly_3)
    b3_r3 = InvertTranspose1D()(b3_r3)
    b3_r3 = Dense(30)(b3_r3)
    b3_r3 = InvertTranspose1D()(b3_r3)
    b3_r3 = Conv1DTranspose(16, 2, strides=2, padding='valid')(b3_r3)
    b3_r3 = Conv1DTranspose(16, 2, strides=2, padding='valid')(b3_r3)
    b3_r3 = Conv1DTranspose(16, 3, strides=3, padding='valid')(b3_r3)
    b3_r3 = Conv1DTranspose(16, 17, padding='valid')(b3_r3)
    #rama 4
    b3_r4 = SlicerLayer(index_work=3)(des_ly_3)
    b3_r4 = InvertTranspose1D()(b3_r4)
    b3_r4 = Dense(170)(b3_r4)
    b3_r4 = InvertTranspose1D()(b3_r4)
    b3_r4 = Conv1DTranspose(16, 2, strides=2, padding='valid')(b3_r4)
    b3_r4 = Conv1DTranspose(16, 37, padding='valid')(b3_r4)
    #rama 5
    b3_r5 = SlicerLayer(index_work=4)(des_ly_3)
    b3_r5 = InvertTranspose1D()(b3_r5)
    b3_r5 = Dense(160)(b3_r5)
    b3_r5 = InvertTranspose1D()(b3_r5)
    b3_r5 = Conv1DTranspose(16, 2, strides=2, padding='valid')(b3_r5)
    b3_r5 = Conv1DTranspose(16, 26, padding='valid')(b3_r5)
    b3_r5 = Conv1DTranspose(16, 32, padding='valid')(b3_r5)
    #rama 6
    b3_r6 = SlicerLayer(index_work=5)(des_ly_3)
    b3_r6 = InvertTranspose1D()(b3_r6)
    b3_r6 = Dense(50)(b3_r6)
    b3_r6 = InvertTranspose1D()(b3_r6)
    b3_r6 = Conv1DTranspose(16, 7, strides=7, padding='valid')(b3_r6)
    b3_r6 = Conv1DTranspose(16, 27, padding='valid')(b3_r6)
    #sumar ramas
    to_connect_3=Add()([b3_r1, b3_r2, b3_r3, b3_r4, b3_r5, b3_r6])
    #concatenar bloques
    merger_b2=Concatenate(axis=1)([to_connect_0, to_connect_1, to_connect_2, to_connect_3])
    merger_b2=Reshape((4, 376, 16))(merger_b2)
    #selector de incice 2
    i_sel_3=InvertTranspose()(merger_b2)
    i_sel_3=Dense(120, name="defly_"+counter.get_next())(i_sel_3)
    i_sel_3=InvertTranspose()(i_sel_3)
    i_sel_3=Conv2D(32, (2, 16), padding='valid', name="defly_"+counter.get_next())(i_sel_3)
    i_sel_3=Conv2D(16, (2, 36), padding='valid', name="defly_"+counter.get_next())(i_sel_3)
    i_sel_3=Conv2D(8, (2, 51), padding='valid', name="defly_"+counter.get_next())(i_sel_3)
    i_sel_3=Dropout(0.3)(i_sel_3)
    i_sel_3=Flatten()(i_sel_3)
    i_sel_3=Dense(128, activation='softmax', name="defly_"+counter.get_next())(i_sel_3)
    #decision layer block 3
    des_ly_3=DecisionLayer2D(output_size=128)([merger_b2, i_sel_3])
    outputs_list=[]
    #Convolutional outputs
    for i in range(64):
        new_output_block = SlicerLayer(index_work=i)(des_ly_3)
        new_output_block = Conv2D(4, (1,1), padding='valid')(new_output_block)
        outputs_list.append(new_output_block)
    #Dense outputs
    for i in range(64):
        new_output_block = SlicerLayer(index_work=(64+i))(des_ly_3)
        new_output_block = Dense(4)(new_output_block)
        outputs_list.append(new_output_block)
    merger_b3=Add()(outputs_list)
    merger_b3=ChannelsToComplex()(merger_b3)
    merger_b3=iFFT()(merger_b3)
    merger_b3=Reshape((3000, 2))(merger_b3)
    wls = LayerNormalization(axis=[1,2])(merger_b3)
    model_normal = Model(ly0, wls)
    model_normal.compile(optimizer=Adamax(learning_rate=0.0005))
    #define default
    model_default = DefaultNetwork(ly0, wls)
    '''for i in range(0, len(model_default.layers)):
        if model_default.layers[i].name[:5]!="defly":
            model_default.layers[i].trainable=True'''
    model_default.compile(optimizer=Adamax(learning_rate=0.0005))
    # store model
    model_list.append([model_normal, model_default, model_normal, model_default])
    # create submodels 
    for i in range(1, n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_generator_block(old_model, counter)
        # store model
        model_list.append(models)
    return model_list

#losses definition

def discriminator_loss(fake_logits, real_logits):
    #fake_logits=tf.reduce_mean(fake_logits)
    #real_logits=tf.reduce_mean(real_logits)
    m=(fake_logits*real_logits)/9
    return 2*tf.math.sin(m)

def generator_loss(fake_logits, real_logits):
    #fake_logits=tf.reduce_mean(fake_logits)
    #real_logits=tf.reduce_mean(real_logits)
    m=(fake_logits*real_logits)/10
    return 3*tf.math.sin(-m)

def get_saved_model(dimension=(4,750,2), bucket_name="music-gen", epoch_checkpoint=20):
    storage_client = storage.Client(project='ia-devs')
    bucket = storage_client.bucket(bucket_name)
    #crear carpeta local si no existe
    path = "restoremodels/" + str(dimension[0]) + "-" + str(dimension[1])
    if not os.path.exists(path):
        os.makedirs(path)
    #cargar discriminador
    gcloud_file_name = "ckeckpoints/" + str(dimension[0]) + "-" + str(dimension[1]) + "/epoch" + str(epoch_checkpoint) + "/d_model.h5"
    local_file_name = "restoremodels/" + str(dimension[0]) + "-" + str(dimension[1]) + "/d_model.h5"
    blob = bucket.blob(gcloud_file_name)
    blob.download_to_filename(local_file_name)
    print("Loading discriminator")
    d_model=keras.models.load_model(local_file_name, custom_objects={"SoftRectifier":SoftRectifier, "StaticOptTanh": StaticOptTanh, "MinibatchStdDev":MinibatchStdDev, "WeightedSum":WeightedSum,   "FFT":FFT, "iFFT":iFFT})
    #cargar generador
    gcloud_file_name = "ckeckpoints/" + str(dimension[0]) + "-" + str(dimension[1]) + "/epoch" + str(epoch_checkpoint) + "/g_model.h5"
    local_file_name = "restoremodels/" + str(dimension[0]) + "-" + str(dimension[1]) + "/g_model.h5"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcloud_file_name)
    blob.download_to_filename(local_file_name)
    print("Loading generator")
    g_model=keras.models.load_model(local_file_name, custom_objects={"SoftRectifier":SoftRectifier, "StaticOptTanh": StaticOptTanh, "MinibatchStdDev":MinibatchStdDev, "WeightedSum":WeightedSum,  'DecisionLayer': DecisionLayer, 'SlicerLayer': SlicerLayer,  "FFT":FFT, "iFFT":iFFT})
    #cargar generador default
    gcloud_file_name = "ckeckpoints/" + str(dimension[0]) + "-" + str(dimension[1]) + "/epoch" + str(epoch_checkpoint) + "/df_model.h5"
    local_file_name = "restoremodels/" + str(dimension[0]) + "-" + str(dimension[1]) + "/df_model.h5"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcloud_file_name)
    blob.download_to_filename(local_file_name)
    print("Loading generator default")
    df_model=keras.models.load_model(local_file_name, custom_objects={"SoftRectifier":SoftRectifier, "StaticOptTanh": StaticOptTanh, "MinibatchStdDev":MinibatchStdDev, "WeightedSum":WeightedSum,  'DecisionLayer': DecisionLayer, 'SlicerLayer': SlicerLayer,  "FFT":FFT, "iFFT":iFFT})
    #cargar generador
    gcloud_file_name = "ckeckpoints/" + str(dimension[0]) + "-" + str(dimension[1]) + "/epoch" + str(epoch_checkpoint) + "/e_model.h5"
    local_file_name = "restoremodels/" + str(dimension[0]) + "-" + str(dimension[1]) + "/e_model.h5"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcloud_file_name)
    blob.download_to_filename(local_file_name)
    print("Loading encoder")
    e_model=keras.models.load_model(local_file_name, custom_objects={"SoftRectifier":SoftRectifier, "StaticOptTanh": StaticOptTanh, "MinibatchStdDev":MinibatchStdDev, "WeightedSum":WeightedSum,  'DecisionLayer': DecisionLayer, 'SlicerLayer': SlicerLayer,  "FFT":FFT, "iFFT":iFFT})
    return g_model, df_model, d_model, e_model

# define composite models for training generators via discriminators

def define_composite(discriminators, generators, encoders, latent_dim):
    resume_models=[False, False, False, False, False, False, False]
    dimensions=[(4,750,2),(8,1500,2),(16,3000,2),(32,6000,2),(64,12000,2),(128,24000,2),(256,48000,2)]
    model_list = list()
    # create composite models
    for i in range(len(discriminators)):
        g_models, d_models, enc_models = generators[i], discriminators[i], encoders[i]
        #precargar pesos previos de un checkpoint
        if resume_models[i]:
            prev_g_model, prev_df_model, prev_d_model, prev_e_model=get_saved_model(dimension=dimensions[i])
            d_models[0].set_weights(prev_d_model.get_weights())
            g_models[0].set_weights(prev_g_model.get_weights())
            enc_models[0].set_weights(prev_e_model.get_weights())
            d_models[0].compile(optimizer=prev_d_model.optimizer)
            g_models[0].optimizer._create_all_weights(g_models[0].trainable_variables)
            g_models[0].optimizer.set_weights(prev_g_model.optimizer.get_weights())
            g_models[1].optimizer._create_all_weights(g_models[1].trainable_variables)
            g_models[1].optimizer.set_weights(prev_df_model.optimizer.get_weights())
            enc_models[0].compile(optimizer=prev_e_model.optimizer)
        # straight-through model
        #d_models[2].trainable = False
        #enc_models[0].summary()
        #d_models[0].summary()
        #g_models[0].summary()
        #g_models[1].summary()
        #print("Real default trainable:", len(g_models[1].trainable_default_network))
        wgan1 = GAN(
            discriminator=d_models[0],
            encoder=enc_models[0],
            generator=g_models[0],
            generator_default=g_models[1],
            latent_dim=latent_dim,
            default_network_extra=4,
        )
        wgan1.compile(
            d_optimizer=Adamax(learning_rate=0.0005),
            enc_optimizer=Adamax(learning_rate=0.0005),
            g_optimizer=Adamax(learning_rate=0.0005),
            df_optimizer=Adamax(learning_rate=0.0005),
            g_loss_fn=generator_loss,
            d_loss_fn=discriminator_loss,
            was_loaded=resume_models[i]
        )
        # fade-in model
        #d_models[3].trainable = False
        wgan2 = GAN(
            discriminator=d_models[1],
            encoder=enc_models[1],
            generator=g_models[2],
            generator_default=g_models[3],
            latent_dim=latent_dim,
            fade_in=True,
            default_network_extra=4,
        )
        wgan2.compile(
            d_optimizer=Adamax(learning_rate=0.0005),
            enc_optimizer=Adamax(learning_rate=0.0005),
            g_optimizer=Adamax(learning_rate=0.0005),
            df_optimizer=Adamax(learning_rate=0.0005),
            g_loss_fn=generator_loss,
            d_loss_fn=discriminator_loss
        )
        # store
        model_list.append([wgan1, wgan2])
    return model_list

#checkpoint

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, job_dir, evaluador, num_examples=20, latent_dim=(1, 10, 2)):
        self.num_examples = num_examples
        self.latent_dim = latent_dim
        self.bucket_name = "music-gen"
        self.job_dir = job_dir
        self.evaluador = evaluador
    
    def on_epoch_end(self, epoch, logs=None):
        iters_gen=self.num_examples
        pred=[]
        gen_shape = self.model.generator.output_shape
        gen_shape = EQ_DIM[gen_shape[-2]]
        if not self.model.fade_in:
            for i in range(iters_gen):
                if ((epoch+1)%10)==0:
                    save=True
                else:
                    save=False
                if i>=15:
                    random_real_data=get_random_real_data((gen_shape[-3], gen_shape[-2]))
                else:
                    random_real_data=[]
                pred_batch=generar_ejemplo(self.model.generator, self.model.encoder, gen_shape, random_real_data, "epoch-"+str(epoch+1)+"/" , i+1, None, self.bucket_name, self.latent_dim, self.evaluador, save)
                pred+=list(pred_batch)
            if ((epoch+1)%10)==0:
                guardar_checkpoint(self.model.generator, self.bucket_name, (gen_shape[-3], gen_shape[-2]), epoch+1, "g_")
                guardar_checkpoint(self.model.generator_default, self.bucket_name, (gen_shape[-3], gen_shape[-2]), epoch+1, "df_")
                guardar_checkpoint(self.model.discriminator, self.bucket_name, (gen_shape[-3], gen_shape[-2]), epoch+1, "d_")
                guardar_checkpoint(self.model.encoder, self.bucket_name, (gen_shape[-3], gen_shape[-2]), epoch+1, "e_")
            save_inception_score(self.model.generator, "epoch-"+str(epoch+1)+"/", self.bucket_name, np.array(pred), gen_shape)