# keras imports
import keras
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Input, UpSampling2D, Layer
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import LeakyReLU
from keras.layers import Add, Multiply
from keras.layers import LayerNormalization
from keras.utils.vis_utils import plot_model
from keras import backend
import tensorflow as tf
from keras.losses import categorical_crossentropy
import numpy as np
from .utils import *

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

class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        fade_in=False,
        discriminator_extra_steps=3,
        gp_weight=10.0,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.actual_step = 0
        self.total_steps = 0
        self.fade_in=fade_in

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def set_train_steps(self, size):
        self.train_steps=size

    def train_step(self, real_images):
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

        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        # Get the latent vector
        for i in range(self.d_steps):
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim[0], self.latent_dim[1], self.latent_dim[2])
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)
                # Calculate the discriminator loss using the fake and real image logits
                d_loss = self.d_loss_fn(fake_logits, real_logits)
            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
        #calculate actual delta value
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim[0], self.latent_dim[1], self.latent_dim[2]))
        generated_images = self.generator(random_latent_vectors, training=True)
        fake_logits = self.discriminator(generated_images, training=True)
        real_logits = self.discriminator(real_images, training=True)
        ci_0=tf.reduce_mean(real_logits)
        cu_0=tf.reduce_mean(fake_logits)
        delta_0=tf.math.abs((cu_0-ci_0))
        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim[0], self.latent_dim[1], self.latent_dim[2]))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Get the logits for the real images
            real_logits = self.discriminator(real_images, training=True)
            # Calculate the generator loss using the fake and real image logits
            g_loss = self.g_loss_fn(gen_img_logits, real_logits)
        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        #calculate actual delta value
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim[0], self.latent_dim[1], self.latent_dim[2]))
        generated_images = self.generator(random_latent_vectors, training=True)
        fake_logits = self.discriminator(generated_images, training=True)
        real_logits = self.discriminator(real_images, training=True)
        ci_1=tf.reduce_mean(real_logits)
        cu_1=tf.reduce_mean(fake_logits)
        delta_1=tf.math.abs((cu_1-ci_1))
        self.actual_step+=1
        return {"delta_0": delta_0, "cu_0": cu_0, "ci_0": ci_0, "delta_1": delta_1, "cu_1": cu_1, "ci_1": ci_1}

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
    def __init__(self, start_alpha=20000.0, **kwargs):
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
    def __init__(self, alpha=1500.0, **kwargs):
        super(StaticOptTanh, self).__init__(**kwargs)
        self.alpha=alpha

    def call(self, inputs):
        return tf.math.tanh(inputs) + tf.math.divide_no_nan(inputs, self.alpha)

    def get_config(self):
        config = super(StaticOptTanh, self).get_config()
        config.update({"alpha": self.alpha})
        return config

# agregar bloque a discriminador para escalar las dimensiones

def add_discriminator_block(old_model, n_input_layers=3):
    # getshapeofexistingmodel
    in_shape = list(old_model.input[0].shape)
    alpha=40000.0
    soft_alpha=20000.0
    if in_shape[-3]==8:
        alpha=60000.0
        soft_alpha=40000.0
    if in_shape[-3]==16:
        alpha=90000.0
        soft_alpha=80000.0
    if in_shape[-3]==32:
        alpha=120000.0
        soft_alpha=120000.0
    if in_shape[-3]==64:
        alpha=150000.0
        soft_alpha=150000.0
    if in_shape[-3]==128:
        alpha=200000.0
        soft_alpha=11000.0
    if in_shape[-3]==256:
        alpha=350000.0
        soft_alpha=200000.0
    # definenewinputshapeasdoublethesize
    input_shape = (in_shape[-3]*2, in_shape[-2]*2, in_shape[-1])
    in_image = Input(shape=input_shape)
    # definenewinputprocessinglayer
    featured_layer = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal')(in_image)
    featured_layer = SoftRectifier(start_alpha=soft_alpha)(featured_layer)
    #convolusion block 1
    d_1 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(featured_layer)
    d_1 = SoftRectifier(start_alpha=soft_alpha)(d_1)
    d_1 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(d_1)
    d_1 = SoftRectifier(start_alpha=soft_alpha)(d_1)
    d_1 = Conv2D(32, (2, 2), padding='same', kernel_initializer='he_normal')(d_1)
    d_1 = SoftRectifier(start_alpha=soft_alpha)(d_1)
    #convolusion block 2
    d_2 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(d_1)
    d_2 = SoftRectifier(start_alpha=soft_alpha)(d_2)
    d_2 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(d_2)
    d_2 = SoftRectifier(start_alpha=soft_alpha)(d_2)
    d_2 = Conv2D(32, (2, 2), padding='same', kernel_initializer='he_normal')(d_2)
    d_2 = SoftRectifier(start_alpha=soft_alpha)(d_2)
    #convolusion block 3
    d_3 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(d_2)
    d_3 = SoftRectifier(start_alpha=soft_alpha)(d_3)
    d_3 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(d_3)
    d_3 = SoftRectifier(start_alpha=soft_alpha)(d_3)
    d_3 = Conv2D(32, (2, 2), strides=(2,2), padding='valid', kernel_initializer='he_normal')(d_3)
    d_3 = SoftRectifier(start_alpha=soft_alpha)(d_3)
    #sumarize blocks
    d_block=Conv2D(128, (1,1), padding='same', kernel_initializer='he_normal')(d_3)
    d_block = SoftRectifier(start_alpha=soft_alpha)(d_block)
    block_new = d_block
    # skiptheinput,1x1andactivationfortheoldmodel
    for i in range(n_input_layers, len(old_model.layers)):
        if isinstance(old_model.layers[i], StaticOptTanh):
            final_layer = StaticOptTanh(alpha=alpha)(d_block)
        else:
            d_block = old_model.layers[i](d_block)
    # model 1 without multiple inputs for composite
    model1_comp = Model(in_image, final_layer)
    # downsamplethenewlargerimage
    downsample = AveragePooling2D()(in_image)
    # connectoldinputprocessingtodownsamplednewinput
    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)
    # fadeinoutputofoldmodelinputlayerwithnewinput
    d = WeightedSum()([block_old, block_new])
    # skiptheinput,1x1andactivationfortheoldmodel
    for i in range(n_input_layers, len(old_model.layers)):
        if isinstance(old_model.layers[i], StaticOptTanh):
            final_layer = StaticOptTanh(alpha=alpha)(d)
        else:
            d = old_model.layers[i](d)
    model2_comp = Model(in_image, final_layer)
    return[model1_comp, model2_comp]

# definir los discriminadores

def define_discriminator(n_blocks, input_shape=(4, 750, 2)):
    model_list = list()
    # base model input
    in_image = Input(shape=input_shape)
    # conv 1x1
    featured_block = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal')(in_image)
    featured_block = SoftRectifier()(featured_block)
    # convolusion block 1
    d_1 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(featured_block)
    d_1 = SoftRectifier()(d_1)
    d_1 = Conv2D(32, (4, 4), padding='same', kernel_initializer='he_normal')(d_1)
    d_1 = SoftRectifier()(d_1)
    d_1 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(d_1)
    d_1 = SoftRectifier()(d_1)
    # convolusion block 2
    d_2 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(d_1)
    d_2 = SoftRectifier()(d_2)
    d_2 = Conv2D(32, (4, 4), padding='same', kernel_initializer='he_normal')(d_2)
    d_2 = SoftRectifier()(d_2)
    d_2 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(d_2)
    d_2 = SoftRectifier()(d_2)
    # convolusion block 3
    d_3 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(d_2)
    d_3 = SoftRectifier()(d_3)
    d_3 = Conv2D(32, (4, 4), padding='same', kernel_initializer='he_normal')(d_3)
    d_3 = SoftRectifier()(d_3)
    d_3 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(d_3)
    d_3 = SoftRectifier()(d_3)
    #sumarize blocks
    d = MinibatchStdDev()(d_3)
    d = Flatten()(d)
    d = Dense(1)(d)
    out_class=StaticOptTanh(alpha=40000.0)(d)
    # define model
    model_comp = Model(in_image, out_class)
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

# agregar bloque a generador para escalar las dimensiones

def add_generator_block(old_model):
    # get the end of the last block
    block_end = old_model.layers[-2].output
    # upsample, and define new block
    upsampling = UpSampling2D()(block_end)
    featured = Conv2D(128, (2, 2), strides=(2,2), padding='valid', kernel_initializer='he_normal')(upsampling)
    # bloque 1 deconvolusion
    g_1 = Conv2DTranspose(16, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(featured)
    g_1 = Dense(32)(g_1)
    g_1 = Conv2DTranspose(64, (1, 2), strides=(1, 2), padding='valid', kernel_initializer='he_normal')(g_1)
    op_1 = Dense(128)(g_1)
    # bloque 2 deconvolusion
    g_2 = Conv2DTranspose(16, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(featured)
    g_2 = Dense(32)(g_2)
    g_2 = Conv2DTranspose(64, (1, 2), strides=(1, 2), padding='valid', kernel_initializer='he_normal')(g_2)
    op_2 = Dense(128)(g_2)
    # bloque 3 deconvolusion
    g_3 = Conv2DTranspose(16, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(featured)
    g_3 = Dense(32)(g_3)
    g_3 = Conv2DTranspose(64, (1, 2), strides=(1, 2), padding='valid', kernel_initializer='he_normal')(g_3)
    op_3 = Dense(128)(g_3)
    # bloque 4 deconvolusion
    g_4 = Conv2DTranspose(16, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(featured)
    g_4 = Dense(32)(g_4)
    g_4 = Conv2DTranspose(64, (1, 2), strides=(1, 2), padding='valid', kernel_initializer='he_normal')(g_4)
    op_4 = Dense(128)(g_4)
    # bloque 5 deconvolusion
    g_5 = Conv2DTranspose(16, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(featured)
    g_5 = Dense(32)(g_5)
    g_5 = Conv2DTranspose(64, (1, 2), strides=(1, 2), padding='valid', kernel_initializer='he_normal')(g_5)
    op_5 = Dense(128)(g_5)
    # bloque 3 deconvolusion
    g_6 = Conv2DTranspose(16, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(featured)
    g_6 = Dense(32)(g_6)
    g_6 = Conv2DTranspose(64, (1, 2), strides=(1, 2), padding='valid', kernel_initializer='he_normal')(g_6)
    op_6 = Dense(128)(g_6)
    #sumarize
    sumarized_blocks=Add()([op_1, op_2, op_3, op_4, op_5, op_6])
    sumarized_blocks=Dense(256)(sumarized_blocks)
    sumarized_blocks=Dense(64)(sumarized_blocks)
    sumarized_blocks=Dense(8)(sumarized_blocks)
    for_sum_layer=Dense(2)(sumarized_blocks)
    out_image = LayerNormalization(axis=[1, 2, 3])(for_sum_layer)
    # define model
    model1 = Model(old_model.input, out_image)
    # define new output image as the weighted sum of the old and new models
    merged = WeightedSum()([upsampling, for_sum_layer])
    output_2 = LayerNormalization(axis=[1, 2, 3])(merged)
    # define model
    model2 = Model(old_model.input, output_2)
    return [model1, model2]

# definir los generadores

def define_generator(n_blocks):
    model_list = list()
    # input
    ly0 = Input(shape=(1, 100, 2))
    featured = Conv2D(128, (1, 2), strides=(1,2), padding='valid', kernel_initializer='he_normal')(ly0)
    # bloque 1 deconvolusion
    g_1 = Conv2DTranspose(4, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(featured)
    g_1 = Dense(8)(g_1)
    g_1 = Conv2DTranspose(16, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(g_1)
    g_1 = Dense(32)(g_1)
    g_1 = Conv2DTranspose(64, (1, 15), strides=(1, 15), padding='valid', kernel_initializer='he_normal')(g_1)
    op_1 = Dense(128)(g_1)
    # bloque 2 deconvolusion
    g_2 = Conv2DTranspose(4, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(featured)
    g_2 = Dense(8)(g_2)
    g_2 = Conv2DTranspose(16, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(g_2)
    g_2 = Dense(32)(g_2)
    g_2 = Conv2DTranspose(64, (1, 15), strides=(1, 15), padding='valid', kernel_initializer='he_normal')(g_2)
    op_2 = Dense(128)(g_2)
    # bloque 3 deconvolusion
    g_3 = Conv2DTranspose(4, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(featured)
    g_3 = Dense(8)(g_3)
    g_3 = Conv2DTranspose(16, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(g_3)
    g_3 = Dense(32)(g_3)
    g_3 = Conv2DTranspose(64, (1, 15), strides=(1, 15), padding='valid', kernel_initializer='he_normal')(g_3)
    op_3 = Dense(128)(g_3)
    # bloque 4 deconvolusion
    g_4 = Conv2DTranspose(4, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(featured)
    g_4 = Dense(8)(g_4)
    g_4 = Conv2DTranspose(16, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(g_4)
    g_4 = Dense(32)(g_4)
    g_4 = Conv2DTranspose(64, (1, 15), strides=(1, 15), padding='valid', kernel_initializer='he_normal')(g_4)
    op_4 = Dense(128)(g_4)
    #combinar canales
    sumarized_blocks=Add()([op_1, op_2, op_3, op_4])
    sumarized_blocks=Dense(256)(sumarized_blocks)
    sumarized_blocks=Dense(64)(sumarized_blocks)
    sumarized_blocks=Dense(8)(sumarized_blocks)
    sumarized_blocks=Dense(2)(sumarized_blocks)
    wls = LayerNormalization(axis=[1, 2, 3])(sumarized_blocks)
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

# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def discriminator_loss(fake_logits, real_logits):
    ci=tf.reduce_mean(real_logits)
    cu=tf.reduce_mean(fake_logits)
    lamb=(cu-ci)
    delta=tf.math.abs(lamb)
    sign=tf.math.divide_no_nan(lamb, delta)+0.0001
    sign_2=(tf.math.divide_no_nan(lamb, delta)+0.0000999)*-1.0
    return (sign * ci) + (sign_2 * cu)

# Define the loss functions for the generator.
def generator_loss(fake_logits, real_logits):
    ci=tf.reduce_mean(real_logits)
    cu=tf.reduce_mean(fake_logits)
    lamb=(cu-ci)
    delta=tf.math.abs(lamb)
    sign=tf.math.divide_no_nan(lamb, delta)+0.0001
    return (sign * ci) + delta
    
# define composite models for training generators via discriminators

def define_composite(discriminators, generators, latent_dim):
    model_list = list()
    # create composite models
    for i in range(len(discriminators)):
        g_models, d_models = generators[i], discriminators[i]
        # straight-through model
        #d_models[2].trainable = False
        wgan1 = WGAN(
            discriminator=d_models[0],
            generator=g_models[0],
            latent_dim=latent_dim,
            discriminator_extra_steps=1,
        )
        wgan1.compile(
            d_optimizer=Adam(lr=0.0005, beta_1=0, beta_2=0.99, epsilon=10e-8),
            g_optimizer=Adam(lr=0.0005, beta_1=0, beta_2=0.99, epsilon=10e-8),
            g_loss_fn=generator_loss,
            d_loss_fn=discriminator_loss
        )
        # fade-in model
        #d_models[3].trainable = False
        wgan2 = WGAN(
            discriminator=d_models[1],
            generator=g_models[1],
            latent_dim=latent_dim,
            fade_in=True,
            discriminator_extra_steps=1,
        )
        wgan2.compile(
            d_optimizer=Adam(lr=0.0005, beta_1=0, beta_2=0.99, epsilon=10e-8),
            g_optimizer=Adam(lr=0.0005, beta_1=0, beta_2=0.99, epsilon=10e-8),
            g_loss_fn=generator_loss,
            d_loss_fn=discriminator_loss
        )
        # store
        model_list.append([wgan1, wgan2])
    return model_list

#checkpoint

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, job_dir, evaluador, num_examples=20, latent_dim=(1, 100, 2)):
        self.num_examples = num_examples
        self.latent_dim = latent_dim
        self.bucket_name = "music-gen"
        self.job_dir = job_dir
        self.evaluador = evaluador
    
    def on_epoch_end(self, epoch, logs=None):
        iters_gen=self.num_examples
        pred=[]
        if not self.model.fade_in:
            for i in range(iters_gen):
                if ((epoch+1)%10)==0:
                    save=True
                else:
                    save=False
                pred_batch=generar_ejemplo(self.model.generator, "epoch-"+str(epoch+1)+"/" , i+1, None, self.bucket_name, self.latent_dim, self.evaluador, save)
                pred+=list(pred_batch)
                gen_shape = self.model.generator.output_shape
                if save:
                    guardar_checkpoint(self.model.generator, self.bucket_name, (gen_shape[-3], gen_shape[-2]), epoch+1, "g_")
                    guardar_checkpoint(self.model.discriminator, self.bucket_name, (gen_shape[-3], gen_shape[-2]), epoch+1, "d_")
            save_inception_score(self.model.generator, "epoch-"+str(epoch+1)+"/", self.bucket_name, np.array(pred))