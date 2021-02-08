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

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, g_loss_fn_extra):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.g_loss_fn_extra = g_loss_fn_extra

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def set_train_steps(self, size):
        self.train_steps=size

    def pre_train_discriminator(self, real_images, batch_size):
        random_latent_vectors = tf.random.normal(
            shape=(real_images.shape[0], self.latent_dim[0], self.latent_dim[1], self.latent_dim[2])
        )
        fake_images = self.generator.predict(random_latent_vectors)
        real_labels=np.ones((real_images.shape[0]))
        fake_labels=-np.ones((real_images.shape[0]))
        X=np.concatenate((real_images,fake_images))
        y=np.concatenate((real_labels,fake_labels))
        print(X.shape)
        print(y.shape)
        self.discriminator.compile(optimizer='adam', loss='mse', metrics=["accuracy"])
        self.discriminator.fit(X,y, batch_size=batch_size, epochs=50)

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
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim[0], self.latent_dim[1], self.latent_dim[2])
        )
        for i in range(self.d_steps):
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)
                # Calculate the discriminator loss using the fake and real image logits
                d_loss = self.d_loss_fn(fake_logits, real_logits)
                # Calculate the gradient penalty
                #gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                #d_loss = d_cost + gp * self.gp_weight
                #d_loss = d_cost
            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            if d_gradient==None:
                raise ValueError( "fake: "+str(fake_logits) +"real: "+str(real_logits) +"loss: "+str(d_loss) )
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        #random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim[0], self.latent_dim[1], self.latent_dim[2]))
        #for i in range(self.d_steps):
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Get the logits for the real images
            real_logits = self.discriminator(real_images, training=True)
            g_loss = self.g_loss_fn_extra(gen_img_logits, real_logits)
        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        if gen_gradient==None:
                raise ValueError( "fake: "+str(gen_img_logits) +"real: "+str(real_logits) +"loss: "+str(g_loss) )
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        #segunda funcion de costo
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Get the logits for the real images
            real_logits = self.discriminator(real_images, training=True)
            g_loss_2 = self.g_loss_fn(gen_img_logits, real_logits)
        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss_2, self.generator.trainable_variables)
        if gen_gradient==None:
                raise ValueError( "fake: "+str(gen_img_logits) +"real: "+str(real_logits) +"loss: "+str(g_loss_2) )
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        #with tf.GradientTape() as tape:
        # Generate fake images using the generator
        generated_images = self.generator(random_latent_vectors, training=True)
        # Get the discriminator logits for fake images
        gen_img_logits = self.discriminator(generated_images, training=True)
        # Get the logits for the real images
        real_logits = self.discriminator(real_images, training=True)
        pd_distance = self.g_loss_fn(gen_img_logits, real_logits)
        self.actual_step+=1
        return {"d_loss": d_loss, "g_loss": g_loss, "g_loss_2": g_loss_2, "pd_distance": pd_distance}

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

# agregar bloque a discriminador para escalar las dimensiones

def add_discriminator_block(old_model, n_input_layers=3):
    # getshapeofexistingmodel
    in_shape = list(old_model.input[0].shape)
    # definenewinputshapeasdoublethesize
    input_shape = (in_shape[-3]*2, in_shape[-2]*2, in_shape[-1])
    in_image = Input(shape=input_shape)
    # definenewinputprocessinglayer
    featured_layer = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal')(in_image)
    featured_layer = LeakyReLU(alpha=0.2)(featured_layer)
    #convolusion block 1
    d_1 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(featured_layer)
    d_1 = LeakyReLU(alpha=0.2)(d_1)
    d_1 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(d_1)
    d_1 = LeakyReLU(alpha=0.2)(d_1)
    #d = AveragePooling2D()(d)
    d_1 = Conv2D(128, (2, 2), padding='same', kernel_initializer='he_normal')(d_1)
    d_1 = LeakyReLU(alpha=0.2)(d_1)
    #convolusion block 2
    d_2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(d_1)
    d_2 = LeakyReLU(alpha=0.2)(d_2)
    d_2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(d_2)
    d_2 = LeakyReLU(alpha=0.2)(d_2)
    d_2 = Conv2D(128, (2, 2), padding='same', kernel_initializer='he_normal')(d_2)
    d_2 = LeakyReLU(alpha=0.2)(d_2)
    #convolusion block 3
    d_3 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(d_1)
    d_3 = LeakyReLU(alpha=0.2)(d_3)
    d_3 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(d_3)
    d_3 = LeakyReLU(alpha=0.2)(d_3)
    d_3 = Conv2D(128, (2, 2), strides=(2,2), padding='valid', kernel_initializer='he_normal')(d_3)
    d_3 = LeakyReLU(alpha=0.2)(d_3)
    #sumarize blocks
    d_block=Conv2D(128, (1,1), padding='same', kernel_initializer='he_normal')(d_3)
    d_block = LeakyReLU(alpha=0.2)(d_block)
    block_new = d_block
    # skiptheinput,1x1andactivationfortheoldmodel
    pointer=0
    for i in range(n_input_layers, len(old_model.layers)):
        if isinstance(old_model.layers[i], Dense):
            final_layer = old_model.layers[i](d_block)
        else:
            d_block = old_model.layers[i](d_block)
    # model 1 without multiple inputs for composite
    model1_comp = Model(in_image, final_layer)
    # compilemodel
    #model1.compile(loss=None, optimizer=Adam(
    #    lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    # downsamplethenewlargerimage
    downsample = AveragePooling2D()(in_image)
    # connectoldinputprocessingtodownsamplednewinput
    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)
    # fadeinoutputofoldmodelinputlayerwithnewinput
    d = WeightedSum()([block_old, block_new])
    # skiptheinput,1x1andactivationfortheoldmodel
    for i in range(n_input_layers, len(old_model.layers)):
        if isinstance(old_model.layers[i], Dense):
            final_layer = old_model.layers[i](d)
        else:
            d = old_model.layers[i](d)
    # definestraight-throughmodel
    #model2 = Model([in_image, y_true, is_weight], final_layer)
    #model2.add_loss(D_wgangp_acgan(y_true, final_layer, is_weight))
    # model 2 without multiple inputs for composite
    model2_comp = Model(in_image, final_layer)
    # compilemodel
    #model2.compile(loss=None, optimizer=Adam(
    #    lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    return[model1_comp, model2_comp]

# definir los discriminadores

def define_discriminator(n_blocks, lstm_layer, input_shape=(4, 750, 2)):
    model_list = list()
    # base model input
    in_image = Input(shape=input_shape)
    # conv 1x1
    featured_block = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal')(in_image)
    featured_block = LeakyReLU(alpha=0.2)(featured_block)
    # convolusion block 1
    d_1 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(featured_block)
    d_1 = LeakyReLU(alpha=0.2)(d_1)
    d_1 = Conv2D(128, (4, 4), padding='same', kernel_initializer='he_normal')(d_1)
    d_1 = LeakyReLU(alpha=0.2)(d_1)
    d_1 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(d_1)
    d_1 = LeakyReLU(alpha=0.2)(d_1)
    # convolusion block 2
    d_2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(d_1)
    d_2 = LeakyReLU(alpha=0.2)(d_2)
    d_2 = Conv2D(128, (4, 4), padding='same', kernel_initializer='he_normal')(d_2)
    d_2 = LeakyReLU(alpha=0.2)(d_2)
    d_2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(d_2)
    d_2 = LeakyReLU(alpha=0.2)(d_2)
    # convolusion block 3
    d_3 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(d_1)
    d_3 = LeakyReLU(alpha=0.2)(d_3)
    d_3 = Conv2D(128, (4, 4), padding='same', kernel_initializer='he_normal')(d_3)
    d_3 = LeakyReLU(alpha=0.2)(d_3)
    d_3 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(d_3)
    d_3 = LeakyReLU(alpha=0.2)(d_3)
    #sumarize blocks
    d = MinibatchStdDev()(d_3)
    d = Flatten()(d)
    out_class = Dense(1, activation='tanh')(d)
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
    featured = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(upsampling)
    #bloque 1
    g_1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(featured)
    g_1 = LeakyReLU(alpha=0.2)(g_1)
    g_1 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(g_1)
    g_1 = LeakyReLU(alpha=0.2)(g_1)
    g_1 = Conv2D(64, (6,6), padding='same', kernel_initializer='he_normal')(g_1)
    g_1 = LeakyReLU(alpha=0.2)(g_1)
    g_1 = Conv2D(128, (6, 6), padding='same', kernel_initializer='he_normal')(g_1)
    g_1 = LeakyReLU(alpha=0.2)(g_1)
    op_1 = Dense(50)(g_1)
    #bloque 2
    g_2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(featured)
    g_2 = LeakyReLU(alpha=0.2)(g_2)
    g_2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(g_2)
    g_2 = LeakyReLU(alpha=0.2)(g_2)
    g_2 = Conv2D(64, (6,6), padding='same', kernel_initializer='he_normal')(g_2)
    g_2 = LeakyReLU(alpha=0.2)(g_2)
    g_2 = Conv2D(128, (6, 6), padding='same', kernel_initializer='he_normal')(g_2)
    g_2 = LeakyReLU(alpha=0.2)(g_2)
    op_2 = Dense(50)(g_2)
    #bloque 3
    g_3 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(featured)
    g_3 = LeakyReLU(alpha=0.2)(g_3)
    g_3 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(g_3)
    g_3 = LeakyReLU(alpha=0.2)(g_3)
    g_3 = Conv2D(64, (6,6), padding='same', kernel_initializer='he_normal')(g_3)
    g_3 = LeakyReLU(alpha=0.2)(g_3)
    g_3 = Conv2D(128, (6, 6), padding='same', kernel_initializer='he_normal')(g_3)
    g_3 = LeakyReLU(alpha=0.2)(g_3)
    op_3 = Dense(50)(g_3)
    #bloque 4
    g_4 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(featured)
    g_4 = LeakyReLU(alpha=0.2)(g_4)
    g_4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(g_4)
    g_4 = LeakyReLU(alpha=0.2)(g_4)
    g_4 = Conv2D(64, (6,6), padding='same', kernel_initializer='he_normal')(g_4)
    g_4 = LeakyReLU(alpha=0.2)(g_4)
    g_4 = Conv2D(128, (6, 6), padding='same', kernel_initializer='he_normal')(g_4)
    g_4 = LeakyReLU(alpha=0.2)(g_4)
    op_4 = Dense(100)(g_4)
    #bloque 5
    g_5 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(featured)
    g_5 = LeakyReLU(alpha=0.2)(g_5)
    g_5 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(g_5)
    g_5 = LeakyReLU(alpha=0.2)(g_5)
    g_5 = Conv2D(64, (6,6), padding='same', kernel_initializer='he_normal')(g_5)
    g_5 = LeakyReLU(alpha=0.2)(g_5)
    g_5 = Conv2D(128, (6, 6), padding='same', kernel_initializer='he_normal')(g_5)
    g_5 = LeakyReLU(alpha=0.2)(g_5)
    op_5 = Dense(100)(g_5)
    #bloque 6
    g_6 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(featured)
    g_6 = LeakyReLU(alpha=0.2)(g_6)
    g_6 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(g_6)
    g_6 = LeakyReLU(alpha=0.2)(g_6)
    g_6 = Conv2D(64, (6,6), padding='same', kernel_initializer='he_normal')(g_6)
    g_6 = LeakyReLU(alpha=0.2)(g_6)
    g_6 = Conv2D(128, (6, 6), padding='same', kernel_initializer='he_normal')(g_6)
    g_6 = LeakyReLU(alpha=0.2)(g_6)
    op_6 = Dense(100)(g_6)
    #sumarize
    sumarized_blocks=Add()([op_1,op_2,op_3])
    sumarized_blocks=Dense(100)(sumarized_blocks)
    # to 2 channels
    for_sum_layer = Conv2D(2, (1, 1), padding='same', kernel_initializer='he_normal', activation='linear')(sumarized_blocks)
    out_image = LayerNormalization(axis=[1, 2, 3])(for_sum_layer)
    # define model
    model1 = Model(old_model.input, out_image)
    #model1.get_layer(name="shared_layer").trainable=False
    # get the output layer from old model
    #out_old = old_model.layers[-2]
    # define new output image as the weighted sum of the old and new models
    merged = WeightedSum()([upsampling, for_sum_layer])
    merged = LayerNormalization(axis=[1, 2, 3])(merged)
    # define model
    model2 = Model(old_model.input, merged)
    #model2.get_layer(name="shared_layer").trainable=False
    return [model1, model2]

# definir los generadores

def define_generator(n_blocks, lstm_layer):
    model_list = list()
    # input
    ly0 = Input(shape=(1, 50, 2))
    featured = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(ly0)
    # bloque 1 deconvolusion
    g_1 = Conv2DTranspose(32, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(featured)
    g_1 = LeakyReLU(alpha=0.2)(g_1)
    g_1 = Conv2DTranspose(64, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(g_1)
    g_1 = LeakyReLU(alpha=0.2)(g_1)
    g_1 = Conv2DTranspose(128, (1, 15), strides=(1, 15), padding='valid', kernel_initializer='he_normal')(g_1)
    g_1 = LeakyReLU(alpha=0.2)(g_1)
    g_1 = Conv2D(64, (6,6), padding='same', kernel_initializer='he_normal')(g_1)
    g_1 = LeakyReLU(alpha=0.2)(g_1)
    g_1 = Conv2D(128, (6, 6), padding='same', kernel_initializer='he_normal')(g_1)
    g_1 = LeakyReLU(alpha=0.2)(g_1)
    op_1 = Dense(50)(g_1)
    # bloque 2 deconvolusion
    g_2 = Conv2DTranspose(32, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(featured)
    g_2 = LeakyReLU(alpha=0.2)(g_2)
    g_2 = Conv2DTranspose(64, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(g_2)
    g_2 = LeakyReLU(alpha=0.2)(g_2)
    g_2 = Conv2DTranspose(128, (1, 15), strides=(1, 15), padding='valid', kernel_initializer='he_normal')(g_2)
    g_2 = LeakyReLU(alpha=0.2)(g_2)
    g_2 = Conv2D(64, (6,6), padding='same', kernel_initializer='he_normal')(g_2)
    g_2 = LeakyReLU(alpha=0.2)(g_2)
    g_2 = Conv2D(128, (6, 6), padding='same', kernel_initializer='he_normal')(g_2)
    g_2 = LeakyReLU(alpha=0.2)(g_2)
    op_2 = Dense(50)(g_2)
    # bloque 3 deconvolusion
    g_3 = Conv2DTranspose(32, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(featured)
    g_3 = LeakyReLU(alpha=0.2)(g_3)
    g_3 = Conv2DTranspose(64, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(g_3)
    g_3 = LeakyReLU(alpha=0.2)(g_3)
    g_3 = Conv2DTranspose(128, (1, 15), strides=(1, 15), padding='valid', kernel_initializer='he_normal')(g_3)
    g_3 = LeakyReLU(alpha=0.2)(g_3)
    g_3 = Conv2D(64, (6,6), padding='same', kernel_initializer='he_normal')(g_3)
    g_3 = LeakyReLU(alpha=0.2)(g_3)
    g_3 = Conv2D(128, (6, 6), padding='same', kernel_initializer='he_normal')(g_3)
    g_3 = LeakyReLU(alpha=0.2)(g_3)
    op_3 = Dense(50)(g_3)
    # bloque 4 deconvolusion
    g_4 = Conv2DTranspose(32, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(featured)
    g_4 = LeakyReLU(alpha=0.2)(g_4)
    g_4 = Conv2DTranspose(64, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(g_4)
    g_4 = LeakyReLU(alpha=0.2)(g_4)
    g_4 = Conv2DTranspose(128, (1, 15), strides=(1, 15), padding='valid', kernel_initializer='he_normal')(g_4)
    g_4 = LeakyReLU(alpha=0.2)(g_4)
    g_4 = Conv2D(64, (6,6), padding='same', kernel_initializer='he_normal')(g_4)
    g_4 = LeakyReLU(alpha=0.2)(g_4)
    g_4 = Conv2D(128, (6, 6), padding='same', kernel_initializer='he_normal')(g_4)
    g_4 = LeakyReLU(alpha=0.2)(g_4)
    op_4 = Dense(100)(g_4)
    # bloque 5 deconvolusion
    g_5 = Conv2DTranspose(32, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(featured)
    g_5 = LeakyReLU(alpha=0.2)(g_5)
    g_5 = Conv2DTranspose(64, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(g_5)
    g_5 = LeakyReLU(alpha=0.2)(g_5)
    g_5 = Conv2DTranspose(128, (1, 15), strides=(1, 15), padding='valid', kernel_initializer='he_normal')(g_5)
    g_5 = LeakyReLU(alpha=0.2)(g_5)
    g_5 = Conv2D(64, (6,6), padding='same', kernel_initializer='he_normal')(g_5)
    g_5 = LeakyReLU(alpha=0.2)(g_5)
    g_5 = Conv2D(128, (6, 6), padding='same', kernel_initializer='he_normal')(g_5)
    g_5 = LeakyReLU(alpha=0.2)(g_5)
    op_5 = Dense(100)(g_5)
    # bloque 6 deconvolusion
    g_6 = Conv2DTranspose(32, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(featured)
    g_6 = LeakyReLU(alpha=0.2)(g_6)
    g_6 = Conv2DTranspose(64, (2, 1), strides=(2, 1), padding='valid', kernel_initializer='he_normal')(g_6)
    g_6 = LeakyReLU(alpha=0.2)(g_6)
    g_6 = Conv2DTranspose(128, (1, 15), strides=(1, 15), padding='valid', kernel_initializer='he_normal')(g_6)
    g_6 = LeakyReLU(alpha=0.2)(g_6)
    g_6 = Conv2D(64, (6,6), padding='same', kernel_initializer='he_normal')(g_6)
    g_6 = LeakyReLU(alpha=0.2)(g_6)
    g_6 = Conv2D(128, (6, 6), padding='same', kernel_initializer='he_normal')(g_6)
    g_6 = LeakyReLU(alpha=0.2)(g_6)
    op_6 = Dense(100)(g_6)
    #to 2 channels
    sumarized_blocks=Add()([op_1, op_2, op_3])
    sumarized_blocks=Dense(100)(sumarized_blocks)
    sumarized_blocks = Conv2D(2, (1, 1), padding='same', kernel_initializer='he_normal', activation='linear')(sumarized_blocks)
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
    real_loss=tf.reduce_mean(real_logits)
    fake_loss=tf.reduce_mean(fake_logits)
    return (((fake_loss-real_loss)/(tf.math.abs((fake_loss-real_loss))+0.0001))+0.0001)*real_loss


# Define the loss functions for the generator.
def generator_loss_extra(fake_logits, real_logits):
    real_loss=tf.reduce_mean(real_logits)
    fake_loss=tf.reduce_mean(fake_logits)
    return (((fake_loss-real_loss)/(tf.math.abs((fake_loss-real_loss))+0.0001))+0.0001)*fake_loss

def generator_loss(fake_logits, real_logits):
    real_loss = tf.reduce_mean(real_logits)
    fake_loss = tf.reduce_mean(fake_logits)
    return tf.math.abs(fake_loss - real_loss)

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
            discriminator_extra_steps=3,
        )
        wgan1.compile(
            d_optimizer=Adam(
            lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8),
            g_optimizer=Adam(
            lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8),
            g_loss_fn=generator_loss,
            g_loss_fn_extra=generator_loss_extra,
            d_loss_fn=discriminator_loss,
        )
        # fade-in model
        #d_models[3].trainable = False
        wgan2 = WGAN(
            discriminator=d_models[1],
            generator=g_models[1],
            latent_dim=latent_dim,
            fade_in=True,
            discriminator_extra_steps=3,
        )
        wgan2.compile(
            d_optimizer=Adam(
            lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8),
            g_optimizer=Adam(
            lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8),
            g_loss_fn=generator_loss,
            g_loss_fn_extra=generator_loss_extra,
            d_loss_fn=discriminator_loss,
        )
        # store
        model_list.append([wgan1, wgan2])
    return model_list

#checkpoint

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, job_dir, evaluador, num_examples=20, latent_dim=(1, 50, 2)):
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
                if ((epoch+1)%5)==0:
                    save=True
                else:
                    save=False
                pred_batch=generar_ejemplo(self.model.generator, "epoch-"+str(epoch+1)+"/" , i+1, None, self.bucket_name, self.latent_dim, self.evaluador, save)
                pred.append(pred_batch[0])
                gen_shape = self.model.generator.output_shape
                if save:
                    guardar_checkpoint(self.model.generator, self.bucket_name, (gen_shape[-3], gen_shape[-2]), epoch+1)
            save_inception_score(self.model.generator, "epoch-"+str(epoch+1)+"/", self.bucket_name, np.array(pred))