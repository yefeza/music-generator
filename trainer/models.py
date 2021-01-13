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

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

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

    def train_step(self, real_images):
        #control actual step
        self.actual_step+=1
        if self.fade_in:
            models=[self.discriminator, self.generator]
            # calculate current alpha (linear from 0 to 1)
            alpha = self.actual_step / float(self.total_steps - 1)
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
        for i in range(self.d_steps):
            # Get the latent vector
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
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim[0], self.latent_dim[1], self.latent_dim[2]))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}

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
    #model1 = Model([in_image, y_true, is_weight], final_layer)
    #model1.add_loss(D_wgangp_acgan(y_true, final_layer, is_weight))
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
    # model.add_loss(D_wgangp_acgan(y_true, out_class, is_weight))
    # compile model
    model.compile(loss=None, optimizer=Adam(
        lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
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
    g = Conv2D(128, (3, 3), padding='same',
               kernel_initializer='he_normal')(upsampling)
    g = LeakyReLU(alpha=0.2)(g)
    g = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(g)
    g = LeakyReLU(alpha=0.2)(g)
    out_image = Conv2DTranspose(2, (1, 1), padding='same')(g)
    # define model
    model1 = Model(old_model.input, out_image)
    model1.get_layer(name="shared_layer").trainable=False
    # get the output layer from old model
    out_old = old_model.layers[-1]
    # connect the upsampling to the old output layer
    out_image2 = out_old(upsampling)
    # define new output image as the weighted sum of the old and new models
    merged = WeightedSum()([out_image2, out_image])
    # define model
    model2 = Model(old_model.input, merged)
    model2.get_layer(name="shared_layer").trainable=False
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
    wls = Reshape(target_shape=(1, 50, 2))(g_lstm_layer)
    wls = Conv2DTranspose(2, (1, 15), strides=(1, 15), padding='valid')(wls)
    wls = LeakyReLU(alpha=0.2)(wls)
    wls = Conv2DTranspose(128, (4, 1), strides=(4, 1), padding='valid')(wls)
    wls = LeakyReLU(alpha=0.2)(wls)
    wls = Conv2DTranspose(2, (1, 1), padding='same')(wls)
    wls = LeakyReLU(alpha=0.2)(wls)
    model = Model(ly0, wls)
    model.get_layer(name="shared_layer").trainable=False
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
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)

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
            discriminator_extra_steps=2,
        )
        wgan1.compile(
            d_optimizer=Adam(
            lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8),
            g_optimizer=Adam(
            lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8),
            g_loss_fn=generator_loss,
            d_loss_fn=discriminator_loss,
        )
        # fade-in model
        #d_models[3].trainable = False
        wgan2 = WGAN(
            discriminator=d_models[1],
            generator=g_models[1],
            latent_dim=latent_dim,
            fade_in=True,
            discriminator_extra_steps=2,
        )
        wgan2.compile(
            d_optimizer=Adam(
            lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8),
            g_optimizer=Adam(
            lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8),
            g_loss_fn=generator_loss,
            d_loss_fn=discriminator_loss,
        )
        # store
        model_list.append([wgan1, wgan2])
    return model_list

#checkpoint

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, job_dir, evaluador, num_examples=100, latent_dim=(1, 5, 2)):
        self.num_examples = num_examples
        self.latent_dim = latent_dim
        self.bucket_name = "music-gen"
        self.job_dir = job_dir
        self.evaluador = evaluador
    
    def on_epoch_end(self, epoch, logs=None):
        iters_gen=int(self.num_examples/2)
        pred=[]
        if not self.model.fade_in:
            for i in range(iters_gen):
                pred_batch=generar_ejemplos(self.model.generator, "epoch-"+str(epoch)+"/" , int(self.num_examples/50), None, self.bucket_name, self.latent_dim, self.evaluador)
                for fila in pred_batch:
                    pred.append(fila)
                gen_shape = self.model.generator.output_shape
                guardar_checkpoint(self.model.generator, self.bucket_name, (gen_shape[-3], gen_shape[-2]), epoch)
            save_inception_score(self.model.generator, "epoch-"+str(epoch)+"/", self.bucket_name, np.array(pred))