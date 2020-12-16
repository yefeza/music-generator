import argparse
import os
from .dataset import *
from .models import *
from .utils import *
import numpy as np

def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting '
             'models')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=1000,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--batch-size',
        default=24,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning-rate',
        default=.01,
        type=float,
        help='learning rate for gradient descent, default=.01')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args = get_args()
    JOB_DIR = args.job_dir
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size

    # direccion de el dataset

    path_dataset = 'keras_dir/dswav/'
    bucket_name='music-gen'

    # lstm shared layer

    lstm_layer = Dense(100)

    # define discriminators

    discriminators = define_discriminator(7, lstm_layer)
    discriminators[0][1].summary()
    
    # define generator

    generators = define_generator(7, lstm_layer)

    # define composite models

    composite = define_composite(discriminators, generators)

    # train a generator and discriminator


    def train_epochs(g_model, d_model, gan_model, dataset, n_epochs, n_batch, fadein=False):
        # calculate the number of batches per training epoch
        bat_per_epo = int(dataset.shape[0] / n_batch)
        # calculate the number of training iterations
        n_steps = bat_per_epo * n_epochs
        # calculate the size of half a batch of samples
        half_batch = int(n_batch / 2)
        # manually enumerate epochs
        for i in range(n_steps):
            # update alpha for all WeightedSum layers when fading in new blocks
            if fadein:
                update_fadein([g_model, d_model, gan_model], i, n_steps)
            # prepare real and fake samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            #X_mixed, y_mixed = generate_mixed_samples(X_real, X_fake, half_batch)
            X = np.concatenate((X_fake, X_real))
            y = np.concatenate((y_fake, y_real))
            # update discriminator model
            gradient_penalty = get_gradient_penalty(
                X_fake, X_real, half_batch, d_model)
            g_paded = np.zeros((half_batch*2, 1))
            g_paded[0][0] = np.array([gradient_penalty, ])
            w_loss = d_model.train_on_batch([X, y, g_paded])
            # update the generator via the discriminator's error
            z_input = generate_latent_points(latent_dim, n_batch)
            y_real2 = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(z_input, y_real2)
            # summarize loss on this batch
            print('>%d, d=%.3f, g=%.3f' % (i+1, w_loss, g_loss))

    # train the generator and discriminator


    def train(g_models, d_models, gan_models, dataset, latent_dim, e_norm, e_fadein, batch_sizes, job_dir, bucket_name):
        # fit the baseline model
        g_normal, d_normal, gan_normal = g_models[0][0], d_models[0][0], gan_models[0][0]
        # scale dataset to appropriate size
        gen_shape = g_normal.output_shape
        scaled_data = get_resampled_data(gen_shape[-3], gen_shape[-2], dataset)
        print('Scaled Data', scaled_data.shape)
        # train normal or straight-through models
        n_batch=batch_sizes[0]
        train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm, n_batch, bucket_name)
        # generate examples
        generar_ejemplos(g_normal, "first-", 3, job_dir, bucket_name)
        # process each level of growth
        for i in range(1, len(g_models)):
            # retrieve models for this level of growth
            [g_normal, g_fadein] = g_models[i]
            [d_normal, d_fadein, dcomp_normal, dcomp_fadein] = d_models[i]
            [gan_normal, gan_fadein] = gan_models[i]
            # scale dataset to appropriate size
            gen_shape = g_normal.output_shape
            scaled_data = get_resampled_data(gen_shape[-3], gen_shape[-2], dataset)
            print('Scaled Data', scaled_data.shape)
            n_batch=batch_sizes[i]
            # train fade-in models for next level of growth
            train_epochs(g_fadein, d_fadein, gan_fadein,
                        scaled_data, e_fadein, n_batch, True)
            # train normal or straight-through models
            train_epochs(g_normal, d_normal, gan_normal,
                        scaled_data, e_norm, n_batch)
            # generate examples
            generar_ejemplos(g_fadein, "fade-3-", 1, job_dir, bucket_name)
            generar_ejemplos(g_normal, "norm-3-", 3, job_dir, bucket_name)
            # guardar modelos
            guardar_modelo(g_normal, job_dir, str(
                gen_shape[-3])+"x"+str(gen_shape[-2]))


    # size of the latent space
    latent_dim = (1, 5, 2)
    batch_sizes=[16,8,4,2,2,2,2]
    # load image data
    dataset = get_audio_list(path_dataset, bucket_name)

    # train model
    train(generators, discriminators, composite, dataset,
        latent_dim, NUM_EPOCHS, NUM_EPOCHS, batch_sizes, JOB_DIR, bucket_name)