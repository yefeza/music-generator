import argparse
import os
from .dataset import *
from .models import *
from .utils import *
import numpy as np
import matplotlib.pyplot as plt
from .evaluacion import *
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
        default=20,
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
    prepare_data = False

    # direccion de el dataset

    path_dataset = 'keras_dir/full-ds/'
    bucket_name='music-gen'
    files_format='mp3'
    download_data=True
    dimension_start=6
    folder_start=7
    song_start=45
    fragment_start=10
    download_evaluators=False
    epochs_evaluadores=100


    #preparar o descargar el dataset

    if prepare_data:
        preprocess_dataset(path_dataset,bucket_name,files_format, download_data, dimension_start, folder_start, song_start, fragment_start)
    else:
        if download_data:
            #download_full_dataset(path_dataset,bucket_name,files_format)
            pass

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # size of the latent space
    latent_dim = (1, 5, 2)

    with strategy.scope():
        # shared layer

        shared_layer = Dense(100, name="shared_layer")

        # define discriminators

        discriminators = define_discriminator(7, shared_layer)

        # define generator

        generators = define_generator(7, shared_layer)

        # define composite models

        composite = define_composite(discriminators, generators, latent_dim)

    def plot_losses(history):
        plt.plot(history.history['d_loss']*-1, label='Negative D Loss')
        plt.plot(history.history['g_loss'], label='G Loss')
        plt.title('Losses History')
        plt.ylabel('Loses')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.show()

    # train the generator and discriminator

    def train(g_models, d_models, gan_models, dataset, latent_dim, e_norm, e_fadein, batch_sizes, job_dir, bucket_name, files_format, path_dataset, download_data, download_evaluators):
        # fit the baseline model
        g_normal, d_normal, gan_normal = g_models[0][0], d_models[0][0], gan_models[0][0]
        # scale dataset to appropriate size
        gen_shape = g_normal.output_shape
        if download_data:
            download_diension_dataset(path_dataset, bucket_name, files_format, (gen_shape[-3], gen_shape[-2]))
        scaled_data, y_evaluator = read_dataset((gen_shape[-3], gen_shape[-2]),files_format)
        #cargar evaluador
        evaluador=load_evaluator((gen_shape[-3], gen_shape[-2]), bucket_name,download_evaluators, (scaled_data, y_evaluator), e_norm)
        # train normal or straight-through models
        n_batch=batch_sizes[0]
        #limit to round sizes data
        limit=int((scaled_data.shape[0]/n_batch))*n_batch
        #total_steps
        n_steps=int((scaled_data.shape[0]/n_batch))*e_norm
        scaled_data=scaled_data[:limit]
        print('Scaled Data', scaled_data.shape)
        #train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm, n_batch, bucket_name)
        gan_models[0][0].set_train_steps(n_steps)
        cbk=GANMonitor(job_dir=job_dir, evaluador=evaluador)
        history = gan_models[0][0].fit(scaled_data, batch_size=n_batch, epochs=e_norm, callbacks=[cbk])
        plot_losses(history)
        # generate examples
        generar_ejemplos(gan_models[0][0].generator, "first-", 3, job_dir, bucket_name, latent_dim)
        # process each level of growth
        for i in range(1, len(g_models)):
            # retrieve models for this level of growth
            [g_normal, g_fadein] = g_models[i]
            [d_normal, d_fadein] = d_models[i]
            [gan_normal, gan_fadein] = gan_models[i]
            # scale dataset to appropriate size
            gen_shape = g_normal.output_shape
            if download_data:
                download_diension_dataset(path_dataset, bucket_name, files_format, (gen_shape[-3], gen_shape[-2]))
            scaled_data, y_evaluator = read_dataset((gen_shape[-3], gen_shape[-2]),files_format)
            #cargar evaluador
            evaluador=load_evaluator((gen_shape[-3], gen_shape[-2]), bucket_name,download_evaluators, (scaled_data, y_evaluator), e_norm)
            #scaled_data = get_resampled_data(gen_shape[-3], gen_shape[-2], dataset)
            #get batch size for model
            n_batch=batch_sizes[i]
            #limit to round sizes data
            limit=int((scaled_data.shape[0]/n_batch))*n_batch
            #total_steps
            n_steps=int((scaled_data.shape[0]/n_batch))*e_fadein
            scaled_data=scaled_data[:limit]
            print('Scaled Data', scaled_data.shape)
            # train fade-in models for next level of growth
            gan_models[i][1].set_train_steps(n_steps)
            history = gan_models[i][1].fit(scaled_data, batch_size=n_batch, epochs=e_fadein, callbacks=[cbk])
            plot_losses(history)
            #train_epochs(g_fadein, d_fadein, gan_fadein,
            #            scaled_data, e_fadein, n_batch, True)
            # train normal or straight-through models
            #total_steps
            n_steps=int((scaled_data.shape[0]/n_batch))*e_norm
            gan_models[i][0].set_train_steps(n_steps)
            history = gan_models[i][0].fit(scaled_data, batch_size=n_batch, epochs=e_norm, callbacks=[cbk])
            plot_losses(history)
            #train_epochs(g_normal, d_normal, gan_normal,
            #            scaled_data, e_norm, n_batch)
            # generate examples
            generar_ejemplos(gan_models[i][1].generator, "fade-3-", 1, job_dir, bucket_name, latent_dim)
            generar_ejemplos(gan_models[i][0].generator, "norm-3-", 3, job_dir, bucket_name, latent_dim)
        print("guardando modelo")
        guardar_modelo(gan_models[6][0].generator,job_dir,"final_100_epoch")

    batch_sizes=[16,8,4,2,2,2,2]
    # load image data
    dataset = []
    #dataset = get_audio_list(path_dataset, bucket_name)
    e_norm=NUM_EPOCHS
    e_fadein=int(e_norm/4)
    # train model
    train(generators, discriminators, composite, dataset,
        latent_dim, e_norm, e_fadein, batch_sizes, JOB_DIR, bucket_name, files_format, path_dataset, download_data, download_evaluators)