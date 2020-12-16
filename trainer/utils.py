from .models import *
from scipy.io.wavfile import write
import numpy as np
import os
from google.cloud import storage

# update alpha for Weighted Sum


def update_fadein(models, step, n_steps):
    # calculate current alpha (linear from 0 to 1)
    alpha = step / float(n_steps - 1)
    # update the alpha for each model
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)

# generate fake samples


def generate_fake_samples(g_model, latent_dim, half_batch):
    noise = np.random.uniform(-1, 1, (half_batch,
                                      latent_dim[0], latent_dim[1], latent_dim[2]))
    gen_auds = g_model.predict(noise)
    fake = np.ones((half_batch, 1))*-1
    return gen_auds, fake

# prepare real samples


def generate_real_samples(dataset, half_batch):
    gen_auds = dataset[np.random.randint(0, dataset.shape[0], half_batch)]
    valid = np.ones((half_batch, 1))
    return gen_auds, valid

# helper for Gradient Penalty


def lerp(a, b, t):
    with tf.name_scope('Lerp'):
        return a + (b - a) * t

# generate random noises


def generate_latent_points(latent_dim, n_batch):
    noise = np.random.uniform(-1, 1, (n_batch,
                                      latent_dim[0], latent_dim[1], latent_dim[2]))
    return noise

# calcuate gradients of prediction


def get_gradients(img_input, top_pred_idx, model, y_true, penalty):
    """Computes the gradients of outputs w.r.t input image.

    Args:
        img_input: 4D image tensor
        top_pred_idx: Predicted label for the input image

    Returns:
        Gradients of the predictions w.r.t img_input
    """
    images = tf.cast(img_input, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(images)
        preds = model([images, y_true, penalty])
        top_class = preds[:, top_pred_idx]

    grads = tape.gradient(top_class, images)
    return grads

# gets Gradient Penalty value for loss


def get_gradient_penalty(fake_images, real_images, minibatch_size, d_model, wgan_target=1.0):
    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.compat.v1.random_uniform(
            [minibatch_size, 1, 1, 1], 0.0, 1.0)
        mixed_images_out = lerp(real_images, fake_images, mixing_factors)
        g_paded = np.zeros((minibatch_size, 1))
        y=np.ones((minibatch_size,1))*-1
        #w_loss = d_model.train_on_batch([mixed_images_out, y, g_paded])
        mixed_scores_out = d_model.predict([mixed_images_out, y, g_paded])
        mixed_loss = tf.reduce_sum(mixed_scores_out)
        top_pred_idx = tf.argmax(mixed_scores_out[0])
        mixed_grads = get_gradients(mixed_images_out, top_pred_idx, d_model, y, g_paded)[0]
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads)))
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    return gradient_penalty

# generar datos con el modelo entrenado

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client(project='ia-devs')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

def generar_ejemplos(g_model, prefix, n_examples, job_dir, bucket_name):
    gen_shape = g_model.output_shape
    noise = np.random.uniform(-1, 1, (n_examples, 1, 5, 2))
    gen_auds = g_model.predict(noise)
    gen_auds = np.reshape(gen_auds, newshape=(
        n_examples, (gen_shape[-3]*gen_shape[-2]), 2))
    for i in range(n_examples):
        signal_gen = gen_auds[i]
        signal_gen /= np.max(np.abs(signal_gen), axis=0)
        local_path = "local_gen/" + \
            str(gen_shape[-3]) + "x" + str(gen_shape[-2]) + \
            "/" + prefix + "-" + str(i) + '.wav'
        path_save = "generated-data/" + \
            str(gen_shape[-3]) + "x" + str(gen_shape[-2]) + \
            "/" + prefix + "-" + str(i) + '.wav'
        folder=os.path.dirname(local_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        write(local_path, gen_shape[-2], signal_gen)
        upload_blob(bucket_name,local_path,path_save)

# save model


def guardar_modelo(keras_model, job_dir, name):
    export_path = tfcompat.v1.keras.experimental.export_saved_model(keras_model, job_dir + '/keras_export_'+name)
    print('Model exported to: {}'.format(export_path))
