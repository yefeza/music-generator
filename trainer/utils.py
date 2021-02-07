from scipy.io.wavfile import write
import numpy as np
import os
from google.cloud import storage
import keras
import tensorflow as tf

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
    fake = np.zeros((half_batch, 1))
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

def generar_ejemplos(g_model, prefix, iter_num, n_examples, job_dir, bucket_name, latent_dim, evaluador):
    gen_shape = g_model.output_shape
    random_latent_vectors = tf.random.normal(shape=(n_examples, latent_dim[0], latent_dim[1], latent_dim[2]))
    gen_auds = g_model(random_latent_vectors, training=False)
    for i in range(n_examples):
        signal_gen = gen_auds[i].numpy()
        signal_gen = np.reshape(signal_gen, ((gen_shape[-3]*gen_shape[-2]), 2))
        #signal_gen /= np.max(np.abs(signal_gen), axis=0)
        local_path = "local_gen/" + \
            str(gen_shape[-3]) + "x" + str(gen_shape[-2]) + \
            "/" + prefix + str(i*iter_num) + '.wav'
        path_save = "generated-data-byepoch/" + \
            str(gen_shape[-3]) + "x" + str(gen_shape[-2]) + \
            "/" + prefix + str(i*iter_num) + '.wav'
        folder=os.path.dirname(local_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        write(local_path, gen_shape[-2], signal_gen)
        upload_blob(bucket_name,local_path,path_save)
    #evaluar resultados
    pred=evaluador.predict(gen_auds)
    return pred

def generar_ejemplo(g_model, prefix, iter_num, job_dir, bucket_name, latent_dim, evaluador, save):
    gen_shape = g_model.output_shape
    random_latent_vectors = tf.random.normal(shape=(1, latent_dim[0], latent_dim[1], latent_dim[2]))
    gen_auds = g_model(random_latent_vectors)
    if save:
        signal_gen = gen_auds[0].numpy()
        signal_gen = np.reshape(signal_gen, ((gen_shape[-3]*gen_shape[-2]), 2))
        signal_gen /= np.max(np.abs(signal_gen), axis=0)
        local_path = "local_gen/" + \
            str(gen_shape[-3]) + "x" + str(gen_shape[-2]) + \
            "/" + prefix + str(iter_num) + '.wav'
        path_save = "generated-data-byepoch/" + \
            str(gen_shape[-3]) + "x" + str(gen_shape[-2]) + \
            "/" + prefix + str(iter_num) + '.wav'
        folder=os.path.dirname(local_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        write(local_path, gen_shape[-2], signal_gen)
        upload_blob(bucket_name,local_path,path_save)
    #evaluar resultados
    pred=evaluador.predict(gen_auds)
    return pred

def save_inception_score(g_model, prefix, bucket_name, pred):
    gen_shape = g_model.output_shape
    #evaluar resultados
    iscore=calculate_inception_score(pred)
    local_path = "local_gen/" + \
        str(gen_shape[-3]) + "x" + str(gen_shape[-2]) + \
        "/" + prefix + 'inception_score.txt'
    path_save = "generated-data-byepoch/" + \
        str(gen_shape[-3]) + "x" + str(gen_shape[-2]) + \
        "/" + prefix + 'inception_score.txt'
    folder=os.path.dirname(local_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_object = open(local_path,"w+")
    file_object.write("Inception Score: "+str(iscore))
    file_object.close()
    upload_blob(bucket_name,local_path,path_save)
    
# save model

def guardar_modelo(keras_model, job_dir, name):
    export_path = tf.compat.v1.keras.experimental.export_saved_model(keras_model, job_dir + '/keras_export_'+name)
    print('Model exported to: {}'.format(export_path))

def guardar_checkpoint(keras_model, bucket_name, dimension, epoch):
    storage_client = storage.Client(project='ia-devs')
    bucket = storage_client.bucket(bucket_name)
    path='ckeckpoints/'+str(dimension[0])+"-"+str(dimension[1])+"/epoch"+str(epoch)+"/"
    file_name='ckeckpoints/'+str(dimension[0])+"-"+str(dimension[1])+"/epoch"+str(epoch)+"/model.h5"
    if not os.path.exists(path):
            os.makedirs(path)
    keras_model.save(file_name)
    upload_blob(bucket_name,file_name,file_name)