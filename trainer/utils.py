from scipy.io.wavfile import write
import numpy as np
import os
from google.cloud import storage
import keras
import tensorflow as tf
import random

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


def generar_ejemplo(g_model, enc_model, gen_shape, random_real_data, prefix, iter_num, job_dir, bucket_name, latent_dim, evaluador, save):
    if iter_num<=5:
        random_latent_vectors = tf.random.uniform(shape=(10, latent_dim[0], latent_dim[1]), minval=-1., maxval=1.)
    else:
        if iter_num<=10:
            random_encoder_input = tf.random.uniform(shape=(10, gen_shape[-3]*gen_shape[-2], gen_shape[-1]), minval=-1., maxval=1.)
            random_latent_vectors = enc_model(random_encoder_input, training=False)
        else:
            if iter_num<=15:
                random_encoder_input = tf.random.uniform(shape=(10, gen_shape[-3]*gen_shape[-2], gen_shape[-1]), minval=-1., maxval=1.)
                random_ecoded = enc_model(random_encoder_input, training=False)
                random_noise = tf.random.uniform(shape=(10, latent_dim[0], latent_dim[1]), minval=-1., maxval=1.)
                random_latent_vectors=random_noise+random_ecoded
            else:
                random_latent_vectors = enc_model(random_real_data, training=False)
    gen_auds = g_model(random_latent_vectors, training=False)
    if save:
        signal_gen = gen_auds[random.randrange(0,len(gen_auds))].numpy()
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

def save_inception_score(g_model, prefix, bucket_name, pred, dimension):
    gen_shape = dimension
    #evaluar resultados
    iscore=calculate_inception_score(pred)
    local_path = "local_gen/" + \
        str(gen_shape[0]) + "x" + str(gen_shape[1]) + \
        "/" + prefix + 'inception_score.txt'
    path_save = "generated-data-byepoch/" + \
        str(gen_shape[0]) + "x" + str(gen_shape[1]) + \
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

def guardar_checkpoint(keras_model, bucket_name, dimension, epoch, prefix):
    storage_client = storage.Client(project='ia-devs')
    bucket = storage_client.bucket(bucket_name)
    path='ckeckpoints/'+str(dimension[0])+"-"+str(dimension[1])+"/epoch"+str(epoch)+"/"
    file_name='ckeckpoints/'+str(dimension[0])+"-"+str(dimension[1])+"/epoch"+str(epoch)+"/"+prefix+"model.h5"
    if not os.path.exists(path):
            os.makedirs(path)
    keras_model.save(file_name)
    upload_blob(bucket_name,file_name,file_name)