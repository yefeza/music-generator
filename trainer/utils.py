from .models import *
from scipy.io.wavfile import write
import numpy as np
import os
from google.cloud import storage
import keras
from .evaluacion import calculate_inception_score

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

def generar_ejemplos(g_model, prefix, n_examples, job_dir, bucket_name, latent_dim, evaluador):
    gen_shape = g_model.output_shape
    random_latent_vectors = tf.random.normal(shape=(n_examples, latent_dim[0], latent_dim[1], latent_dim[2]))
    gen_auds = g_model(random_latent_vectors)
    for i in range(n_examples):
        signal_gen = gen_auds[i].numpy()
        signal_gen = np.reshape(signal_gen, ((gen_shape[-3]*gen_shape[-2]), 2))
        signal_gen /= np.max(np.abs(signal_gen), axis=0)
        local_path = "local_gen/" + \
            str(gen_shape[-3]) + "x" + str(gen_shape[-2]) + \
            "/" + prefix + str(i) + '.wav'
        path_save = "generated-data/" + \
            str(gen_shape[-3]) + "x" + str(gen_shape[-2]) + \
            "/" + prefix + str(i) + '.wav'
        folder=os.path.dirname(local_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        write(local_path, gen_shape[-2], signal_gen)
        upload_blob(bucket_name,local_path,path_save)
    #evaluar resultados
    pred=evaluador.predict(gen_auds)
    iscore=calculate_inception_score(pred)
    local_path = "local_gen/" + \
        str(gen_shape[-3]) + "x" + str(gen_shape[-2]) + \
        "/" + prefix + 'inception_score_'+str(iscore)+'.txt'
    path_save = "generated-data/" + \
        str(gen_shape[-3]) + "x" + str(gen_shape[-2]) + \
        "/" + prefix + 'inception_score_'+str(iscore)+'.txt'
    folder=os.path.dirname(local_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_object = open(local_path,"w+")
    file_object.write("Inception Score: "+str(iscore))
    upload_blob(bucket_name,local_path,path_save)

# save model

def guardar_modelo(keras_model, job_dir, name):
    export_path = tf.compat.v1.keras.experimental.export_saved_model(keras_model, job_dir + '/keras_export_'+name)
    print('Model exported to: {}'.format(export_path))

def guardar_checkpoint(keras_model, job_dir, dimension, epoch):
    path=job_dir + '/ckeckpoints/'+dimension[0]+"-"+dimension[1]+"/epoch"+str(epoch)+"/"
    export_path = tf.compat.v1.keras.experimental.export_saved_model(keras_model, path)
    keras_model = tf.compat.v1.keras.experimental.load_from_saved_model(path)

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, job_dir, evaluador, num_examples=1000, latent_dim=(1, 5, 2)):
        self.num_examples = num_examples
        self.latent_dim = latent_dim
        self.bucket_name = "music-gen"
        self.job_dir = job_dir
        self.evaluador = evaluador
    
    def on_epoch_end(self, epoch, logs=None):
        if not self.model.fade_in:
            generar_ejemplos(self.model.generator, "epoch-"+str(epoch)+"/" , self.num_examples, None, self.bucket_name, self.latent_dim, self.evaluador)
            gen_shape = self.model.generator.output_shape
            guardar_checkpoint(self.model.generator, self.job_dir, (gen_shape[-3], gen_shape[-2]), epoch)