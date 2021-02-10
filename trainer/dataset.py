from audio2numpy import open_audio
import numpy as np
import librosa
from google.cloud import storage
import os
from .utils import *
from scipy.io.wavfile import write
import tensorflow as tf

#descargar de cloud storage
def download_audio_files(path_dataset, bucket_name):
    storage_client = storage.Client(project='ia-devs')
    bucket = storage_client.bucket(bucket_name)
    local_files_list=[]
    for i in range(25):
        source_blob_name = path_dataset + str(i+1) + ".wav"
        blob = bucket.blob(source_blob_name)
        dest_file="local_ds/"+ str(i+1) + ".wav"
        if not os.path.exists("local_ds/"):
            os.makedirs("local_ds/")
        blob.download_to_filename(dest_file)
        local_files_list.append(dest_file)
    return local_files_list

# leer audios

def get_audio_list(path_dataset, bucket_name):
    audio_list = []
    local_files_paths=download_audio_files(path_dataset,bucket_name)
    for fp in local_files_paths:
        signal, sampling_rate = open_audio(fp)
        audio_list.append((signal, sampling_rate))
    return audio_list


#descargar dataset original de cloud storage

def download_originals(path_dataset, bucket_name, files_format):
    storage_client = storage.Client(project='ia-devs')
    bucket = storage_client.bucket(bucket_name)
    #songs_by_folder=[79, 39, 31, 72, 37, 39, 21, 186, 58]
    songs_by_folder=[2, 1, 1, 1, 1, 1, 1, 1, 1]
    for folder in range(9):
        limit_songs=songs_by_folder[folder]
        for song in range(limit_songs):
            source_blob_name = path_dataset + files_format + "/original/" + str(folder+1) + "/" + str(song+1) + "."+ files_format
            blob = bucket.blob(source_blob_name)
            dest_file="local_ds/" + files_format + "/original/" + str(folder+1) + "/" + str(song+1) + "."+ files_format
            dest_folder="local_ds/" + files_format + "/original/" + str(folder+1) + "/"
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            blob.download_to_filename(dest_file)

#resample song

def resample_song(dimensiones, audio_array, actual_sample_rate):
    # resample
    resampled = librosa.resample(audio_array.transpose(), actual_sample_rate, dimensiones[1])
    resampled = resampled.transpose()
    #reshape and pad
    long_limit = dimensiones[0]*dimensiones[1]
    # pad song
    if audio_array.shape[0] > long_limit:
        inicio = 0
        fin = inicio+long_limit
        segundos_song = resampled.shape[0]/dimensiones[1]
        max_trozos = int(segundos_song/dimensiones[0])
        fragmentos=[]
        for j in range(max_trozos):
            song_flat = resampled[inicio:fin]
            fragmentos.append(song_flat)
            inicio = fin
            fin = inicio+long_limit
        return np.array(fragmentos)
    else:
        faltantes = long_limit-resampled.shape[0]
        song = np.pad(resampled, ((0, faltantes), (0, 0)),
                        'constant', constant_values=0)
        return np.array([song, ])

#resamplear y cortar el dataset para todas las redes progresivas

def resample_and_save_datasets(path_dataset, bucket_name, files_format, dimension_start, folder_start, song_start, fragment_start):
    dimensiones_progresivas=[
        (4,750),
        (8,1500),
        (16,3000),
        (32,6000),
        (64,12000),
        (128,24000),
        (256,48000)
    ]
    dimensiones_progresivas=dimensiones_progresivas[dimension_start:len(dimensiones_progresivas)]
    for dimension in dimensiones_progresivas:
        for folder in range(folder_start,9):
            cant_fragmentos=fragment_start
            directory="local_ds/" + files_format + "/original/" + str(folder+1) + "/"
            lista_canciones=os.listdir(directory)
            lista_canciones=lista_canciones[song_start:len(lista_canciones)]
            for song_dirname in lista_canciones:
                print("Preparando canción...: "+ directory + song_dirname)
                try:
                    signal, sampling_rate = open_audio(directory+song_dirname)
                    list_resampled_songs=resample_song(dimension, signal, sampling_rate)
                    for i in range(len(list_resampled_songs)):
                        signal = list_resampled_songs[i]
                        #guardar en mp3
                        local_path = "local_ds/mp3/"+ str(dimension[0]) + "-" + str(dimension[1]) + "/" + str(folder+1) + "/" + str(cant_fragmentos) + ".mp3"
                        path_upload = path_dataset + "mp3/"+ str(dimension[0]) + "-" + str(dimension[1]) + "/" + str(folder+1) + "/" + str(cant_fragmentos) + ".mp3"
                        folder_name=os.path.dirname(local_path)
                        if not os.path.exists(folder_name):
                            os.makedirs(folder_name)
                        write(local_path, dimension[1], signal)
                        upload_blob(bucket_name, local_path, path_upload)
                        #guardar en wav
                        local_path = "local_ds/wav/"+ str(dimension[0]) + "-" + str(dimension[1]) + "/" + str(folder+1) + "/" + str(cant_fragmentos) + ".wav"
                        path_upload = path_dataset + "wav/"+ str(dimension[0]) + "-" + str(dimension[1]) + "/" + str(folder+1) + "/" + str(cant_fragmentos) + ".wav"
                        folder_name=os.path.dirname(local_path)
                        if not os.path.exists(folder_name):
                            os.makedirs(folder_name)
                        write(local_path, dimension[1], signal)
                        upload_blob(bucket_name, local_path, path_upload)
                        cant_fragmentos+=1
                except:
                    pass
            #restablecer para la siguiente carpeta
            song_start=0
        #restablecer para la siguiente dimension
        folder_start=0
        fragment_start=1
#preparar dataset y guardar datos preparados

def preprocess_dataset(path_dataset, bucket_name, files_format, download_data, dimension_start, folder_start, song_start, fragment_start):
    audio_list = []
    if download_data:
        download_originals(path_dataset,bucket_name, files_format)
    resample_and_save_datasets(path_dataset,bucket_name, files_format, dimension_start, folder_start, song_start, fragment_start)

#descargar dataset completo de cloud storage (Cuando ya hay un dataset preparado)

def download_full_dataset(path_dataset, bucket_name, files_format):
    storage_client = storage.Client(project='ia-devs')
    bucket = storage_client.bucket(bucket_name)
    dimensiones_progresivas=[
        (4,750),
        (8,1500),
        (16,3000),
        (32,6000),
        (64,12000),
        (128,24000),
        (256,48000)
    ]
    for dimension in dimensiones_progresivas:
        for folder in range(9):
            for song in range(15000):
                try:
                    source_blob_name = path_dataset + files_format + "/" + str(dimension[0]) + "-" + str(dimension[1]) + "/" + str(folder+1) + "/" + str(song+1) + "."+ files_format
                    blob = bucket.blob(source_blob_name)
                    dest_file="local_ds/" + files_format + "/" + str(dimension[0]) + "-" + str(dimension[1]) + "/" + str(folder+1) + "/" + str(song+1) + "."+ files_format
                    dest_folder="local_ds/" + files_format + "/" + str(dimension[0]) + "-" + str(dimension[1]) + "/" + str(folder+1) + "/"
                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)
                    blob.download_to_filename(dest_file)
                except:
                    song=15000

#descargar dataset completo de cloud storage (Cuando ya hay un dataset preparado)

def download_diension_dataset(path_dataset, bucket_name, files_format, dimension):
    storage_client = storage.Client(project='ia-devs')
    bucket = storage_client.bucket(bucket_name)
    #limit_songs=300
    #limit_songs_list=[3000,1500,1000,600,300,200,100,80,50]
    #limit_songs_list=[600,90,80,60,50,40,30,20,10]
    #limit_songs_list=[600,560,480,360,300,240,180,120,60]
    limit_songs_list=[20,15,10,5,5,5,5,5,5]
    limit_songs=50
    if dimension[0]==4:
        limit_songs=limit_songs_list[0]
    if dimension[0]==8:
        limit_songs=limit_songs_list[1]
    if dimension[0]==16:
        limit_songs=limit_songs_list[2]
    if dimension[0]==32:
        limit_songs=limit_songs_list[3]
    if dimension[0]==64:
        limit_songs=limit_songs_list[4]
    if dimension[0]==128:
        limit_songs=limit_songs_list[5]
    if dimension[0]==256:
        limit_songs=limit_songs_list[6]
    for folder in range(9):
        print("downloading from folder "+str(folder+1) + " and dimension " + str(dimension[0]) + "-" + str(dimension[1]))
        for song in range(limit_songs):
            try:
                dest_file="local_ds/" + files_format + "/" + str(dimension[0]) + "-" + str(dimension[1]) + "/" + str(folder+1) + "/" + str(song+1) + "."+ files_format
                if not os.path.exists(dest_file):
                    source_blob_name = path_dataset + files_format + "/" + str(dimension[0]) + "-" + str(dimension[1]) + "/" + str(folder+1) + "/" + str(song+1) + "."+ files_format
                    #habilitar para todas las pistas de la carpeta
                    #prefix = path_dataset + files_format + "/" + str(dimension[0]) + "-" + str(dimension[1]) + "/" + str(folder+1)
                    #blobs = bucket.list_blobs(prefix=prefix)
                    blob = bucket.blob(source_blob_name)
                    dest_folder="local_ds/" + files_format + "/" + str(dimension[0]) + "-" + str(dimension[1]) + "/" + str(folder+1) + "/"
                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)
                    blob.download_to_filename(dest_file)
            except:
                song=limit_songs

#read dataset by dimension

def read_dataset(dimension, files_format):
    data=[]
    y_evaluator=[]
    limit_songs=20
    for folder in range(9):
        continuos_error=0
        print("Leyendo dataset en folder "+str(folder+1))
        directory="local_ds/" + files_format + "/"+str(dimension[0])+"-"+str(dimension[1])+"/" + str(folder+1) + "/"
        songs_dir=1
        for song_dirname in os.listdir(directory):
            try:
                signal, sampling_rate=open_audio(directory+song_dirname)
                song_reshaped = np.reshape(
                        signal, newshape=(dimension[0], dimension[1], 2))
                data.append(song_reshaped)
                y_evaluator.append(folder)
                continuos_error=0
                songs_dir+=1
                #if songs_dir>=limit_songs and dimension[0]==4:
                #    break
            except:
                continuos_error+=1
                if continuos_error==5:
                    break
    categorical_y=tf.compat.v1.keras.utils.to_categorical(np.array(y_evaluator), num_classes=9)
    categorical_y=tf.constant(categorical_y)
    return np.array(data), categorical_y.numpy()


# resamplear y recortar audios

def get_resampled_data(seconds, resample_rate, audio_list):
    # resample
    resampled_data = []
    for song in audio_list:
        signal = song[0]
        sample_rate = song[1]
        resampled = librosa.resample(
            signal.transpose(), sample_rate, resample_rate)
        resampled_data.append(resampled.transpose())
        print(resampled.transpose().shape)
    #reshape and pad
    long_limit = seconds*resample_rate
    padded_data = []
    song_number = 1
    for song in resampled_data:
        # pad song
        if song.shape[0] > long_limit:
            inicio = 0
            fin = inicio+(seconds*resample_rate)
            segundos_song = song.shape[0]/resample_rate
            max_trozos = int(segundos_song/seconds)
            for j in range(max_trozos):
                song_flat = song[inicio:fin]
                if (song_flat.shape[0] >= (seconds*resample_rate)):
                    song_reshaped = np.reshape(
                        song_flat, newshape=(seconds, resample_rate, 2))
                    padded_data.append(song_reshaped)
                    inicio = fin
                    fin = inicio+(seconds*resample_rate)
        else:
            faltantes = long_limit-song.shape[0]
            song = np.pad(song, ((0, faltantes), (0, 0)),
                          'constant', constant_values=0)
            song_flat = song
            song_reshaped = np.reshape(
                song_flat, newshape=(seconds, resample_rate, 2))
            print(song_reshaped.shape)
            padded_data.append(song_reshaped)
        song_number += 1
    return np.array(padded_data)
