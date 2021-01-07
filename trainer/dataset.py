from audio2numpy import open_audio
import numpy as np
import librosa
from google.cloud import storage
import os
from utils import *
from scipy.io.wavfile import write

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
    songs_by_folder=[79, 39, 31, 72, 37, 39, 21, 186, 58]
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

def resample_and_save_datasets(path_dataset, bucket_name, files_format):
    dimensiones_progresivas=[
        (4,750),
        (8,1500),
        (16,3000),
        (32,6000),
        (64,12000),
        (128,24000),
        (256,48000)
    ]
    songs_by_folder=[79, 39, 31, 72, 37, 39, 21, 186, 58]
    for dimension in dimensiones_progresivas:
        for folder in range(9):
            cant_fragmentos=1
            directory="local_ds/" + files_format + "/original/" + str(folder+1) + "/"
            for song_dirname in os.listdir(directory):
                print("Preparando canción...: "+song_dirname)
                signal, sampling_rate = open_audio(song_dirname)
                list_resampled_songs=resample_song(dimension, signal, sampling_rate)
                for i in range(len(list_resampled_songs)):
                    signal = list_resampled_songs[i]
                    signal /= np.max(np.abs(signal), axis=0)
                    #guardar en mp3
                    local_path = "local_ds/mp3/"+ str(dimension[0]) + "-" + str(dimension[1]) + "/" + str(folder+1) + "/" + str(cant_fragmentos+1) + ".mp3"
                    path_upload = path_dataset + "mp3/"+ str(dimension[0]) + "-" + str(dimension[1]) + "/" + str(folder+1) + "/" + str(cant_fragmentos+1) + ".mp3"
                    folder=os.path.dirname(local_path)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    write(local_path, dimension[1], signal)
                    upload_blob(bucket_name, local_path, path_upload)
                    #guardar en wav
                    local_path = "local_ds/wav/"+ str(dimension[0]) + "-" + str(dimension[1]) + "/" + str(folder+1) + "/" + str(cant_fragmentos+1) + ".wav"
                    path_upload = path_dataset + "wav/"+ str(dimension[0]) + "-" + str(dimension[1]) + "/" + str(folder+1) + "/" + str(cant_fragmentos+1) + ".wav"
                    folder=os.path.dirname(local_path)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    write(local_path, dimension[1], signal)
                    upload_blob(bucket_name, local_path, path_upload)
                    cant_fragmentos+=1

#preparar dataset y guardar datos preparados

def preprocess_dataset(path_dataset, bucket_name, files_format):
    audio_list = []
    download_originals(path_dataset,bucket_name, files_format)
    resample_and_save_datasets(path_dataset,bucket_name, files_format)

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
            for song in range(200):
                try:
                    source_blob_name = path_dataset + files_format + "/" + str(dimension[0]) + "-" + str(dimension[1]) + "/" + str(folder+1) + "/" + str(song+1) + "."+ files_format
                    blob = bucket.blob(source_blob_name)
                    dest_file="local_ds/" + files_format + "/" + str(dimension[0]) + "-" + str(dimension[1]) + "/" + str(folder+1) + "/" + str(song+1) + "."+ files_format
                    dest_folder="local_ds/" + files_format + "/" + str(dimension[0]) + "-" + str(dimension[1]) + "/" + str(folder+1) + "/"
                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)
                    blob.download_to_filename(dest_file)
                except:
                    song=200

#read dataset by dimension

def read_dataset(dimension, files_format):
    data=[]
    directory="local_ds/" + files_format + "/"+str(dimension[0])+"-"+str(dimension[1])+"/" + str(folder+1) + "/"
    for song_dirname in os.listdir(directory):
        signal, sampling_rate=open_audio(song_dirname)
        song_reshaped = np.reshape(
                signal, newshape=(dimension[0], dimension[1], 2))
        data.append(song_reshaped)
    return np.array(data)


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
