from audio2numpy import open_audio
import numpy as np
import librosa
from google.cloud import storage
import os

#descargar de cloud storage
def download_audio_files(path_dataset, bucket_name):
    storage_client = storage.Client(project='ia-devs')
    bucket = storage_client.bucket(bucket_name)
    local_files_list=[]
    for i in range(13):
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
