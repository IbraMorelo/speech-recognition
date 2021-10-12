import numpy as np
from scipy.io import wavfile
import pathlib
import os
from playsound import playsound

def preprocess_audio(file_path:str):
    if file_path.endswith('.wav'):
        fs, signal = wavfile.read(file_path)
        signal = signal / (2**15)
        signal_len = len(signal)
        segment_size_t = 0.25 
        segment_size = int(segment_size_t * fs)
        segments = np.array([signal[x:x + segment_size] for x in np.arange(0, signal_len, segment_size)])
        energies = [(s**2).sum() / len(s) for s in segments]
        thres = 0.5 * np.median(energies)
        index_of_segments_to_keep = (np.where(energies > thres)[0])
        print('Indices originales: ', index_of_segments_to_keep)
        segments2 = segments[index_of_segments_to_keep]
        new_signal = np.concatenate(segments2)
        print('Sample rate original: ', len(new_signal))

        # Borrar segmento
        index_of_segments_to_keep = np.delete(index_of_segments_to_keep, 0)
        # index_of_segments_to_keep = np.delete(index_of_segments_to_keep, 0)
        # index_of_segments_to_keep = np.delete(index_of_segments_to_keep, 0)
        # index_of_segments_to_keep = np.delete(index_of_segments_to_keep, 0)
        # index_of_segments_to_keep = np.delete(index_of_segments_to_keep, 0)
        # index_of_segments_to_keep = np.delete(index_of_segments_to_keep, 8)
        # index_of_segments_to_keep = np.delete(index_of_segments_to_keep, 7)
        # index_of_segments_to_keep = np.delete(index_of_segments_to_keep, 6)
        # index_of_segments_to_keep = np.delete(index_of_segments_to_keep, 5)
        # index_of_segments_to_keep = np.delete(index_of_segments_to_keep, 4)

        print('Nuevos indices: ', index_of_segments_to_keep)
        segments2 = segments[index_of_segments_to_keep]
        new_signal = np.concatenate(segments2)
        print('Nuevo sample rate: ', len(new_signal))
        file_path_new = file_path.split('.')[:-1][0].split('/')
        file_name = file_path_new[-1]
        file_path_new = file_path_new[:-1]
        file_path_new.append("audio_cut")
        file_path_new = "/".join(file_path_new)
        wavfile.write(file_path_new+"/"+file_name+".wav", fs, new_signal)

        playsound(file_path_new+"/"+file_name+".wav")

        print("Eliminar File: 1: YES, 2: No")
        file_name_remove = input()
        if int(file_name_remove) == 1:
            if os.path.exists(file_path):
                os.remove(file_path)

def cut_bulk():
    var = 0
    for file in pathlib.Path("/Users/ibrahimmemorelo/U/proyecto_de_grado/code/speech-recognition-data-set/rapido/initial_convertion").iterdir():
        if str(file).endswith('.wav'):
            var = var + 1
            if var == 1:
                print('-------------------------')
                print('Nombre archivo', file)
                print('-------------------------')
                preprocess_audio(str(file))
    #     if not str(file).endswith('.DS_Store'):

cut_bulk()