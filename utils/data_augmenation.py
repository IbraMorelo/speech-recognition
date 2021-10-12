import numpy as np
import librosa
import soundfile as sf
import pathlib

def aumentation(file_path:str):
	wav, sr = librosa.load(file_path,sr=None)
	add_noise(wav, sr, file_path)
	time_shifting(wav, sr, file_path)
	time_stretch(wav, sr, file_path, 0.8)
	pitch_shifting(wav, sr, file_path)

def add_noise(wav:np.array, sr:int, file_path:str):
	wav_n = np.random.randn(len(wav))
	wav_n = wav + 0.005*wav_n
	sf.write(file_path.split('.')[0]+'_noise_add'+'.wav', wav_n, sr, 'PCM_16')

def time_shifting(wav:np.array, sr:int, file_path:str):
	wav_roll = np.roll(wav, 1600)
	sf.write(file_path.split('.')[0]+'_roll_add'+'.wav', wav_roll, sr, 'PCM_16')

def time_stretch(wav:np.array, sr:int, file_path:str, rate=1):
    input_length = 16000
    data = librosa.effects.time_stretch(wav, rate)
    if len(data) > input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    sf.write(file_path.split('.')[0]+'_stretch_add'+'.wav', data, sr, 'PCM_16')


def pitch_shifting(wav:np.array, sr:int, file_path:str):
	wav_pitch_sf = librosa.effects.pitch_shift(wav,sr,n_steps=4)
	sf.write(file_path.split('.')[0]+'_pitch_shift'+'.wav', wav_pitch_sf, sr, 'PCM_16')

def aumentation_bulk():
	data_path = pathlib.Path('/Users/ibrahimmemorelo/U/proyecto_de_grado/code/speech-recognition-data-set/rapido/initial_convertion/audio_cut/final_convertion/aumentation')
	for x in data_path.iterdir():
		if not str(x).endswith('.DS_Store'):
			aumentation(str(x))

aumentation_bulk()