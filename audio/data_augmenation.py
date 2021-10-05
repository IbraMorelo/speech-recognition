import numpy as np
import librosa
import soundfile as sf
import pathlib

def aumentation(file_path:str):
	wav, sr = librosa.load(file_path,sr=None)
	add_noise(wav, sr, file_path)
	time_shifting(wav, sr, file_path)
	time_stretching(wav, sr, file_path)
	pitch_shifting(wav, sr, file_path)

def add_noise(wav:np.array,sr:int,file_path:str):
	wav_n = wav + 0.005*np.random.normal(0,1,len(wav))
	sf.write("../5_aumentation/"+file_path.split('.')[0]+'_noise_add'+'.wav', wav_n, sr, 'PCM_16')

def time_shifting(wav:np.array,sr:int,file_path:str):
	wav_roll = np.roll(wav,int(sr/10))
	sf.write("../5_aumentation/"+file_path.split('.')[0]+'_roll_add'+'.wav', wav_roll, sr, 'PCM_16')

def time_stretching(wav:np.array,sr:int,file_path:str):
	factor = 0.4
	wav_time_stch = librosa.effects.time_stretch(wav,factor)
	sf.write("../5_aumentation/"+file_path.split('.')[0]+'_time_stech'+'.wav', wav_time_stch, sr, 'PCM_16')

def pitch_shifting(wav:np.array,sr:int,file_path:str):
	wav_pitch_sf = librosa.effects.pitch_shift(wav,sr,n_steps=4)
	sf.write("../5_aumentation/"+file_path.split('.')[0]+'_pitch_shift'+'.wav', wav_pitch_sf, sr, 'PCM_16')

def aumentation_bulk():
	data_path = pathlib.Path('.')
	for x in data_path.iterdir():
		if not str(x).endswith('.DS_Store'):
			aumentation(str(x))

aumentation_bulk()