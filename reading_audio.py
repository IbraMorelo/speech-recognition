from tensorflow.python.keras.backend import dtype
import pyaudio
import wave
import random
from scipy.io.wavfile import write
import sounddevice as sd
import pathlib
import os


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1
PATH = os.path.dirname(os.path.realpath(__file__))
WAVE_OUTPUT_FILENAME = PATH+'/data_sets/'+str(random.getrandbits(128)) + '.wav'

p = pyaudio.PyAudio()

def showDivice():
    device_count = p.get_device_count()
    for i in range(0, device_count):
        print("Name: " + p.get_device_info_by_index(i)["name"])
        print(p.get_device_info_by_index(i)["index"])
        print("\n")


def convertAudio(frames):
    # write(WAVE_OUTPUT_FILENAME, RATE, frames)
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def recordAudio():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    convertAudio(frames)
    print("* done recording")
    return WAVE_OUTPUT_FILENAME
