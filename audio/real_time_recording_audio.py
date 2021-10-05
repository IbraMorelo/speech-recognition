from scipy.io import wavfile
from scipy import signal
import numpy as np
import argparse 
import pyaudio
import wave
import time
import tensorflow as tf
import scipy.signal

# from tflite_runtime.interpreter import Interpreter

def main():
    CHUNK = 4096
    FORMAT = pyaudio.paInt32
    CHANNELS = 1
    RATE = 48000
    RE_RATE = 16000 
    RECORD_SECONDS = 2
    WAVE_OUTPUT_FILENAME = "test.wav"
    NFRAMES = int((RATE * RECORD_SECONDS) / CHUNK)

    # initialize pyaudio
    p = pyaudio.PyAudio()

    print('opening stream...')
    stream = p.open(format = FORMAT,
                    channels = CHANNELS,
                    rate = RATE,
                    input = True,
                    frames_per_buffer = CHUNK,
                    input_device_index = 2)

    try:
        while True:
            print("Listening...")

            frames = []
            for i in range(0, NFRAMES):
                data = stream.read(CHUNK, exception_on_overflow = False)
                frames.append(data)

            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            sampling_rate, data = wavfile.read(WAVE_OUTPUT_FILENAME)

            data = signal.decimate(data, int(RATE / RE_RATE))

            max_index = np.argmax(data)  
            start_index = max(0, max_index-8000)
            end_index = min(max_index+8000, data.shape[0])
            data = data[start_index:end_index]
 
            run_inference(data)
    except KeyboardInterrupt:
        print("exiting...")

    stream.stop_stream()
    stream.close()
    p.terminate()

def process_audio_data(waveform):

    # if stereo, pick the left channel
    if len(waveform.shape) == 2:
        print("Stereo detected. Picking one channel.")
        waveform = waveform.T[1]
    else: 
        waveform = waveform 

    # normalise audio
    wabs = np.abs(waveform)
    wmax = np.max(wabs)
    waveform = waveform / wmax

    PTP = np.ptp(waveform)
    print("peak-to-peak: %.4f. Adjust as needed." % (PTP,))

    # return None if too silent 
    if PTP < 0.5:
        return []

    # scale and center
    waveform = 2.0*(waveform - np.min(waveform))/PTP - 1

    # extract 16000 len (1 second) of data   
    max_index = np.argmax(waveform)  
    start_index = max(0, max_index-8000)
    end_index = min(max_index+8000, waveform.shape[0])
    waveform = waveform[start_index:end_index]

    waveform_padded = np.zeros((16000,))
    waveform_padded[:waveform.shape[0]] = waveform

    return waveform_padded

def get_spectrogram(waveform):
    
    waveform_padded = process_audio_data(waveform)

    if not len(waveform_padded):
        return []

    # compute spectrogram 
    f, t, Zxx = signal.stft(waveform_padded, fs=16000, nperseg=255, 
        noverlap = 124, nfft=256)
    # Output is complex, so take abs value
    spectrogram = np.abs(Zxx)
        
    return spectrogram


def run_inference(waveform):

    # get spectrogram data 
    spectrogram = get_spectrogram(waveform)

    if not len(spectrogram):
        print("Too silent. Skipping...")
        #time.sleep(1)
        return 

    spectrogram1= np.reshape(spectrogram, (-1, spectrogram.shape[0], spectrogram.shape[1], 1))

    # load TF Lite model
    interpreter = tf.lite.Interpreter('/Users/ibrahimmemorelo/U/proyecto_de_grado/models/model_commands_recognition.tflite')
    interpreter.allocate_tensors()
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_data = spectrogram1.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    print("running inference...")
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    yvals = output_data[0]
    commands = ['adelante', 'atras', 'alto', 'derecha', 'izquierda', 'rapido', 'lento']

    print(output_data[0])
    print(">>> " + commands[np.argmax(output_data[0])].upper())

if __name__ == '__main__':
    main()