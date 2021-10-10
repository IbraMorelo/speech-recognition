from scipy import signal
from scipy.special import softmax
from scipy.io import wavfile
from array import array
from timeit import default_timer as timer
# import tflite_runtime.interpreter as tflite
import tensorflow as tf
import numpy as np
import pyaudio
import noisereduce as nr
import wave
import logging

# Variables
chunk = 4096
threshold_volumen = 500
threshold_commands = 0.6
format_audio = pyaudio.paInt16
channels = 1
input_microphone_rate = 48000
resample_rate = 16000
record_seconds = 2
nframes = int(input_microphone_rate / chunk * record_seconds)
modelo_path = 'model_commands_recognition.tflite'
commands = ['derecha', 'rapido', 'lento', 'atras', 'adelante', 'alto', 'izquierda']
log_file = 'time_elapsed.log'
temporal_audio_file = 'temporal_file.wav'

# Init Interpreter
interpreter = tf.lite.Interpreter(modelo_path)
    # experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

logging.basicConfig(filename=log_file, 
                    encoding='utf-8', 
                    level=logging.DEBUG)


def main():
    p = pyaudio.PyAudio()
    stream = p.open(format = format_audio,
                    channels = channels,
                    rate = input_microphone_rate,
                    input = True,
                    frames_per_buffer = chunk,
                    input_device_index = 2)
    frames_noisy = []
    for i in range(0, nframes):
        data_noisy = stream.read(chunk, exception_on_overflow = False)
        frames_noisy.append(data_noisy)

    buffer_noisy = b''.join(frames_noisy)
    noise_sample = np.frombuffer(buffer_noisy, dtype=np.int16)
    loud_threshold = np.mean(np.abs(noise_sample)) * 10

    try:
        print('Escuchando')
        while True:
            
            frames = []
            for i in range(0, nframes):
                data = stream.read(chunk, exception_on_overflow = False)
                frames.append(data)

            buffer = b''.join(frames)

            wf = wave.open(temporal_audio_file, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(format_audio))
            wf.setframerate(input_microphone_rate)
            wf.writeframes(buffer)
            wf.close()

            sampling_rate, data = wavfile.read(temporal_audio_file)
            start_time = timer()
            print('data vol: ', max(data))
            if max(data) >= threshold_volumen:
                current_window = nr.reduce_noise(y=data, 
                                                 sr=input_microphone_rate, 
                                                 y_noise=noise_sample)

                data = signal.decimate(current_window, 
                                       int(sampling_rate / resample_rate))

                run_inference(data)

                end_time = timer()
                time_elapsed = end_time - start_time
                logging.debug(time_elapsed)
            else:
                print('En silencio.....')

    except KeyboardInterrupt:
        print("Saliendo...")

    stream.stop_stream()
    stream.close()
    p.terminate()

def process_audio_data(waveform):
    if len(waveform.shape) == 2:
        waveform = waveform.T[1]
    else:
        waveform = waveform
    wabs = np.abs(waveform)
    wmax = np.max(wabs)
    waveform = waveform / wmax

    PTP = np.ptp(waveform)

    waveform = 2.0 * (waveform - np.min(waveform)) / PTP - 1
 
    max_index = np.argmax(waveform)  
    start_index = max(0, max_index - 8000)
    end_index = min(max_index + 8000, waveform.shape[0])
    waveform = waveform[start_index: end_index]

    waveform_padded = np.zeros((16000,))
    waveform_padded[:waveform.shape[0]] = waveform

    return waveform_padded

def get_spectrogram(waveform):
    waveform_padded = process_audio_data(waveform)
    f, t, Zxx = signal.stft(waveform_padded, 
                            fs=16000, 
                            nperseg=255, 
                            noverlap = 124, 
                            nfft=256)
    spectrogram = np.abs(Zxx)
    return spectrogram


def run_inference(waveform):
    spectrogram = get_spectrogram(waveform)

    spectrogram= np.reshape(spectrogram, 
                            (-1, spectrogram.shape[0], spectrogram.shape[1], 1))

    input_shape = input_details[0]['shape']
    input_data = spectrogram.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], 
                           input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    tensor_results = output_data[0]
    index_selected = np.argmax(tensor_results)
    command = commands[index_selected]
    command_value = softmax(tensor_results)[index_selected]
    if command_value >= threshold_commands:
        if command == 'derecha':
            print('DERECHA')
        elif command == 'rapido':
            print('RAPIDO')
        elif command == 'lento':
            print('LENTO')
        elif command == 'atras':
            print('ATRAS')
        elif command == 'adelante':
            print('ADELANTE')
        elif command == 'alto':
            print('ALTO')
        elif command == 'izquierda':
            print('IZQUIERDA')
        else:
            print('Lo sentimos no se reconoce el comando')

if __name__ == '__main__':
    main()