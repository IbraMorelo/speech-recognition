from scipy import signal
from scipy.special import softmax
from array import array
from timeit import default_timer as timer
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify
import numpy as np
import pyaudio
import noisereduce as nr
import logging
import uuid

# Variables
chunk = 1024
threshold_volumen = 500
threshold_commands = 0.6
format_audio = pyaudio.paInt16
channels = 1
input_microphone_rate = 48000
resample_rate = 16000
model_path = 'models/model_commands_recognition_edgetpu.tflite'
log_file = 'time_elapsed.log'
factor_downsampling = int(input_microphone_rate / resample_rate)
commands = np.empty(0)
with open('commands.txt', 'r') as f:
    new = f.readlines()
    commands = []
    for w in new:
        k = w.replace('\n', '')
        commands.append(k)

# Init Interpreter
interpreter = edgetpu.make_interpreter(model_path)
interpreter.allocate_tensors()

logging.basicConfig(filename=log_file,
                    format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)

class SpeechRecognition:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.frames = []

    def __enter__(self):
        self.stream = self.p.open(format = format_audio,
                                  channels = channels,
                                  rate = input_microphone_rate,
                                  input = True,
                                  frames_per_buffer = chunk,
                                  input_device_index = 2,
                                  stream_callback=self.enqueue_frames)

    def __exit__(self, exception_type, exception_value, traceback):
        self.stream.stop_stream()
        self.stream.close() 
        self.p.terminate()   

    def enqueue_frames(self, in_data, *_):
        self.frames.append(in_data)  
        return None, pyaudio.paContinue

    def get_record(self):
        start_time = timer()
        isFinished = False
        count = 0
        record_frames = []
        copy_frame = self.frames[:]
        for frame in copy_frame:
            if count > 0 and count <= 43:
                count += 1
                record_frames.append(frame)
                if count == 43:
                    isFinished = True
                    self.frames = []
                    break
            elif not max(array('h',frame)) < threshold_volumen:
                record_frames.append(frame)
                count = 1
            else:
                self.frames.remove(frame)

        if isFinished:
            isStarted = False
            new_frames = b''.join(record_frames)
            waveform_original = np.frombuffer(new_frames, dtype=np.int16)
            waveform_processed = self.process_audio_data(waveform_original)
            spectrogram = self.get_spectrogram(waveform_processed)
            command = self.run_inference(spectrogram)

            # Log
            time_elapsed = timer() - start_time
            random_name = uuid.uuid4().hex
            logging.debug(' | ' + str(time_elapsed) + ' | ' + command + ' | ' + random_name)

            return command
        else:
            return ''

    def waveform_reduce_noise(self, waveform):
        return nr.reduce_noise(y=waveform,
                               sr=input_microphone_rate)

    def waveform_downsampling(self, waveform):
        return signal.decimate(waveform, 
                               factor_downsampling)
    
    def waveform_complete_if_needed(self, waveform):
        waveform_padded = np.zeros((resample_rate,))
        waveform_padded[:waveform.shape[0]] = waveform
        return waveform_padded     

    def process_audio_data(self, waveform):
        waveform_without_noisy = self.waveform_reduce_noise(waveform)
        waveform_resampling = self.waveform_downsampling(waveform_without_noisy)
        waveform_padded = self.waveform_complete_if_needed(waveform_resampling)
        return waveform_padded

    def get_spectrogram(self, waveform):
        f, t, Zxx = signal.stft(waveform, 
                                fs=resample_rate, 
                                nperseg=255, 
                                noverlap = 124, 
                                nfft=256)
        spectrogram = np.abs(Zxx)
        return spectrogram

    def run_inference(self, spectrogram):
        spectrogram= np.reshape(spectrogram, 
                                (-1, spectrogram.shape[0], spectrogram.shape[1], 1))

        input_data = spectrogram.astype(np.float32)

        common.set_input(interpreter, input_data)
        
        interpreter.invoke()
        
        tensor_results = classify.get_scores(interpreter)
        index_selected = np.argmax(tensor_results)
        command = commands[index_selected]
        command_value = softmax(tensor_results)[index_selected]
        if command_value >= threshold_commands:
            return command
        return '' 