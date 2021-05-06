import os
import pathlib
import tensorflow as tf
import utils as ut
from tensorflow.keras.models import load_model
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE
data_dir = pathlib.Path('../speech-recognition-data-set/english')
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
sample_file = data_dir/'yes/0c5027de_nohash_0.wav'
sample_ds = ut.preprocess_dataset([str(sample_file)], AUTOTUNE)
nameModelTf = os.path.dirname(os.path.realpath(__file__))+'/model_speech.tflite'

for spectrogram, _ in sample_ds.batch(1):
    interpreter = tf.lite.Interpreter(model_path=nameModelTf)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], spectrogram)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    print('The answer is: ' + commands[np.argmax(tf.nn.softmax(output_data[0]))])