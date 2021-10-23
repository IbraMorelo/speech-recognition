import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from sklearn.metrics import classification_report
from IPython import display
from scipy import signal
from tensorflow import lite

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

data_dir = pathlib.Path('/Users/ibrahimmemorelo/U/proyecto_de_grado/repos/speech-recognition-data-set/users/audios_augmentation')

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != '.DS_Store']

print('Commands:', commands)

file_name = 'commands.txt'

if os.path.exists(file_name):
  os.remove(file_name)

tmp = ""
for c in commands:
  tmp += c + "\n"

f = open(file_name, 'a+')
f.write(tmp)
f.close()

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
print('Number of total examples:', num_samples)
print('Number of examples per label:',
      len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
print('Example file tensor:', filenames[0])

import math
train_amount = math.ceil((num_samples*0.8)) 
val_test_files = math.ceil((num_samples*0.1)) 
train_files = filenames[:train_amount]
val_files = filenames[train_amount: train_amount + val_test_files]
test_files = filenames[-val_test_files:]

print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  return parts[-2]

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

# rows = 2
# cols = 2
# n = rows*cols
# fig, axes = plt.subplots(rows, cols, figsize=(5, 6))
# for i, (audio, label) in enumerate(waveform_ds.take(n)):
#   r = i // cols
#   c = i % cols
#   ax = axes[r][c]
#   ax.plot(audio.numpy())
#   ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
#   label = label.numpy().decode('utf-8')
#   ax.set_title(label)

# plt.show()

def stft(x):
    f, t, spec = signal.stft(x.numpy(), fs=16000, nperseg=255, noverlap = 124, nfft=256)
    return tf.convert_to_tensor(np.abs(spec))
    
def get_spectrogram(waveform):
  zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)

  spectrogram = tf.py_function(func=stft, inp=[equal_length], Tout=tf.float32)
       
  spectrogram.set_shape((129, 124))

  return spectrogram

def plot_spectrogram(spectrogram, ax):
  log_spec = np.log(spectrogram)
  height = log_spec.shape[0]
  X = np.arange(16000, step=height + 1)
  Y = range(height)
  print(X.shape, Y, log_spec.shape)
  ax.pcolormesh(X, Y, log_spec)


def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  spectrogram = tf.expand_dims(spectrogram, -1)
  label_id = tf.argmax(tf.cast(label == commands, "uint32"))
  return spectrogram, label_id

spectrogram_ds = waveform_ds.map(
    get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

# rows = 2
# cols = 2
# n = rows*cols
# fig, axes = plt.subplots(rows, cols, figsize=(6, 6))
# for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
#   r = i // cols
#   c = i % cols
#   ax = axes[r][c]
#   plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
#   ax.set_title(commands[label_id.numpy()])
#   ax.axis('off')
# plt.show()

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
  return output_ds

train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

batch_size = 64
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

for spectrogram, _ in spectrogram_ds.take(1):
  input_shape = spectrogram.shape
print('Input shape:', input_shape)
num_labels = len(commands)

norm_layer = preprocessing.Normalization()
norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

model = models.Sequential([
    layers.Input(shape=input_shape),
    preprocessing.Resizing(32, 32), 
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 10
history = model.fit(
    train_ds, 
    validation_data=val_ds,  
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

test_audio = []
test_labels = []

for audio, label in test_ds:
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

report_dictionary = classification_report(y_true, 
                                          y_pred, 
                                          output_dict = False)
print(report_dictionary)              

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands, 
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

commands = np.empty(0)
with open('commands.txt', 'r') as f:
    new = f.readlines()
    commands = []
    for w in new:
        k = w.replace('\n', '')
        commands.append(k)
print(commands)

model_filename_tflite = 'model_commands_recognition.tflite'

converter = lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
tflite_models_dir = pathlib.Path("/Users/ibrahimmemorelo/U/proyecto_de_grado/repos/speech-recognition/models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/model_filename_tflite
tflite_model_file.write_bytes(tflite_model)