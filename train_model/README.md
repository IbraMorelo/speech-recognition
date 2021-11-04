## model_speech_recognition.py

## Prerrequisitos
* Scipy 1.6.2
* Numpy 1.2.0
* sklearn 2.0.0
* Tensorflow 2.6.0

*data_dir* es la ruta de los datos de entranamiento

El entrenamiento arroja dos archivos:
model_commands_recognition.tflite: Modelo que debe ser compilado para ser usado con la Google Coral, se puede usar este [enlace](https://colab.research.google.com/github/google-coral/tutorials/blob/master/compile_for_edgetpu.ipynb) para este proposito.
commands.txt: Listado de comandos soportados por el modelo.
Estos archivos se integran con el módulo [audio_stream](https://github.com/IbraMorelo/speech-recognition/tree/main/audio_stream), para más detalle ver su documentación.

Por último esta red neuronal esta basada en [Tensorflow](https://www.tensorflow.org/tutorials/audio/simple_audio), para más detalle visitar en anterior enlace.
