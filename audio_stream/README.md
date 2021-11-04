# Módulo para capturar señales de audio en tiempo real

## Prerrequisitos
* Scipy 1.6.2
* Numpy 1.2.0
* Pycoral 2.0.0
* [Noisereduce](https://github.com/timsainb/noisereduce) 2.0.0

## SpeechRecognition.py

### __init__
* Inicializa la configuración del micrófono
* Inicializa el array de señales de audio

### __enter__
* Abre la recepción de señales de audio

### __exit__
* Detiene la recepción de audio y para el sistema

### enqueue_frames
* Recibe la seña de audio y los añade a la cola en el array frames

### get_record
Consulta la cola de señales de audio, si detecta sonido/ruido por encima del umbral *threshold_volumen* captura el audio hasta completar un muestra para ser clasificada por el modelo. Si detecta muestras incompletas o libres de ruidos limpia la cola para no saturar el arreglo. 

### waveform_reduce_noise

### waveform_downsampling

### waveform_complete_if_needed

### process_audio_data

### get_spectrogram

### run_inference
