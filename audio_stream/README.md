# Módulo para capturar señales de audio en tiempo real

## Prerrequisitos
* Scipy 1.6.2
* Numpy 1.2.0
* Pycoral 2.0.0
* [Noisereduce](https://github.com/timsainb/noisereduce) 2.0.0

## SpeechRecognition.py

### Variables
* chunk: Tamaño de las muestras de audios capturas por segundo, entre mayor sea este número, mayor es la latencia del micrófono.
* threshold_volumen: Umbral para detectar actividad de sonido
* threshold_commands: Umbral de clasificación de audios
* format_audio: Formato del audio
* channels: Canal de salida
* input_microphone_rate: Frecuencia de muestreo del micrófono
* resample_rate: Frecuencia de muestreo requerida por tensorflow
* model_path: Ruta del modelo usado para la clasificación
* log_file: Archivo de loggeo
* factor_downsampling: Factor para reducir frecuencia de muestreo 
* commands: Comandos soportado por el modelo

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
Reduce el sonido del ambiente.

### waveform_downsampling
Reduce la frecuencia de muestreo desde 48.000hz hasta aproximadamente 16.000hz, debido a que esta última es la soportada por Tensorflow 

### waveform_complete_if_needed
Si la frecuencia de muestreo es inferior a 16.000 hz, completa el faltante con 0.

### process_audio_data
Agrupa las tres funciones anteriores y retorno una señal de audio lista para ser convertida a espectrograma.

### get_spectrogram
Convierte una señal de audio a espectrograma.

### run_inference
Clasica un espectrograma en el modelo previamente entrenado, retorna el comando a ejecutar 

## TestSpeechRecognition.py
Pruebas unitarias del módulo anterior

## audio_test
Audios usados para pruebas unitarias

## htmlcov
Reporte de pruebas unitarias
