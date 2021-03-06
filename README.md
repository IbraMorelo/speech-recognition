# Reconocimiento de comandos de voz Offline 

## Prerrequisitos
* Versión de python >= 3.7.5
* Raspberry Pi 4
* Google USB Acelerador
* Micrófono que trabajé a una frecuencia de muestreo mínima de 48.000 hz

## Ambiente virtual
Python ofrece la posibilidad de crear ambientes virtuales para no afectar la configuración global del sistema.
```
python3 -m venv ./virtual-environment
```
```
source ./venv/bin/activate
```

## Instalación de dependencias
```
pip3 install -r requirements.txt 
```

## Módulos
Ingresar a cada carpeta para más detalles.
* [audio_tream](https://github.com/IbraMorelo/speech-recognition/blob/main/audio_stream) = Orientado el reconocimiento de comandos de voz en tiempo real
* [models](https://github.com/IbraMorelo/speech-recognition/tree/main/models) = Modelos machine learning generados 
* [train_model](https://github.com/IbraMorelo/speech-recognition/tree/main/train_model) = Entrenamiento del modelo
* Utils = Utilidaes para cortar (audio_preprocessing.py), convertir (audio_convert.py) y aumentar muestras de audios (data_augmenation.py)

## Nota
El sistema entero esta desarrollado con Python haciendo uso de [TensorFlow](https://www.tensorflow.org/install). 
