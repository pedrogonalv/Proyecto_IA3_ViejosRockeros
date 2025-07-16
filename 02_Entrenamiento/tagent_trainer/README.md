# Tagent Trainer

Este proyecto está diseñado para ejecutarse en RunPod.  
La carpeta del proyecto debe subirse al directorio 'workspace' o similar en RunPod.  
Se aconseja el uso de ssh para evitar problemas de conexión con el pod.
Se debe incluir almacenamiento suficiente para poder entrenar el modelo y almacenar ficheros temporales.
Todos los procesos interactuan con hugging-face, descargando y subiendo modelos. Es necesario disponer de una clave Hugging Face para completar el proceso.

## Estructura del Proyecto

- **project**: Contiene scripts y configuraciones necesarias para configurar y ejecutar el proyecto en RunPod.
  - **requirements.txt**: Fichero para ajustar python.
  - **setup_pod.sh**: Un script para actualizar el pod ,instalar `llama.cpp` y configurar el entorno python.
  - **training**: Este subdirectorio incluye scripts para la generación de datos, ajuste fino y fusión de adaptadores con el modelo base.
    - **generate_react_ft-data.py**: Script para generar datos artificiales en formato React.
    - **fine_tuning.py**: Script para realizar el ajuste fino del modelo.
    - **fusion.py**: Script para fusionar adaptadores con el modelo base.
    - **config.py**: Debe ajustarse con los datos relevantes del modelo antes de la ejecución.
  - **data**: Subdirectorio donde se encuentran todos los conjuntos de datos.
  - **convert**: Contiene el script para la conversión al formato GGUF y cuantización Q4.
    - **convert_quantize.py**: Script para conversión y cuantización.
    - **config.py**: Necesita ser configurado con los datos pertinentes para la conversión.

## Instrucciones de Uso.

1. Sube la carpeta `project` al directorio 'workspace' en RunPod.
2. Ajusta los archivos `config.py` en los subdirectorios `training` y `convert` con los datos apropiados del modelo.
3. Usa el script `setup_pod.sh` para preparar el pod.
4. Sigue los scripts en el subdirectorio `training` para generar datos, en el siguiente orden:  
   4.1. `generate_react_ft-data.py`
   4.2. `fine_tuning.py`
   4.3. `fusion.py`
5. Utiliza el script `convert_quantize.py` del subdirectorio `convert` para la conversión y cuantización del modelo.

Todas las operaciones generan archivos de registro, en directorios `log` que aparecerán durante el proceso.

Se recomienda descargar del pod los archivos de registro antes de abandonarlo.

## Licencia

Este proyecto está licenciado bajo la Licencia Pública General de GNU (GPL). Para más detalles, consulta el archivo LICENSE incluido en este repositorio.
