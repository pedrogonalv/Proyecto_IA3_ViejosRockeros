# Model Installer

Este proyecto permite la instalación de modelos para ejecutarse con `ollama` en local.

## Requisitos

- Python 3.11
- Hardware para ejecutar el modelo seleccionado.

## Uso

1. **Instalar dependencias**

   Se aconseja emplear un entorno virtual.
   ```bash
   pip install -r requirements.txt
   ```
   
2. **Ajustar parámetros**

   Ajustar los parmátros requeridos en  `config.py`.

3. **Ejecutar el script**

   ```bash
   python install.py
   ```

## Almacenamiento.
El modelo se descargará en el directorio `models_storage`.
Se instalará en `ollama` fijando ese directorio como ruta de almacenamiento del modelo.
Tener en cuenta que si se borra el directorio o el modelo contenido, `ollama` mantendrá la configuración del modelo pero sin existencia del archivo requerido, lo cual generará un error. Deberá desinstalarse el modelo manualmente de `ollama`. 

