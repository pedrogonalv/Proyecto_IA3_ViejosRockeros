# GTagent. GUI Technical Agent.

Asistente basado en RAG con GUI en Streamlit. Este sistema permite interactuar con un agente inteligente entrenado para consultar documentos técnicos usando recuperación aumentada por generación.
La ejecución es local.

## 🔧 Requisitos

- Python 3.11 o superior
- Ollama instalado y con el modelo disponible para ejecución.
- Virtualenv recomendado
## 📦 Instalación
El único requerimiento es tener el modelo preparado en ollama.
Se deberá crear un entorno virtual e instalar las dependencias:

```
pip install -r requirements.txt
```

## 🚀 Ejecución

```
streamlit run app.py
```

Esto abrirá la GUI en el navegador predeterminado.

## 🗂️ Características

- Interfaz chat con historial de conversación
- Selección dinámica de documentos del RAG para filtrar la búsqueda
- Respuestas del agente condicionadas a los documentos seleccionados
- Logging por sesión con nivel de detalle configurable (niveles 0 a 3)
- Soporte para depuración de errores de herramientas, cadenas y LLM
- Separación clara entre interfaz (GUI) y lógica del sistema agente
- Posibilida de selección de tema.

## 📁 Estructura del proyecto

- `agent/` → lógica del agente, herramientas, ejecución y envoltorios
- `data/` → base de datos vectorial
- `themes/` → definición de estilo de GUI
- `app.py` → GUI en Streamlit
- `config.py` → parámetros configurables del sistema (modelo, RAG, GUI, etc.)
- `main.py` → ejecución en línea de comandos (alternativa a la GUI)
- `requirements.txt` → dependencias
- `sample_questions.txt` → ejemplos de preguntas basadas en los manuales de demostración
- `README.md` → esta guía

Tras ejecución, aparecera el siguiente directorio:
- `logs/` → registros de ejecución del sistema


---

