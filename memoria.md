
# Memoria de proyecto final. Bootcamp IA3

**Autores:** Pedro González Álvarez y Santiago Jorda Ramos

## Índice

1. Introducción
2. Objetivos del proyecto
3. Ciclo de vida y uso del sistema
4. Desarrollo del sistema
   - 4.1 Preprocesado
   - 4.2 Entrenamiento
   - 4.3 Ejecución
5. Resultados obtenidos
6. Conclusiones y trabajo futuro

## 1. Introducción

Este proyecto se centra en el diseño y desarrollo de un sistema asistente destinado a técnicos especializados que, en momentos críticos, necesiten acceder a documentación técnica específica.

El sistema está pensado para operar en contextos donde se cumplen dos condiciones fundamentales: (1) la bibliografía requerida por el técnico está claramente definida de antemano y (2) existe la posibilidad de no contar con acceso a red en el momento en que se necesita la información.

En particular, el proyecto toma como referencia el contexto de las plantas industriales. Este sector presenta desafíos particulares en cuanto a conectividad, debido tanto a limitaciones de infraestructura como a políticas estrictas de seguridad informática. Además, el uso de tecnologías inalámbricas en estos entornos suele ser problemático por las interferencias que se producen en planta.

Por otro lado, las fábricas suelen contar con un inventario bien definido de maquinaria, equipos y componentes. Esto permite que la bibliografía necesaria para los técnicos de mantenimiento e ingenieros esté predefinida y solo experimente cambios menores con la incorporación de nuevos elementos.

Este escenario convierte al entorno industrial en un caso ideal para la implementación de un sistema de información local, que funcione de forma autónoma sin requerir conexión a internet. El presente proyecto propone una solución que explota esta ventaja, proporcionando acceso inmediato y eficiente a la documentación técnica relevante en el momento preciso.

## 2. Objetivos del proyecto

Los objetivos principales del proyecto son los siguientes:

Desarrollar un sistema basado en inteligencia artificial que permita realizar inferencias de forma local, utilizando hardware con recursos limitados. En concreto, el sistema debe ser capaz de operar con un máximo de 6GB de memoria GPU.

Permitir al usuario realizar consultas sobre cualquier tema contenido en la bibliografía incorporada al sistema. Dado que dicha bibliografía puede ser heterogénea, se proporcionará una funcionalidad de filtrado que permita seleccionar los títulos o documentos de interés en cada momento.

Implementar un sistema robusto de registro de información de ejecución, con el objetivo de facilitar la depuración del sistema. Esta funcionalidad es clave dado que el entorno de ejecución será local y no se dispondrá de otras vías para recoger retroalimentación de los usuarios finales.

Automatizar, en la medida de lo posible, las herramientas necesarias para la generación de datos y la preparación del modelo. Esto permitirá realizar iteraciones rápidas en la actualización del sistema, facilitando su mantenimiento y mejora continua.

Utilizar Hugging Face como repositorio central para los modelos de base y los modelos entrenados. Esta elección facilita tanto el entrenamiento como la distribución eficiente del sistema y sus futuras actualizaciones.

Definir claramente el alcance del proyecto, con las siguientes limitaciones:

Idioma: El sistema manejará exclusivamente el idioma inglés. Tanto los manuales técnicos como la interacción del usuario estarán en este idioma, seleccionado para garantizar una mayor disponibilidad de documentación.

Interfaz: La interacción con el sistema será exclusivamente textual. No obstante, se permitirá el procesamiento de imágenes durante la fase de preprocesamiento de la documentación.

Realizar el desarrollo del sistema íntegramente en Python, con el objetivo de favorecer su ejecución en distintas plataformas (Linux, macOS, Windows), aunque las pruebas se hayan limitado a entornos Linux.

Considerar que el cliente no dispone de infraestructura para el entrenamiento del modelo. Por tanto, el proceso de entrenamiento se llevará a cabo en servicios de computación en la nube.


## 3. Ciclo de vida y uso del sistema

El sistema propuesto se comercializaría en forma de servicio orientado a la preparación y mantenimiento de un agente especializado. El cliente define la especialidad concreta del perfil técnico al que va dirigido el sistema y proporciona la bibliografía disponible relacionada con la maquinaria y los equipos industriales presentes en planta. Esta información suele estar alineada con el inventario de instalaciones existente.

A partir de este punto comienza la etapa de preprocesamiento. Se distingue entre documentación general de la especialidad y documentación particular de máquinas o equipos específicos. Se intentará incrementar la bibliografía proporcionada por el cliente con documentación general adicional que permita mejorar el comportamiento del agente.

Ambos tipos de información (general y particular) son procesados con la ayuda de un modelo LLM auxiliar para generar preguntas y respuestas que se utilizarán en la fase de fine-tuning del modelo. Además, la información particular también se emplea para generar la base de datos vectorial que será utilizada por el sistema RAG (Retrieval-Augmented Generation).

Una vez finalizada la etapa de preprocesamiento, se pasa al entrenamiento. Este se lleva a cabo en servicios en la nube tipo RunPod, permitiendo escalar los recursos de hardware según la cantidad de información disponible. En esta fase se realiza el fine-tuning del modelo base y las conversiones necesarias para que el modelo resultante pueda ejecutarse en entornos locales.

Tras el entrenamiento, el sistema está listo para su uso. El modelo estará disponible a través de Hugging Face, desde donde podrá ser descargado por el cliente para su instalación local. La aplicación del agente, junto con la base de datos vectorial RAG, se instala en el equipo del usuario, permitiendo la ejecución del sistema bajo demanda.

Un aspecto crucial en el ciclo de vida del sistema es el mantenimiento. En cada equipo se generarán archivos de registro con las interacciones del usuario con el agente. La recuperación de estos archivos es fundamental para realizar iteraciones de mejora, como ajustes en los prompts o actualizaciones en la bibliografía, lo que podrá conllevar una nueva versión del modelo adaptado a las necesidades cambiantes del cliente.

## 4. Desarrollo del sistema

### 4.1 Preprocesado

*El sistema de preprocesado constituye el componente fundamental para la transformación de documentación técnica en conocimiento estructurado. Esta etapa implementa un pipeline modular que procesa documentos PDF heterogéneos, extrayendo y estructurando su contenido para las fases posteriores de entrenamiento y ejecución.
El proceso inicia con la validación y análisis adaptativo de cada documento PDF. Un analizador especializado examina las características estructurales del manual para determinar la estrategia de procesamiento óptima, identificando la presencia de texto, tablas, imágenes y otros elementos multimodales. Esta clasificación permite la activación selectiva de extractores especializados según el tipo de contenido detectado.
La extracción de contenido se realiza mediante múltiples módulos que operan de forma coordinada. El extractor de texto preserva la estructura jerárquica del documento, manteniendo información sobre formato y posición. Para documentos escaneados o con contenido no procesable directamente, se activa el módulo OCR. Las tablas complejas se procesan mediante algoritmos especializados que las convierten en formatos CSV estructurados. Las imágenes técnicas y diagramas se extraen y clasifican para su posterior procesamiento.
Una fase crítica del preprocesado es la segmentación inteligente del contenido textual. Los documentos se dividen en fragmentos semánticamente coherentes mediante técnicas de chunking que consideran la longitud óptima, la coherencia temática y la preservación del contexto. Esta segmentación es esencial para la calidad de las operaciones posteriores de búsqueda y generación.
El módulo de generación de preguntas y respuestas transforma el contenido procesado en datasets de entrenamiento. Implementa cinco estrategias diferenciadas (factual, síntesis, causal, aplicación y análisis) que generan pares de pregunta-respuesta diversos y representativos del dominio técnico. Cada par generado pasa por un riguroso proceso de validación de calidad antes de incluirse en el dataset final en formato JSONL.
El almacenamiento utiliza un modelo híbrido: SQLite para metadatos estructurados, ChromaDB para búsquedas vectoriales mediante embeddings, y sistema de archivos para contenido binario como imágenes y archivos CSV. Esta arquitectura optimiza cada tipo de operación mientras mantiene la coherencia entre los diferentes sistemas.
Durante todo el proceso se implementan puntos de control y validación que aseguran la calidad del procesamiento. Se generan logs detallados que permiten el seguimiento y depuración de cada etapa, facilitando la identificación de problemas y la optimización continua del sistema.


### 4.2 Entrenamiento

En esta etapa se considera que se parte con un archivo jsonl que contiene los registros de preguntas y respuestas requeridos para el entrenamiento.

El modelo base elegido es el 'Mistral-7B-Instruct-v0.3'. Ello se debe por un lado a que los modelos tipo Mistral son compatibles para su inferencia con ollama y por otro lado se buscó deliberadamente un modelo instruido que favorezca el razonamiento.

Un punto clave en esta etapa es tener definido el tipo de prompt que se empleará para interactuar con el llm. En este caso es un prompt react, disponiendo de una plantilla que defina las diferentes claves.

Gracias a esa plantilla se genera un script generador de datos sintéticos. Ese script toma como entrada las preguntas y respuestas y su salida es una mezcla de preguntas y respuestas originales con otro grupo de preguntas y respuestas en formato react.

Las preguntas y respuestas en formato react están divididas por categorías, simulando diferentes flujos de conversación en función de hipotéticos contextos (consultas simples, consultas corregidas por el usuario, manejo de herramientas...).

El script permite tanto la definición del porcentaje de Q&A que se convierten a formato react como la definición del porcentaje parcial de generación de cada tipo de prompt react.

Con la salida del generador de registros react, se pasa al fine-tuning del modelo. El script de entrenamiento descarga el modelo base desde Hugging Face y lanza el entrenamiento con qlora a 4 bits para moderar el consumo de memoria. El adaptador resultante, a parte de almacenarse localmente, se sube a Hugging Face.

El siguiente paso es la fusión del adaptador con el modelo base. El script que implementa esa etapa también concluye con el modelo fusionado subido en Hugging Face.

El modelo fusionado es convertido a gguf y cuantizado a 4 bits por un último script de conversión. El script termina subiendo el modelo resultante a Hugging Face. Un detalle importante de la conversión a gguf es que durante el proceso, basado en llama.cpp, se intercepta el 'chat template' del modelo base y se sustituye por la plantilla del prompt que efectivamente creará el agente.

Durante todo este proceso de entrenamiento se generan registros de cada etapa (generación de datos sintéticos, entrenamiento, fusión, conversión).


### 4.3 Ejecución

Para permitir la ejecución será necesario disponer del modelo entrenado en Hugging Face y de la base de datos vectorial ('chromadb') con los datos del 'rag' actualizados.

Se supone que en el equipo del usuario se dispone del script de instalación del modelo y de los scripts de la aplicación en sí misma. Dado que la ejecución del llm es en base a 'ollama', el equipo del usuario deberá tener instalada esa aplicación.

Una vez instalado el modelo en ollama, el usuario podrá arrancar la aplicación.

El GUI está basado en 'streamlit'. Se tiene la posibilidad de modificar su aspecto en base a templates. La funcionalidad que ofrece al usuario es:

Acceso a una pequeña guía de uso.

Posibilidad de resetear el historial de la conversación.

Selector de documentos dentro del rag. De este modo el rag devolverá sólo registros generados a partir de alguno de los documentos seleccionados por el usuario.

Historial de conversación.

Prompt de entrada de texto.

Con cada ejecución de la aplicación se genera un registro sobre la operación del agente y la interacción con el llm, siendo configurable el nivel de detalle. De ese modo se almacenan de forma ordenada temporalmente todas las interacciones del usuario con el sistema, incluyendo la traza de la evolución del agente, la intervención del llm y el uso de las herramientas.

## 5. Resultados obtenidos

En el desarrollo del sistema se han logrado buenos resultados en varios aspectos, aunque también se han identificado limitaciones derivadas principalmente de la falta de tiempo.

Uno de los logros más destacados es la creación de un conjunto de herramientas compacto que facilita tanto la ejecución local por parte del usuario final como la iteración en tareas de mantenimiento. El sistema proporciona suficiente información de depuración como para identificar con rapidez cualquier incidencia y facilitar su resolución.

La interfaz de la aplicación funciona correctamente y la operación general del sistema agente es estable en un equipo con 6GB de memoria GPU. Desde el punto de vista del agente, el sistema RAG se comporta correctamente, especialmente en lo relativo al filtrado de registros por título de documento, ofreciendo un flujo de ejecución consistente y eficaz.

No obstante, también se han identificado varios puntos negativos:

Falta de coordinación entre fases: No se logró una coordinación adecuada entre el procesamiento y la etapa de entrenamiento/aplicación. Esto resultó en la imposibilidad de disponer de un RAG completamente actualizado con el conjunto más amplio de registros generados mediante las técnicas más recientes. Aunque el modelo fue entrenado con registros actualizados, los problemas de formato causaron retrasos que redujeron el tiempo disponible para las pruebas integrales.

Limitaciones del LLM y arquitectura del agente: El modelo LLM no se comporta de manera completamente satisfactoria y la versión de langchain utilizada (0.3.26) no permite ajustar fácilmente el prompt del agente. Se probaron versiones alternativas en las que sí era posible configurar el prompt, pero en dichas versiones surgían otros problemas como comportamientos inestables del agente o fallos de sincronía entre el LLM y el ejecutor. En el momento de la entrega del proyecto, el modelo presenta errores en el uso de herramientas, limitada capacidad de interacción natural con el usuario (enfocándose excesivamente en la extracción de datos) y dificultades para concluir correctamente las iteraciones del agente.


## 6. Conclusiones y trabajo futuro

El proyecto ha alcanzado buenos resultados en diversas áreas, especialmente en lo relativo a la capacidad de depuración, el diseño del interfaz de usuario y la estructura de herramientas para el entrenamiento del modelo. Estos elementos constituyen una base sólida para el desarrollo, mantenimiento y despliegue del sistema en entornos reales.

A pesar de los problemas encontrados, el sistema cuenta con una capacidad funcional razonable y un sistema de registros de ejecución eficaz. Aunque el desarrollo ha estado limitado por el tiempo disponible, el estado actual del proyecto permite prever que futuras iteraciones podrían desarrollarse con mayor agilidad y eficiencia.

Para alcanzar una experiencia de usuario verdaderamente satisfactoria, será necesario abordar dos aspectos clave en futuras fases:

Integración de la nueva estructura de la base de datos vectorial en el agente: Esta integración es necesaria para poder explotar completamente los datos procesados y mejorar la precisión del sistema RAG.

Migración del sistema agente a langraph: La arquitectura actual basada en langchain ha demostrado ser poco flexible para agentes complejos con RAG, herramientas, historial de conversación, scratchpad y prompts configurables. Los cambios en la estructura tienden a provocar errores difíciles de depurar y comportamientos imprevisibles. Además, durante la ejecución se reciben advertencias del propio langchain recomendando la migración a langraph, lo que refuerza esta necesidad.

Una vez migrado, se espera que el sistema permita una mejor integración de funcionalidades y mayor control sobre el comportamiento del LLM, lo que impactaría directamente en la calidad de la interacción con el usuario y la robustez del agente.

A mayor plazo, sería conveniente mejorar los siguientes aspectos:

Preprocesado:

El sistema actual sienta una base sólida para el procesamiento de documentación técnica, pero existen oportunidades significativas de mejora que podrían implementarse en futuras versiones. Una línea prioritaria de desarrollo sería la expansión de las capacidades multimodales del sistema, particularmente en el procesamiento avanzado de diagramas técnicos y esquemas mediante técnicas de visión por computador. Esto permitiría no solo extraer información textual de las imágenes, sino también comprender las relaciones estructurales y funcionales representadas en diagramas de flujo, esquemas eléctricos y planos técnicos. Adicionalmente, la implementación de un sistema de actualización incremental permitiría procesar únicamente las diferencias entre versiones de documentos, optimizando significativamente los recursos computacionales y reduciendo los tiempos de procesamiento en actualizaciones de manuales.

Otra área crítica de mejora sería la evolución hacia una arquitectura distribuida que permita el procesamiento paralelo masivo de documentación, implementando técnicas de orquestación de contenedores para escalar dinámicamente según la demanda. La integración de capacidades de aprendizaje continuo en el módulo de generación de preguntas y respuestas permitiría que el sistema mejore automáticamente la calidad de los datasets generados basándose en retroalimentación de los modelos entrenados. Asimismo, sería valioso desarrollar interfaces de usuario especializadas que permitan a expertos del dominio validar y enriquecer el contenido procesado, creando un ciclo de mejora continua supervisado. La implementación de métricas avanzadas de calidad y dashboards de monitoreo en tiempo real proporcionaría visibilidad completa sobre el rendimiento del sistema y facilitaría la identificación proactiva de áreas de optimización.


Entrenamiento:

Las herramientas se han preparado en previsión de entrenar hasta 50.000 registros. Esta cantidad podría ser insuficiente en un contexto de aplicación real. Sería necesario realizar pruebas adicionales y dimensionar las herramientas para permitir el entrenamiento con una cantidad mayor de registros. Este incremento también implicaría revisar la configuración del pod de entrenamiento y el coste asociado.

Aplicación:

Sería recomendable distribuir la aplicación en formatos como Docker, AppImage u otro similar, permitiendo una instalación sencilla y actualizable. Además, se debería habilitar la actualización independiente tanto del modelo como del RAG, así como la extracción de los archivos de registro.

Asimismo, se debería incluir la funcionalidad para almacenar y recuperar conversaciones del usuario, lo que mejoraría la experiencia de uso y permitiría mayor trazabilidad del sistema.

