# PROYECTO FINAL. BOOTCAMP IA3.
Resolución del **proyecto final**, bootcamp **IA3**.  

Realizado por el grupo **Viejos Rockeros**,compuesto por:  

- Pedro González Álvarez.
- Santiago Jorda Ramos.


## Estructura del proyecto.

El proyecto está estructurado según las diferentes etapas del ciclo de vida del mismo.  
En el directorio raiz se encuentra la **memoria del proyecto**, *Memoria.md*. También se encuentra la **presentación**, *Presentacion.pptx*.

A parte se encuentran una serie de subcarpetas, correspondiendo cada una a una etapa del ciclo de vida del sistema. Dentro de cada una de las carpetas se encuentra otro fichero `readme.md` con información más detalladas. 

De ese modo se pueden encontrar las carpeats correspodientes a las siguientes etapas:

### 1. Preprocesamiento:  

Contiene la lógica empleada durante la etapa de preprocesamiento de la información.  
En esta etapa se genera información destinada a:  
  - **Generación de registros para entrenamiento del modelo**. Generación de los datos empleados para fine-tuning del modelo base.  
  - **Generación de embeddings y metadatos para el RAG**. Generación de los datos que alimentan el sistema RAG.  

### 2. Entrenamiento:
Esta etapa puede requerir el empleo de un hardware dotado de gran capacidad de memoria GPU.  Es por ello que está diseñado para emplearse en algún servicio cloud, en concreto `runpod`.  

En esta etapa se realizan las siguientes acciones:  
  - **Fine-tuning del modelo**. Entrenamiento de un modelo base por *qlora*. Generación de **adaptador**.   
  - **Fusión del modelo**. Fusión del modelo base y adaptador.  
  - **Conversión a GGUF**. Conversión del modelo fusionado en formato *gguf* para optimizar la inferencia en local.
  - **Cuantización**. Se cuantiza a **4 bits** para permitir inferencia en equipos con hardware limitado.

### 3. Ejecución:  
Se refiere a los recursos necesarios para **instalar** y **ejecutar** el sistema agentico *gtagent* por parte del usuario.  




