# Sistema RAG para Documentación Técnica

Sistema de Recuperación Aumentada por Generación (RAG) especializado en el procesamiento inteligente de manuales técnicos en PDF, con capacidades de extracción multimodal, búsqueda semántica y generación automática de datasets Q&A.

## 🚀 Características Principales

### Procesamiento de Documentos
- **Procesamiento Adaptativo**: Detecta automáticamente el tipo de documento (técnico, texto, escaneado, mixto) y aplica la estrategia óptima
- **Extracción Multimodal**: 
  - Texto con chunking inteligente y preservación de contexto
  - Tablas con preservación de estructura (exportación a CSV)
  - Imágenes raster embebidas con metadatos
  - Diagramas técnicos renderizados de alta calidad
  - OCR integrado para documentos escaneados (Tesseract)
- **Organización Inteligente**: Estructura de carpetas automática por manual con separación por tipo de contenido

### Almacenamiento y Búsqueda
- **Almacenamiento Híbrido**: 
  - SQLite para metadatos estructurados y búsqueda rápida
  - ChromaDB para embeddings vectoriales
  - Sistema de archivos para contenido binario (imágenes, CSVs)
- **Búsqueda Híbrida**: Combina búsqueda vectorial semántica con búsqueda por palabras clave (FTS5)
- **Cache Inteligente**: LMDB para embeddings y resultados de búsqueda frecuentes

### Generación de Datasets Q&A
- **Generación Automática**: Crea pares pregunta-respuesta de alta calidad desde chunks procesados
- **Múltiples Tipos de Preguntas**: Factual, Síntesis, Causal, Aplicación, Análisis
- **Control de Calidad**: Validación automática y filtrado de relevancia
- **Procesamiento Masivo**: Manejo eficiente con rate limiting y reanudación automática

### Soporte Multiidioma
- Optimizado para documentación técnica en **español** e **inglés**
- Modelo de embeddings multilingüe de alto rendimiento

## 📋 Requisitos del Sistema

### Hardware
- **CPU**: 4+ cores recomendado
- **RAM**: 8GB mínimo (16GB recomendado para procesamiento masivo)
- **Disco**: 10GB+ espacio libre para procesamiento y almacenamiento
- **GPU**: Opcional, mejora el rendimiento de embeddings

### Software
- **Python**: 3.8+ (probado con 3.12)
- **Sistema Operativo**: macOS, Linux, Windows

### Dependencias del Sistema

#### Tesseract OCR (opcional, para documentos escaneados)
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-spa

# Windows
# Descargar desde: https://github.com/UB-Mannheim/tesseract/wiki
```

#### Java (opcional, para extracción avanzada de tablas con Tabula)
```bash
# Verificar instalación
java -version
```

## 🛠️ Instalación

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd clode_technical_rag_system
```

### 2. Crear y activar entorno virtual
```bash
python -m venv venv_rag_clean
source venv_rag_clean/bin/activate  # En Windows: venv_rag_clean\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Inicializar el sistema
```bash
python scripts/init_system.py
```

Este script automáticamente:
- ✓ Verifica versión de Python y todas las dependencias
- ✓ Detecta dependencias del sistema (Tesseract, Java)
- ✓ Crea la estructura completa de directorios
- ✓ Inicializa la base de datos SQLite con esquema actualizado
- ✓ Genera un reporte detallado de inicialización

### 5. Configurar variables de entorno (opcional)
```bash
cp .env.example .env
# Editar .env según necesidades
```

Variables disponibles:
- `OPENAI_API_KEY`: Para generación de Q&A (opcional)
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR
- `MAX_WORKERS`: Número de workers paralelos
- `BATCH_SIZE`: Tamaño de lote para procesamiento

## 📚 Uso del Sistema

### 1. Procesamiento de Manuales PDF

#### Procesamiento básico
```bash
# Procesar todos los PDFs en el directorio por defecto
python scripts/process_manuals_sqlite.py --pdf-dir data/raw_pdfs/

# Procesar un solo PDF
python scripts/process_manuals_sqlite.py --single-pdf manual.pdf

# Procesar con generación de embeddings
python scripts/process_manuals_sqlite.py --pdf-dir data/raw_pdfs/ --embeddings
```

#### Procesamiento con metadatos específicos
```bash
python scripts/process_manuals_sqlite.py --single-pdf manual.pdf \
  --manufacturer "Beckhoff" \
  --model "AX5000" \
  --doc-type technical \
  --embeddings
```

#### Reprocesamiento selectivo
```bash
# Reprocesar solo imágenes y tablas para un manual
python scripts/process_manuals_sqlite.py --reprocess 1 --steps images tables

# Reprocesar todo para un manual específico
python scripts/process_manuals_sqlite.py --reprocess 1 \
  --steps analysis chunks images tables embeddings
```

### 2. Construcción de Base de Datos Vectorial

```bash
# Construir base vectorial desde SQLite
python scripts/build_vectordb_sqlite.py

# Forzar reconstrucción completa
python scripts/build_vectordb_sqlite.py --force

# Procesar solo un manual específico
python scripts/build_vectordb_sqlite.py --manual-id 1

# Ver estadísticas actuales
python scripts/build_vectordb_sqlite.py --stats

# Verificar sincronización SQLite ↔ ChromaDB
python scripts/build_vectordb_sqlite.py --verify
```

### 3. Extracción de Contenido Visual

```bash
# Extraer y organizar todo el contenido visual
python scripts/extract_with_manual_folders.py

# Procesar diagramas técnicos específicamente
python scripts/process_technical_diagrams.py data/raw_pdfs/ --batch
```

### 4. Generación de Dataset Q&A

```bash
# Generar dataset completo (requiere OPENAI_API_KEY)
cd qa_generator
python process_all_chunks_v4.py \
    --model gpt-3.5-turbo \
    --batch-size 3 \
    --rps 2 \
    --output-file qa_dataset.jsonl
```

### 5. Verificación del Sistema

```bash
# Verificar integridad del sistema
python scripts/verify_system.py

# Test de extracción de metadatos
python scripts/test_metadata_extraction.py
```

## 📁 Estructura del Proyecto

```
clode_technical_rag_system/
├── config/                      # Configuración del sistema
│   └── settings.py             # Configuración centralizada
├── core/                       # Componentes principales
│   ├── embedding_pipeline.py   # Pipeline de generación de embeddings
│   ├── hybrid_search.py        # Búsqueda híbrida (vectorial + keyword)
│   ├── intelligent_chunking.py # Chunking semántico inteligente
│   └── unified_architecture.py # Arquitectura unificada del sistema
├── data/                       # Almacenamiento de datos
│   ├── raw_pdfs/              # PDFs originales a procesar
│   ├── processed/             # Contenido procesado por manual
│   │   └── [Manual_Name]/     
│   │       ├── images/        # Imágenes raster extraídas
│   │       ├── diagrams/      # Diagramas técnicos renderizados
│   │       └── tables/        # Tablas en formato CSV
│   ├── cache/                 # Sistemas de cache
│   │   ├── embedding_cache/   # Cache LMDB de embeddings
│   │   └── search_cache/      # Cache de resultados de búsqueda
│   ├── sqlite/                # Base de datos SQLite
│   │   └── manuals.db        # BD principal con toda la metadata
│   ├── vectordb/             # Base de datos vectorial ChromaDB
│   └── logs/                 # Logs del sistema
│       ├── processing/       # Logs detallados por manual
│       └── initialization_report.json
├── database/                  # Gestión de base de datos
│   ├── schema.sql            # Esquema SQLite actualizado
│   └── sqlite_manager.py     # Manager de operaciones SQLite
├── extractors/               # Extractores de contenido
│   ├── adaptive_processor.py # Procesador adaptativo por tipo
│   ├── document_analyzer.py  # Analizador de tipo de documento
│   ├── pdf_extractor.py      # Extractor de texto (PyMuPDF)
│   ├── table_extractor.py    # Extractor de tablas (pdfplumber)
│   ├── enhanced_image_extractor.py # Extractor de imágenes con OCR
│   └── sqlite_extractors.py  # Extractores integrados con SQLite
├── models/                   # Modelos y embeddings
│   └── embeddings.py         # Gestión de modelos de embeddings
├── qa_generator/             # Generador de datasets Q&A
│   ├── qa_generator.py       # Generador principal
│   ├── process_all_chunks_v4.py # Procesamiento masivo
│   ├── chunk_manager.py      # Gestión de chunks
│   ├── prompt_templates.py   # Templates de preguntas
│   ├── quality_evaluator.py  # Evaluador de calidad
│   └── qa_dataset/          # Datasets generados
├── scripts/                  # Scripts ejecutables principales
│   ├── init_system.py        # Inicialización del sistema
│   ├── verify_system.py      # Verificación de integridad
│   ├── process_manuals_sqlite.py # Procesamiento principal
│   ├── build_vectordb_sqlite.py  # Construcción de vectorDB
│   ├── extract_with_manual_folders.py # Extracción organizada
│   ├── process_technical_diagrams.py  # Procesamiento de diagramas
│   ├── migrate_to_sqlite.py  # Migración de datos legacy
│   └── generate_qa_dataset.py # Generación de Q&A
├── utils/                    # Utilidades generales
├── vectorstore/              # Gestión de almacenamiento vectorial
│   ├── vector_manager.py     # Manager principal de vectores
│   ├── sqlite_adapter.py     # Adaptador SQLite-ChromaDB
│   ├── indexing.py          # Estrategias de indexación
│   └── retrieval.py         # Mecanismos de recuperación
├── requirements.txt         # Dependencias Python
├── .env.example            # Ejemplo de configuración
├── .gitignore              # Archivos ignorados por git
├── CLAUDE.md               # Guía para Claude Code
└── README.md               # Este archivo
```

## 🗄️ Base de Datos SQLite

### Esquema Principal

#### Tabla `documents`
- Información completa de manuales procesados
- Metadatos extraídos del nombre de archivo
- Versionado y timestamps

#### Tabla `content_chunks`
- Fragmentos de texto procesados (~512 caracteres)
- Referencias a página y posición
- Keywords y entidades extraídas
- Embeddings opcionales

#### Tabla `visual_content`
- Metadatos de imágenes y diagramas
- Hash MD5 para deduplicación
- Dimensiones y tipo de contenido
- Rutas de archivos en filesystem

#### Tabla `structured_tables`
- Metadatos de tablas extraídas
- Estructura y headers
- Rutas a archivos CSV
- Estadísticas de contenido

#### Tabla `document_analysis`
- Análisis automático de tipo de documento
- Métricas de contenido
- Recomendaciones de procesamiento

#### Índices y Optimizaciones
- Índices en campos de búsqueda frecuente
- FTS5 para búsqueda de texto completo
- Triggers para actualización automática

## ⚙️ Configuración Avanzada

### Ajuste de Parámetros de Chunking

En `config/settings.py`:

```python
# Tamaños adaptativos por tipo de documento
ADAPTIVE_CHUNK_SIZES = {
    'technical_diagram_heavy': 1024,  # Más contexto para diagramas
    'text_heavy': 512,                # Estándar para texto
    'table_heavy': 768,               # Balance para tablas
    'scanned': 768,                   # Más contexto para OCR
    'mixed': 512                      # Por defecto
}

# Configuración base
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MIN_CHUNK_SIZE = 200  # Chunks menores se descartan
```

### Configuración de Modelos de Embeddings

```python
# Modelo multilingüe optimizado para español/inglés
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384

# Alternativas disponibles:
# - "all-MiniLM-L6-v2": Más rápido, solo inglés
# - "distiluse-base-multilingual-cased-v2": Más preciso, más lento
```

### Optimización de Rendimiento

```python
# Procesamiento paralelo
MAX_WORKERS = 4  # Ajustar según CPU disponible
BATCH_SIZE = 32  # Para generación de embeddings

# Límites de memoria
MAX_IMAGE_SIZE = (3000, 3000)  # Redimensionar imágenes grandes
MAX_TABLE_ROWS = 1000  # Límite para tablas muy grandes

# Cache
CACHE_TTL = 86400  # 24 horas para cache de búsqueda
EMBEDDING_CACHE_SIZE = 10000  # Número máximo de embeddings en cache
```

## 📊 Estadísticas y Monitoreo

### Ejemplo de Procesamiento

Para 3 manuales técnicos típicos:

```
RESUMEN DE PROCESAMIENTO POR LOTES
============================================================
Total de manuales: 3
Exitosos: 3 (100.0%)

Estadísticas agregadas:
  - Páginas procesadas: 892
  - Chunks generados: 4,558
  - Imágenes extraídas: 743
  - Diagramas renderizados: 868
  - Tablas procesadas: 1,244
  
Tiempo total: 48 minutos
Velocidad promedio: 18.5 páginas/minuto
Tokens estimados: ~630,000
```

### Monitoreo en Tiempo Real

```bash
# Ver progreso de procesamiento
tail -f data/logs/processing/manual_*.json | jq '.'

# Estadísticas de base de datos
sqlite3 data/sqlite/manuals.db "
  SELECT 
    COUNT(DISTINCT document_id) as manuals,
    COUNT(*) as total_chunks,
    AVG(LENGTH(content)) as avg_chunk_size
  FROM content_chunks;
"

# Estado de vectorDB
python scripts/build_vectordb_sqlite.py --stats
```

## 🔍 API de Búsqueda (Ejemplo)

```python
from vectorstore.retrieval import AdvancedRetrieval
from config.settings import get_config

# Inicializar sistema de búsqueda
config = get_config()
retrieval = AdvancedRetrieval(config)

# Búsqueda híbrida
results = retrieval.hybrid_search(
    query="conexión servo drive AX5000",
    k=10,
    filters={
        "manufacturer": "Beckhoff",
        "doc_type": "technical"
    },
    search_type="hybrid"  # vectorial + keyword
)

# Procesar resultados
for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Manual: {result['metadata']['manual_name']}")
    print(f"Página: {result['metadata']['page_num']}")
    print(f"Contenido: {result['content'][:200]}...")
    print("-" * 50)
```

## 🐛 Solución de Problemas

### Error: "UNIQUE constraint failed"
```bash
# Usar opción de reprocesamiento para actualizar
python scripts/process_manuals_sqlite.py --reprocess [manual_id] --steps [pasos]
```

### Error: "No module named 'jpype'"
- Es una advertencia de Tabula, no afecta el funcionamiento
- Instalar jpype1 si se requiere mejor rendimiento: `pip install jpype1`

### Error de memoria con PDFs grandes
```python
# En config/settings.py reducir:
BATCH_SIZE = 16  # De 32 a 16
MAX_WORKERS = 2  # De 4 a 2
PAGE_BATCH_SIZE = 10  # Procesar menos páginas a la vez
```

### Imágenes CMYK no se visualizan
- Limitación conocida de algunas librerías
- El sistema las detecta y omite automáticamente
- Los diagramas renderizados compensan esta limitación

### Rate limiting en generación Q&A
```bash
# Ajustar parámetros de rate limiting
python qa_generator/process_all_chunks_v4.py \
    --rps 1 \              # Reducir requests por segundo
    --batch-size 2 \       # Reducir tamaño de lote
    --model gpt-3.5-turbo  # Usar modelo con límites más altos
```

## 📈 Rendimiento y Optimización

### Cache de Embeddings
- **Tecnología**: LMDB (Lightning Memory-Mapped Database)
- **Ubicación**: `data/cache/embedding_cache/`
- **Beneficio**: Reduce tiempo de reprocesamiento ~80%
- **Tamaño típico**: 500MB-2GB según volumen

### Procesamiento Paralelo
- Extracción de texto: Hasta 4x más rápido
- Generación de embeddings: Procesamiento por lotes
- Renderizado de diagramas: Pool de workers

### Almacenamiento Eficiente
- **SQLite**: Consultas de metadatos en <10ms
- **ChromaDB**: Búsqueda vectorial optimizada con HNSW
- **Filesystem**: Acceso directo a binarios grandes

## 🚦 Roadmap

### En Desarrollo
- [ ] API REST completa para integración
- [ ] Interfaz web de búsqueda y visualización
- [ ] Soporte para más formatos (DOCX, HTML, EPUB)
- [ ] Pipeline de fine-tuning para modelos específicos

### Próximas Características
- [ ] Integración con LLMs locales (Ollama, llama.cpp)
- [ ] Dashboard de analytics y métricas
- [ ] Exportación de conocimiento a formatos estándar
- [ ] Versionado automático de documentos
- [ ] Clustering semántico de contenido
- [ ] Detección de cambios entre versiones

## 🤝 Contribuir

1. Fork el repositorio
2. Crear rama para tu feature (`git checkout -b feature/NuevaCaracteristica`)
3. Realizar cambios y tests
4. Commit con mensaje descriptivo (`git commit -m 'Add: Nueva característica X'`)
5. Push a la rama (`git push origin feature/NuevaCaracteristica`)
6. Crear Pull Request con descripción detallada

### Guías de Contribución
- Seguir PEP 8 para estilo de código Python
- Agregar docstrings a todas las funciones públicas
- Incluir tests para nuevas características
- Actualizar documentación relevante

## 📝 Licencia

Este proyecto está licenciado bajo [especificar licencia].

## 💬 Soporte

### Canales de Soporte
- **Issues**: Reportar bugs o solicitar características en GitHub
- **Documentación**: Wiki del proyecto para guías detalladas
- **Ejemplos**: Carpeta `examples/` con casos de uso

### Preguntas Frecuentes
- **¿Soporta PDFs protegidos?** No, deben estar desprotegidos
- **¿Límite de tamaño de PDF?** Recomendado <100MB por archivo
- **¿Puedo usar mis propios embeddings?** Sí, configurable en settings.py

## 🙏 Agradecimientos

- **PyMuPDF** - Extracción robusta de contenido PDF
- **ChromaDB** - Base de datos vectorial de alto rendimiento  
- **Sentence Transformers** - Modelos de embeddings multilingües
- **Tabula-py / pdfplumber** - Extracción precisa de tablas
- **Tesseract OCR** - Reconocimiento óptico de caracteres
- **LangChain** - Framework para aplicaciones LLM
- **OpenAI** - APIs para generación de Q&A

---

*Desarrollado para el procesamiento eficiente de documentación técnica industrial con enfoque en manuales de automatización y control.*