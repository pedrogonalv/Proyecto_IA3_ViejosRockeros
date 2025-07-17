# Sistema de Migración pdf_data.db → ChromaDB

## Descripción

Sistema modular y robusto para migrar datos de Q&A desde SQLite a ChromaDB, optimizado para búsquedas semánticas en sistemas RAG.

## Estructura del Proyecto

```
data/
├── source/         # Base de datos fuente
│   └── pdf_data.db # Pares Q&A extraídos de PDFs
└── vector/         # Base de datos vectorial
    ├── chroma.sqlite3      # Persistencia de ChromaDB
    └── chroma_vector_index/ # Archivos de índice HNSW
```

## Características

- ✅ Preserva estructura existente de ChromaDB
- ✅ Procesamiento por lotes con checkpoints
- ✅ Chunking inteligente según tipo de contenido
- ✅ Cache de embeddings con límite LRU
- ✅ Gestión automática de memoria
- ✅ Sistema de logging detallado
- ✅ Validación automática post-migración
- ✅ Recuperación ante fallos con --resume

## Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt

# Para embeddings OpenAI (opcional)
export OPENAI_API_KEY="tu-api-key"
```

## Uso Básico

### 1. Migración Estándar

```bash
python migration_pipeline.py
```

### 2. Migración con Parámetros Personalizados

```bash
python migration_pipeline.py \
    --pdf-db "data/source/pdf_data.db" \
    --chroma-db "data/vector" \
    --batch-size 200 \
    --embedding-model "all-mpnet-base-v2" \
    --memory-limit 2000
```

### 3. Reanudar Migración Interrumpida

```bash
python migration_pipeline.py --resume
```

### 4. Validar Migración

```bash
python test_migration.py
```

## Modelos de Embeddings Disponibles

1. **all-MiniLM-L6-v2** (por defecto)
   - Rápido y eficiente
   - 384 dimensiones
   - Ideal para textos cortos-medianos

2. **all-mpnet-base-v2**
   - Mayor calidad
   - 768 dimensiones
   - Mejor para textos técnicos

3. **openai**
   - Requiere API key
   - 1536 dimensiones
   - Máxima calidad pero con costo

## Estructura de Archivos Generados

```
migration_logs/
├── migration_YYYYMMDD_HHMMSS.log    # Log detallado
├── errors_YYYYMMDD_HHMMSS.log       # Solo errores
├── metrics_YYYYMMDD_HHMMSS.json     # Métricas de proceso
├── final_report_YYYYMMDD_HHMMSS.txt # Reporte final
└── test_results.json                 # Resultados de validación

migration_checkpoints/
└── migration_state.json              # Estado para resume

embeddings_cache.db                   # Cache de embeddings
```

## Monitoreo del Proceso

Durante la migración verás:
- Progreso por fases
- Batches procesados en tiempo real
- Uso de memoria
- Errores (si los hay)

## Solución de Problemas

### Error: "Collection not found"
La collection 'tech_docs' debe existir. El sistema la detectará automáticamente.

### Error: "Out of memory"
Reduce el batch_size o ajusta el límite de memoria:
```bash
python migration_pipeline.py --batch-size 50 --memory-limit 1000
```

### Warning: "No se pudieron cachear embeddings"
Este warning ha sido corregido en la última versión. Si persiste, es seguro ignorarlo ya que no afecta la migración.

### Alto uso de memoria (>1GB)
El sistema ahora incluye gestión automática de memoria:
- Libera memoria entre batches automáticamente
- Ejecuta garbage collection cuando detecta uso alto
- Cache de embeddings con límite de tamaño (LRU)

### Migración muy lenta
Usa el modelo más rápido:
```bash
python migration_pipeline.py --embedding-model "all-MiniLM-L6-v2"
```

## Validación Post-Migración

El script `test_migration.py` valida:
1. **Integridad**: Todos los registros migrados
2. **Búsquedas**: Funcionamiento correcto
3. **Metadatos**: Preservación completa
4. **Filtros**: Queries con where clause
5. **Rendimiento**: Tiempos de respuesta

## Arquitectura del Sistema

```
migration_pipeline.py     # Orquestador principal
├── migration_logger.py   # Sistema de logging
├── db_analyzer.py       # Análisis y extracción SQLite
├── data_processor.py    # Transformación y chunking
└── chroma_manager.py    # Gestión de ChromaDB
```

## Mejores Prácticas

1. **Siempre hacer backup** antes de migrar
2. **Ejecutar validación** después de migrar
3. **Revisar logs** si hay errores
4. **Usar --resume** si se interrumpe
5. **Monitorear memoria** en datasets grandes

## Ejemplo de Uso Completo

```bash
# 1. Backup
cp chroma.sqlite3 chroma_backup_$(date +%s).sqlite3

# 2. Migrar
python migration_pipeline.py --batch-size 100

# 3. Validar
python test_migration.py

# 4. Revisar logs
cat migration_logs/final_report_*.txt
```

## Métricas Esperadas

Para 6,356 registros:
- Tiempo: 60-75 minutos
- Documentos generados: ~9,500
- Uso RAM: 2-4 GB
- Espacio disco: ~500 MB adicional