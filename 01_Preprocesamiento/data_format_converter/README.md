# Q&A Dataset Transformation

Este proyecto contiene un script para transformar datasets de preguntas y respuestas entre diferentes formatos.

## Archivos

- **comprehensive_qa_dataset_final.jsonl**: Dataset original con 21,778 entradas en formato conversacional
- **comprehensive_qa_dataset_transformed.jsonl**: Dataset transformado a formato simplificado
- **transform_qa_dataset.py**: Script de transformación

## Formato de Datos

### Formato Original (comprehensive_qa_dataset_final.jsonl)
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "pregunta"},
    {"role": "assistant", "content": "respuesta"}
  ],
  "metadata": {
    "difficulty": "basic",
    "reasoning_type": "factual",
    "source_pdfs": ["documento.pdf"],
    "page_numbers": [123],
    "chunk_preview": "..."
  }
}
```

### Formato Transformado (comprehensive_qa_dataset_transformed.jsonl)
```json
{
  "instruction": "Question: pregunta",
  "output": "Answer: respuesta",
  "context": "vista previa del contenido",
  "doc_name": "documento.pdf",
  "doc_page": "123",
  "type": "unknown"
}
```

## Uso

Para ejecutar la transformación:

```bash
python transform_qa_dataset.py
```

El script procesará el archivo de entrada y generará el archivo transformado automáticamente.