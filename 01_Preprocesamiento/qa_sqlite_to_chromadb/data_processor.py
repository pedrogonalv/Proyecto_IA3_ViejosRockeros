"""
Procesador de datos para transformación y chunking de registros Q&A
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
import re
from db_analyzer import QAPair
from migration_logger import MigrationLogger


@dataclass
class ChromaDocument:
    """Estructura de documento para ChromaDB"""
    id: str
    document: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a formato compatible con ChromaDB"""
        return {
            'id': self.id,
            'document': self.document,
            'metadata': self.metadata
        }


class DataProcessor:
    """Procesa y transforma datos para ChromaDB"""
    
    def __init__(self, logger: MigrationLogger):
        self.logger = logger
        
        # Configuración de chunking
        self.chunk_config = {
            'text': {
                'max_length': 2000,
                'chunk_size': 1000,
                'overlap': 200
            },
            'table': {
                'max_length': 3000,
                'chunk_size': 1500,
                'overlap': 300
            },
            'figure': {
                'max_length': 5000,  # Sin chunking para figuras
                'chunk_size': 5000,
                'overlap': 0
            }
        }
        
        # Cache de hashes para deduplicación
        self.content_hashes = set()
        
    def process_record(self, record: QAPair) -> List[ChromaDocument]:
        """Procesa un registro QAPair en documentos ChromaDB"""
        documents = []
        
        # 1. Crear documento Q&A combinado
        qa_doc = self._create_qa_document(record)
        documents.append(qa_doc)
        
        # 2. Crear documentos de contexto si es necesario
        if self._should_process_context(record):
            context_docs = self._create_context_documents(record)
            documents.extend(context_docs)
            
        self.logger.log_record_processed(record.id, record.type)
        return documents
        
    def _create_qa_document(self, record: QAPair) -> ChromaDocument:
        """Crea documento para el par Q&A"""
        # Formatear contenido Q&A
        qa_content = self._format_qa_content(record.instruction, record.response)
        
        # Crear metadatos
        metadata = {
            'qa_id': record.id,
            'doc_name': record.doc_name,
            'doc_page': record.doc_page,
            'type': record.type,
            'created_at': record.created_at,
            'doc_type': 'qa_pair',
            'has_context': bool(record.context and len(record.context) > 200),
            'instruction_length': len(record.instruction),
            'response_length': len(record.response)
        }
        
        # Generar ID único
        doc_id = f"qa_{record.id}"
        
        return ChromaDocument(
            id=doc_id,
            document=qa_content,
            metadata=metadata
        )
        
    def _create_context_documents(self, record: QAPair) -> List[ChromaDocument]:
        """Crea documentos para el contexto con chunking si es necesario"""
        documents = []
        
        # Obtener configuración de chunking para el tipo
        config = self.chunk_config.get(record.type, self.chunk_config['text'])
        
        # Aplicar chunking si es necesario
        if len(record.context) > config['max_length']:
            chunks = self._chunk_text(
                record.context,
                config['chunk_size'],
                config['overlap']
            )
            
            for i, chunk in enumerate(chunks):
                metadata = {
                    'qa_id': record.id,
                    'doc_name': record.doc_name,
                    'doc_page': record.doc_page,
                    'type': record.type,
                    'created_at': record.created_at,
                    'doc_type': 'context',
                    'parent_qa': f"qa_{record.id}",
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'context_length': len(record.context)
                }
                
                doc_id = f"ctx_{record.id}_{i}"
                
                documents.append(ChromaDocument(
                    id=doc_id,
                    document=chunk,
                    metadata=metadata
                ))
        else:
            # Contexto sin chunking
            metadata = {
                'qa_id': record.id,
                'doc_name': record.doc_name,
                'doc_page': record.doc_page,
                'type': record.type,
                'created_at': record.created_at,
                'doc_type': 'context',
                'parent_qa': f"qa_{record.id}",
                'context_length': len(record.context)
            }
            
            doc_id = f"ctx_{record.id}"
            
            documents.append(ChromaDocument(
                id=doc_id,
                document=record.context,
                metadata=metadata
            ))
            
        return documents
        
    def _should_process_context(self, record: QAPair) -> bool:
        """Determina si el contexto debe ser procesado como documento separado"""
        # No procesar contextos muy cortos
        if len(record.context) < 200:
            return False
            
        # No procesar contextos que son casi idénticos a la respuesta
        if self._calculate_similarity(record.response, record.context) > 0.9:
            return False
            
        return True
        
    def _format_qa_content(self, instruction: str, response: str) -> str:
        """Formatea el contenido Q&A para embeddings óptimos"""
        # Limpiar y normalizar instrucción
        if instruction.startswith("Question:"):
            instruction = instruction[9:].strip()
        elif instruction == "Provide a summary.":
            instruction = "Summary Request"
            
        # Limpiar y normalizar respuesta
        if response.startswith("Answer:"):
            response = response[7:].strip()
            
        # Formato optimizado para embeddings
        formatted = f"Question: {instruction}\n\nAnswer: {response}"
        
        return formatted
        
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Divide texto en chunks con overlap"""
        chunks = []
        
        # Limpiar texto
        text = self._clean_text(text)
        
        # Si es una tabla, intentar dividir por filas
        if self._is_table_content(text):
            chunks = self._chunk_table(text, chunk_size)
        else:
            # Chunking estándar con overlap
            start = 0
            while start < len(text):
                end = start + chunk_size
                
                # Intentar cortar en un límite de oración
                if end < len(text):
                    # Buscar el final de una oración
                    sentence_end = text.rfind('.', start + overlap, end)
                    if sentence_end > start:
                        end = sentence_end + 1
                        
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                    
                start = end - overlap
                
        return chunks
        
    def _chunk_table(self, table_text: str, chunk_size: int) -> List[str]:
        """Chunking especial para contenido de tablas"""
        chunks = []
        
        # Intentar dividir por secciones o filas
        lines = table_text.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line)
            
            if current_size + line_size > chunk_size and current_chunk:
                # Crear chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
                
        # Agregar último chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks
        
    def _clean_text(self, text: str) -> str:
        """Limpia y normaliza texto"""
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar caracteres de control
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        
        # Normalizar saltos de línea
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text.strip()
        
    def _is_table_content(self, text: str) -> bool:
        """Detecta si el texto es contenido de tabla"""
        # Heurísticas para detectar tablas
        indicators = [
            text.count('|') > 10,  # Pipes para columnas
            text.count('\t') > 5,   # Tabs para columnas
            bool(re.search(r'^\s*\d+\s*\|', text, re.MULTILINE)),  # Filas numeradas
            'Table' in text[:100]   # Palabra "Table" al inicio
        ]
        
        return sum(indicators) >= 2
        
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcula similitud simple entre dos textos"""
        # Implementación básica usando conjuntos de palabras
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
        
    def deduplicate_documents(self, documents: List[ChromaDocument]) -> List[ChromaDocument]:
        """Elimina documentos duplicados basándose en hash de contenido"""
        unique_docs = []
        
        for doc in documents:
            content_hash = hashlib.md5(doc.document.encode()).hexdigest()
            
            if content_hash not in self.content_hashes:
                self.content_hashes.add(content_hash)
                unique_docs.append(doc)
            else:
                self.logger.logger.debug(
                    f"Documento duplicado omitido: {doc.id} (hash: {content_hash})"
                )
                
        return unique_docs
        
    def batch_process_records(self, records: List[QAPair], 
                            batch_size: int = 100) -> List[List[ChromaDocument]]:
        """Procesa registros en lotes"""
        batches = []
        
        for i in range(0, len(records), batch_size):
            batch_records = records[i:i + batch_size]
            batch_docs = []
            
            for record in batch_records:
                try:
                    docs = self.process_record(record)
                    batch_docs.extend(docs)
                except Exception as e:
                    self.logger.log_record_failed(
                        record.id, 
                        f"Error procesando: {str(e)}"
                    )
                    
            # Deduplicar dentro del batch
            batch_docs = self.deduplicate_documents(batch_docs)
            batches.append(batch_docs)
            
            self.logger.log_batch_progress(
                len(batches),
                (len(records) + batch_size - 1) // batch_size,
                len(batch_records)
            )
            
        return batches
        
    def get_processing_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del procesamiento"""
        return {
            'total_hashes': len(self.content_hashes),
            'chunk_configs': self.chunk_config
        }