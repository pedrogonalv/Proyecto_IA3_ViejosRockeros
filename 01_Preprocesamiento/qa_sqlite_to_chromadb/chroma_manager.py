"""
Gestor de ChromaDB para inserción y gestión de embeddings
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import chromadb.errors
from typing import List, Dict, Any, Optional, Tuple
import sqlite3
import hashlib
import json
import time
import gc
import psutil
from pathlib import Path
from data_processor import ChromaDocument
from migration_logger import MigrationLogger


class EmbeddingCache:
    """Cache SQLite para embeddings generados"""
    
    def __init__(self, cache_path: str = "embeddings_cache.db", max_cache_size: int = 10000):
        self.cache_path = cache_path
        self.connection = sqlite3.connect(cache_path)
        self.max_cache_size = max_cache_size
        self._init_cache_table()
        self._init_cache_maintenance()
        
    def _init_cache_table(self):
        """Inicializa tabla de cache"""
        cursor = self.connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings_cache (
                text_hash TEXT PRIMARY KEY,
                embedding TEXT,
                model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Índice para facilitar limpieza LRU
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_last_accessed 
            ON embeddings_cache(last_accessed)
        """)
        self.connection.commit()
        
    def _init_cache_maintenance(self):
        """Inicializa mantenimiento del cache"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM embeddings_cache")
        count = cursor.fetchone()[0]
        if count > self.max_cache_size:
            self._cleanup_cache()
        
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Obtiene embedding del cache"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cursor = self.connection.cursor()
        
        cursor.execute(
            "SELECT embedding FROM embeddings_cache WHERE text_hash = ? AND model = ?",
            (text_hash, model)
        )
        
        result = cursor.fetchone()
        if result:
            # Actualizar timestamp de último acceso
            cursor.execute(
                "UPDATE embeddings_cache SET last_accessed = CURRENT_TIMESTAMP WHERE text_hash = ?",
                (text_hash,)
            )
            self.connection.commit()
            return json.loads(result[0])
        return None
        
    def put(self, text: str, embedding: List[float], model: str):
        """Guarda embedding en cache"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cursor = self.connection.cursor()
        
        # Convertir numpy array a lista si es necesario
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        
        cursor.execute(
            """INSERT OR REPLACE INTO embeddings_cache 
               (text_hash, embedding, model, last_accessed) 
               VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
            (text_hash, json.dumps(embedding), model)
        )
        self.connection.commit()
        
        # Verificar si necesitamos limpiar el cache
        cursor.execute("SELECT COUNT(*) FROM embeddings_cache")
        count = cursor.fetchone()[0]
        if count > self.max_cache_size:
            self._cleanup_cache()
            
    def _cleanup_cache(self):
        """Limpia el cache eliminando entradas menos usadas (LRU)"""
        cursor = self.connection.cursor()
        # Mantener solo el 80% del tamaño máximo
        keep_count = int(self.max_cache_size * 0.8)
        
        cursor.execute("""
            DELETE FROM embeddings_cache 
            WHERE text_hash IN (
                SELECT text_hash FROM embeddings_cache 
                ORDER BY last_accessed DESC 
                LIMIT -1 OFFSET ?
            )
        """, (keep_count,))
        
        deleted = cursor.rowcount
        self.connection.commit()
        
        if deleted > 0:
            # Compactar la base de datos
            cursor.execute("VACUUM")
        
        return deleted
        
    def close(self):
        """Cierra conexión del cache"""
        self.connection.close()


class ChromaManager:
    """Gestiona las operaciones con ChromaDB"""
    
    def __init__(self, 
                 persist_directory: str = ".",
                 collection_name: str = "tech_docs",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 logger: MigrationLogger = None):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.logger = logger
        
        # Cliente ChromaDB
        self.client = None
        self.collection = None
        
        # Cache de embeddings
        self.embedding_cache = EmbeddingCache()
        
        # Función de embeddings
        self.embedding_function = None
        
        # Estadísticas
        self.stats = {
            'documents_added': 0,
            'embeddings_generated': 0,
            'embeddings_cached': 0,
            'batches_processed': 0,
            'errors': []
        }
        
    def connect(self):
        """Conecta con ChromaDB y verifica/crea collection"""
        try:
            # Configurar cliente persistente
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Configurar función de embeddings
            if self.embedding_model == "openai":
                # Requiere OPENAI_API_KEY en variables de entorno
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=None,  # Usa variable de entorno
                    model_name="text-embedding-ada-002"
                )
            else:
                # Usar modelo local por defecto
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model
                )
            
            # Obtener o crear collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                self.logger.logger.info(
                    f"Conectado a collection existente: {self.collection_name}"
                )
                
                # Verificar configuración
                self._verify_collection_config()
                
            except chromadb.errors.NotFoundError:
                # Crear collection si no existe
                self.logger.logger.info(
                    f"Collection {self.collection_name} no encontrada, creando nueva..."
                )
                
                metadata = {
                    "hnsw:space": "l2",
                    "hnsw:construction_ef": 100,
                    "hnsw:search_ef": 100
                }
                
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata=metadata
                )
                
                self.logger.logger.info(
                    f"Collection {self.collection_name} creada exitosamente"
                )
                
            except Exception as e:
                self.logger.logger.error(
                    f"Error con collection {self.collection_name}: {e}"
                )
                raise
                
        except Exception as e:
            self.logger.logger.error(f"Error conectando a ChromaDB: {e}")
            raise
            
    def _verify_collection_config(self):
        """Verifica que la configuración de la collection sea compatible"""
        # Obtener metadatos de la collection
        collection_metadata = self.collection.metadata
        
        self.logger.logger.info(f"Configuración de collection: {collection_metadata}")
        
        # Verificar espacio vectorial solo si hay metadatos
        if collection_metadata and 'hnsw:space' in collection_metadata:
            space = collection_metadata['hnsw:space']
            if space != 'l2':
                self.logger.logger.warning(
                    f"Espacio vectorial diferente al esperado: {space}"
                )
                
    def add_documents_batch(self, documents: List[ChromaDocument]) -> Tuple[int, int]:
        """Agrega un lote de documentos a ChromaDB"""
        if not documents:
            return 0, 0
            
        success_count = 0
        error_count = 0
        
        # Monitorear memoria antes del procesamiento
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Preparar datos para ChromaDB
        ids = []
        texts = []
        metadatas = []
        embeddings = []
        
        for doc in documents:
            # Verificar si necesitamos generar embedding o usar cache
            cached_embedding = self.embedding_cache.get(
                doc.document, 
                self.embedding_model
            )
            
            if cached_embedding:
                embeddings.append(cached_embedding)
                self.stats['embeddings_cached'] += 1
                self.logger.log_embedding_generated(
                    hashlib.md5(doc.document.encode()).hexdigest(),
                    from_cache=True
                )
            else:
                # El embedding se generará automáticamente por ChromaDB
                embeddings = None  # ChromaDB generará todos los embeddings
                self.stats['embeddings_generated'] += len(documents)
                
            ids.append(doc.id)
            texts.append(doc.document)
            metadatas.append(doc.metadata)
            
        try:
            # Agregar a ChromaDB
            if embeddings and len(embeddings) == len(documents):
                # Usar embeddings cacheados
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            else:
                # Dejar que ChromaDB genere embeddings
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )
                
                # Cachear los embeddings generados para uso futuro
                if self.embedding_model != "openai":  # Solo cachear modelos locales
                    self._cache_generated_embeddings(ids, texts)
                    
            success_count = len(documents)
            self.stats['documents_added'] += success_count
            self.stats['batches_processed'] += 1
            
            self.logger.logger.info(
                f"Batch agregado exitosamente: {success_count} documentos"
            )
            
        except Exception as e:
            error_count = len(documents)
            self.stats['errors'].append({
                'batch_size': len(documents),
                'error': str(e),
                'timestamp': time.time()
            })
            
            self.logger.logger.error(
                f"Error agregando batch: {e}"
            )
            
            # Intentar agregar documentos individualmente
            success_count, error_count = self._add_documents_individually(documents)
            
        finally:
            # Liberar memoria después del batch
            del ids, texts, metadatas, embeddings
            
            # Forzar recolección de basura si el uso de memoria es alto
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = memory_after - memory_before
            
            if memory_after > 1500 or memory_increase > 200:  # MB
                gc.collect()
                memory_final = psutil.Process().memory_info().rss / 1024 / 1024
                self.logger.logger.debug(
                    f"Liberando memoria: {memory_after:.0f}MB -> {memory_final:.0f}MB"
                )
            
        return success_count, error_count
        
    def _add_documents_individually(self, documents: List[ChromaDocument]) -> Tuple[int, int]:
        """Agrega documentos uno por uno en caso de fallo del batch"""
        success_count = 0
        error_count = 0
        
        for doc in documents:
            try:
                self.collection.add(
                    ids=[doc.id],
                    documents=[doc.document],
                    metadatas=[doc.metadata]
                )
                success_count += 1
                
            except Exception as e:
                error_count += 1
                self.logger.log_record_failed(
                    doc.metadata.get('qa_id', 'unknown'),
                    str(e)
                )
                
        return success_count, error_count
        
    def _cache_generated_embeddings(self, ids: List[str], texts: List[str]):
        """Cachea embeddings generados por ChromaDB"""
        try:
            # Obtener embeddings recién generados
            results = self.collection.get(ids=ids, include=['embeddings'])
            
            if results and 'embeddings' in results:
                for i, embedding in enumerate(results['embeddings']):
                    # Corregir la evaluación ambigua del array
                    if embedding is not None and len(embedding) > 0 and i < len(texts):
                        self.embedding_cache.put(
                            texts[i],
                            embedding,
                            self.embedding_model
                        )
                        
        except Exception as e:
            self.logger.logger.warning(
                f"No se pudieron cachear embeddings: {e}"
            )
            
    def search_similar(self, query: str, n_results: int = 5, 
                      filter_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Busca documentos similares"""
        try:
            where_clause = filter_metadata if filter_metadata else None
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            return results
            
        except Exception as e:
            self.logger.logger.error(f"Error en búsqueda: {e}")
            return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
            
    def get_collection_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la collection"""
        try:
            count = self.collection.count()
            
            # Obtener muestra de metadatos para estadísticas
            sample = self.collection.get(limit=100, include=['metadatas'])
            
            doc_types = {}
            content_types = {}
            
            if sample and 'metadatas' in sample:
                for metadata in sample['metadatas']:
                    doc_type = metadata.get('doc_type', 'unknown')
                    content_type = metadata.get('type', 'unknown')
                    
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    content_types[content_type] = content_types.get(content_type, 0) + 1
                    
            return {
                'total_documents': count,
                'doc_types_sample': doc_types,
                'content_types_sample': content_types,
                'embedding_model': self.embedding_model,
                'collection_name': self.collection_name
            }
            
        except Exception as e:
            self.logger.logger.error(f"Error obteniendo estadísticas: {e}")
            return {}
            
    def verify_document_exists(self, doc_id: str) -> bool:
        """Verifica si un documento existe en la collection"""
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result['ids']) > 0
        except:
            return False
            
    def cleanup(self):
        """Limpia recursos y cierra conexiones"""
        if self.embedding_cache:
            self.embedding_cache.close()
            
        self.logger.logger.info("ChromaManager limpieza completada")
        
    def get_processing_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del procesamiento"""
        return {
            **self.stats,
            'collection_stats': self.get_collection_stats()
        }