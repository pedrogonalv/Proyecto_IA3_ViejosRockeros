"""
Sistema de logging detallado para la migración de pdf_data.db a ChromaDB
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


class MigrationLogger:
    """Logger personalizado para el proceso de migración"""
    
    def __init__(self, log_dir: str = "migration_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Timestamp para esta sesión
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Configurar logger principal
        self.logger = self._setup_logger()
        
        # Métricas de proceso
        self.metrics = {
            "start_time": datetime.now(),
            "records_processed": 0,
            "records_failed": 0,
            "embeddings_generated": 0,
            "embeddings_cached": 0,
            "batches_completed": 0,
            "errors": []
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Configura el sistema de logging con múltiples handlers"""
        logger = logging.getLogger("migration")
        logger.setLevel(logging.DEBUG)
        
        # Formato detallado
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler para archivo detallado
        file_handler = logging.FileHandler(
            self.log_dir / f"migration_{self.session_id}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Handler para errores
        error_handler = logging.FileHandler(
            self.log_dir / f"errors_{self.session_id}.log"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # Agregar handlers
        logger.addHandler(file_handler)
        logger.addHandler(error_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_phase_start(self, phase: str):
        """Registra el inicio de una fase de migración"""
        self.logger.info(f"{'='*60}")
        self.logger.info(f"INICIANDO FASE: {phase}")
        self.logger.info(f"{'='*60}")
        
    def log_phase_end(self, phase: str, duration_seconds: float):
        """Registra el fin de una fase de migración"""
        self.logger.info(f"FASE COMPLETADA: {phase} - Duración: {duration_seconds:.2f}s")
        self.logger.info(f"{'='*60}\n")
        
    def log_batch_progress(self, batch_num: int, total_batches: int, 
                          records_in_batch: int):
        """Registra el progreso de procesamiento por lotes"""
        progress = (batch_num / total_batches) * 100
        self.logger.info(
            f"Batch {batch_num}/{total_batches} ({progress:.1f}%) - "
            f"Procesando {records_in_batch} registros"
        )
        self.metrics["batches_completed"] = batch_num
        
    def log_record_processed(self, record_id: int, doc_type: str):
        """Registra un registro procesado exitosamente"""
        self.metrics["records_processed"] += 1
        self.logger.debug(f"Registro procesado: ID={record_id}, Tipo={doc_type}")
        
    def log_record_failed(self, record_id: int, error: str):
        """Registra un registro que falló"""
        self.metrics["records_failed"] += 1
        self.metrics["errors"].append({
            "record_id": record_id,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        self.logger.error(f"Error procesando registro {record_id}: {error}")
        
    def log_embedding_generated(self, text_hash: str, from_cache: bool = False):
        """Registra generación o recuperación de embedding"""
        if from_cache:
            self.metrics["embeddings_cached"] += 1
            self.logger.debug(f"Embedding recuperado de caché: {text_hash}")
        else:
            self.metrics["embeddings_generated"] += 1
            self.logger.debug(f"Embedding generado: {text_hash}")
            
    def log_memory_usage(self, usage_mb: float):
        """Registra uso de memoria"""
        self.logger.debug(f"Uso de memoria: {usage_mb:.2f} MB")
        if usage_mb > 1000:  # Advertencia si supera 1GB
            self.logger.warning(f"Alto uso de memoria detectado: {usage_mb:.2f} MB")
            
    def log_validation_result(self, check: str, passed: bool, details: str = ""):
        """Registra resultados de validación"""
        status = "PASÓ" if passed else "FALLÓ"
        self.logger.info(f"Validación [{check}]: {status} {details}")
        
    def save_metrics(self):
        """Guarda las métricas finales en un archivo JSON"""
        self.metrics["end_time"] = datetime.now()
        duration = (self.metrics["end_time"] - self.metrics["start_time"]).total_seconds()
        self.metrics["total_duration_seconds"] = duration
        
        # Convertir datetime a string para JSON
        self.metrics["start_time"] = self.metrics["start_time"].isoformat()
        self.metrics["end_time"] = self.metrics["end_time"].isoformat()
        
        metrics_file = self.log_dir / f"metrics_{self.session_id}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
        self.logger.info(f"Métricas guardadas en: {metrics_file}")
        
    def get_summary(self) -> str:
        """Retorna un resumen del proceso de migración"""
        duration = (datetime.now() - self.metrics["start_time"]).total_seconds()
        
        summary = f"""
RESUMEN DE MIGRACIÓN
====================
Sesión ID: {self.session_id}
Duración total: {duration:.2f} segundos

Registros:
- Procesados: {self.metrics['records_processed']}
- Fallidos: {self.metrics['records_failed']}

Embeddings:
- Generados: {self.metrics['embeddings_generated']}
- Desde caché: {self.metrics['embeddings_cached']}

Batches completados: {self.metrics['batches_completed']}
Errores totales: {len(self.metrics['errors'])}
"""
        return summary