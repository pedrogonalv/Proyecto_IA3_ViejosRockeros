"""
Pipeline principal de migración de pdf_data.db a ChromaDB
"""

import os
import sys
import time
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import psutil
import argparse

from migration_logger import MigrationLogger
from db_analyzer import DatabaseAnalyzer
from data_processor import DataProcessor
from chroma_manager import ChromaManager


class MigrationCheckpoint:
    """Gestiona checkpoints para recuperación de fallos"""
    
    def __init__(self, checkpoint_dir: str = "migration_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "migration_state.json"
        
    def save_state(self, state: Dict[str, Any]):
        """Guarda el estado actual de la migración"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2)
            
    def load_state(self) -> Optional[Dict[str, Any]]:
        """Carga el estado previo si existe"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return None
        
    def clear(self):
        """Limpia los checkpoints"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


class MigrationPipeline:
    """Pipeline completo de migración"""
    
    def __init__(self, 
                 pdf_db_path: str = "data/source/pdf_data.db",
                 chroma_db_path: str = "data/vector",
                 batch_size: int = 100,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 resume: bool = False):
        
        # Paths
        self.pdf_db_path = pdf_db_path
        self.chroma_db_path = chroma_db_path
        
        # Configuración
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.resume = resume
        
        # Componentes
        self.logger = MigrationLogger()
        self.db_analyzer = DatabaseAnalyzer(pdf_db_path, self.logger)
        self.data_processor = DataProcessor(self.logger)
        self.chroma_manager = ChromaManager(
            persist_directory=chroma_db_path,
            embedding_model=embedding_model,
            logger=self.logger
        )
        
        # Checkpoint manager
        self.checkpoint = MigrationCheckpoint()
        
        # Estado de migración
        self.state = {
            'phase': 'init',
            'records_processed': 0,
            'total_records': 0,
            'batches_completed': 0,
            'start_time': None,
            'errors': []
        }
        
    def run(self):
        """Ejecuta el pipeline completo de migración"""
        try:
            self.logger.logger.info("="*80)
            self.logger.logger.info("INICIANDO MIGRACIÓN PDF_DATA.DB → CHROMADB")
            self.logger.logger.info("="*80)
            
            # Cargar estado previo si es resume
            if self.resume:
                saved_state = self.checkpoint.load_state()
                if saved_state:
                    self.state = saved_state
                    self.logger.logger.info(f"Resumiendo desde: {self.state['phase']}")
                    
            self.state['start_time'] = time.time()
            
            # Fase 1: Preparación
            if self.state['phase'] in ['init', 'preparation']:
                self._phase_preparation()
                
            # Fase 2: Extracción y Análisis
            if self.state['phase'] in ['preparation', 'extraction']:
                records = self._phase_extraction()
            else:
                # Si estamos resumiendo, necesitamos cargar los registros
                self.db_analyzer.connect()
                records = self.db_analyzer.extract_all_records()
                
            # Fase 3: Transformación y Carga
            if self.state['phase'] in ['extraction', 'transformation']:
                self._phase_transformation_and_load(records)
                
            # Fase 4: Validación
            if self.state['phase'] in ['transformation', 'validation']:
                self._phase_validation()
                
            # Fase 5: Finalización
            self._phase_completion()
            
        except KeyboardInterrupt:
            self.logger.logger.warning("Migración interrumpida por usuario")
            self._handle_interruption()
            
        except Exception as e:
            self.logger.logger.error(f"Error crítico en migración: {e}")
            self._handle_error(e)
            raise
            
        finally:
            self._cleanup()
            
    def _phase_preparation(self):
        """Fase 1: Preparación y validación inicial"""
        self.logger.log_phase_start("PREPARACIÓN")
        start_time = time.time()
        
        # Backup de ChromaDB si existe
        if Path(self.chroma_db_path, "chroma.sqlite3").exists():
            backup_path = Path(self.chroma_db_path, f"chroma_backup_{int(time.time())}.sqlite3")
            shutil.copy2(
                Path(self.chroma_db_path, "chroma.sqlite3"),
                backup_path
            )
            self.logger.logger.info(f"Backup creado: {backup_path}")
            
        # Conectar a bases de datos
        self.db_analyzer.connect()
        self.chroma_manager.connect()
        
        # Verificar estructura
        structure = self.db_analyzer.analyze_structure()
        self.logger.logger.info(f"Estructura verificada: {structure['qa_pairs']['record_count']} registros")
        
        self.state['phase'] = 'extraction'
        self.checkpoint.save_state(self.state)
        
        duration = time.time() - start_time
        self.logger.log_phase_end("PREPARACIÓN", duration)
        
    def _phase_extraction(self):
        """Fase 2: Extracción y análisis de datos"""
        self.logger.log_phase_start("EXTRACCIÓN Y ANÁLISIS")
        start_time = time.time()
        
        # Extraer todos los registros
        records = self.db_analyzer.extract_all_records()
        self.state['total_records'] = len(records)
        
        # Analizar contenido
        content_stats = self.db_analyzer.analyze_content(records)
        self.logger.logger.info(f"Análisis de contenido: {content_stats}")
        
        # Validar datos
        is_valid, issues = self.db_analyzer.validate_data(records)
        if not is_valid:
            self.logger.logger.warning(f"Problemas de validación encontrados: {len(issues)}")
            
        self.state['phase'] = 'transformation'
        self.checkpoint.save_state(self.state)
        
        duration = time.time() - start_time
        self.logger.log_phase_end("EXTRACCIÓN Y ANÁLISIS", duration)
        
        return records
        
    def _phase_transformation_and_load(self, records):
        """Fase 3: Transformación y carga a ChromaDB"""
        self.logger.log_phase_start("TRANSFORMACIÓN Y CARGA")
        start_time = time.time()
        
        # Procesar en lotes
        total_batches = (len(records) + self.batch_size - 1) // self.batch_size
        
        # Determinar desde qué batch continuar
        start_batch = self.state.get('batches_completed', 0)
        
        for batch_num in range(start_batch, total_batches):
            # Obtener registros del batch
            start_idx = batch_num * self.batch_size
            end_idx = min((batch_num + 1) * self.batch_size, len(records))
            batch_records = records[start_idx:end_idx]
            
            # Procesar batch
            self.logger.log_batch_progress(batch_num + 1, total_batches, len(batch_records))
            
            # Transformar registros a documentos ChromaDB
            documents = []
            for record in batch_records:
                try:
                    docs = self.data_processor.process_record(record)
                    documents.extend(docs)
                except Exception as e:
                    self.logger.log_record_failed(record.id, str(e))
                    self.state['errors'].append({
                        'record_id': record.id,
                        'error': str(e)
                    })
                    
            # Deduplicar documentos
            documents = self.data_processor.deduplicate_documents(documents)
            
            # Cargar a ChromaDB
            if documents:
                success, errors = self.chroma_manager.add_documents_batch(documents)
                self.state['records_processed'] += len(batch_records)
                
            # Actualizar estado y checkpoint
            self.state['batches_completed'] = batch_num + 1
            self.checkpoint.save_state(self.state)
            
            # Monitorear memoria
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self.logger.log_memory_usage(memory_mb)
            
            # Pausa entre batches para evitar sobrecarga
            if batch_num < total_batches - 1:
                time.sleep(0.5)
                
        self.state['phase'] = 'validation'
        self.checkpoint.save_state(self.state)
        
        duration = time.time() - start_time
        self.logger.log_phase_end("TRANSFORMACIÓN Y CARGA", duration)
        
    def _phase_validation(self):
        """Fase 4: Validación de la migración"""
        self.logger.log_phase_start("VALIDACIÓN")
        start_time = time.time()
        
        # Obtener estadísticas de ChromaDB
        chroma_stats = self.chroma_manager.get_collection_stats()
        
        # Validación 1: Conteo de documentos
        expected_min = self.state['total_records']  # Al menos 1 doc por registro
        actual = chroma_stats.get('total_documents', 0)
        
        self.logger.log_validation_result(
            "Conteo de documentos",
            actual >= expected_min,
            f"Esperado mínimo: {expected_min}, Actual: {actual}"
        )
        
        # Validación 2: Verificar muestra de documentos
        sample_records = self.db_analyzer.get_sample_records(10)
        verified = 0
        
        for record in sample_records:
            doc_id = f"qa_{record.id}"
            exists = self.chroma_manager.verify_document_exists(doc_id)
            if exists:
                verified += 1
                
        self.logger.log_validation_result(
            "Verificación de muestra",
            verified == len(sample_records),
            f"Verificados: {verified}/{len(sample_records)}"
        )
        
        # Validación 3: Prueba de búsqueda
        test_queries = [
            "How can I check the firmware version",
            "servo drive configuration",
            "error troubleshooting"
        ]
        
        search_success = True
        for query in test_queries:
            results = self.chroma_manager.search_similar(query, n_results=5)
            if not results['documents'][0]:
                search_success = False
                break
                
        self.logger.log_validation_result(
            "Pruebas de búsqueda",
            search_success,
            f"Probadas {len(test_queries)} queries"
        )
        
        self.state['phase'] = 'completed'
        self.checkpoint.save_state(self.state)
        
        duration = time.time() - start_time
        self.logger.log_phase_end("VALIDACIÓN", duration)
        
    def _phase_completion(self):
        """Fase 5: Finalización y reporte"""
        self.logger.log_phase_start("FINALIZACIÓN")
        
        # Guardar métricas finales
        self.logger.save_metrics()
        
        # Generar reporte final
        total_duration = time.time() - self.state['start_time']
        
        final_report = f"""
MIGRACIÓN COMPLETADA EXITOSAMENTE
=================================

Duración total: {total_duration:.2f} segundos ({total_duration/60:.1f} minutos)

Registros procesados: {self.state['records_processed']}/{self.state['total_records']}
Batches completados: {self.state['batches_completed']}
Errores encontrados: {len(self.state['errors'])}

Estadísticas de ChromaDB:
{json.dumps(self.chroma_manager.get_collection_stats(), indent=2)}

Estadísticas de Procesamiento:
{json.dumps(self.data_processor.get_processing_stats(), indent=2)}

Logs guardados en: migration_logs/
"""
        
        self.logger.logger.info(final_report)
        
        # Guardar reporte
        report_path = Path("migration_logs") / f"final_report_{self.logger.session_id}.txt"
        with open(report_path, 'w') as f:
            f.write(final_report)
            
        # Limpiar checkpoints si todo salió bien
        if len(self.state['errors']) == 0:
            self.checkpoint.clear()
            
        self.logger.log_phase_end("FINALIZACIÓN", 0)
        
    def _handle_interruption(self):
        """Maneja interrupciones del usuario"""
        self.logger.logger.info("Guardando estado para resume posterior...")
        self.checkpoint.save_state(self.state)
        self.logger.logger.info(
            f"Estado guardado. Para continuar, ejecute con --resume"
        )
        
    def _handle_error(self, error: Exception):
        """Maneja errores críticos"""
        self.state['errors'].append({
            'type': 'critical',
            'error': str(error),
            'timestamp': time.time()
        })
        self.checkpoint.save_state(self.state)
        self.logger.save_metrics()
        
    def _cleanup(self):
        """Limpieza de recursos"""
        self.db_analyzer.disconnect()
        self.chroma_manager.cleanup()
        self.logger.logger.info("Recursos liberados")


def main():
    """Función principal con argumentos CLI"""
    parser = argparse.ArgumentParser(
        description="Migración de pdf_data.db a ChromaDB"
    )
    
    parser.add_argument(
        "--pdf-db",
        default="data/source/pdf_data.db",
        help="Path a la base de datos SQLite de origen"
    )
    
    parser.add_argument(
        "--chroma-db",
        default="data/vector",
        help="Path al directorio de ChromaDB"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Tamaño del batch para procesamiento (reducir si hay problemas de memoria)"
    )
    
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=1500,
        help="Límite de memoria en MB para reducir batch size automáticamente"
    )
    
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        choices=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "openai"],
        help="Modelo de embeddings a usar"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reanudar migración interrumpida"
    )
    
    args = parser.parse_args()
    
    # Verificar requisitos
    if args.embedding_model == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY no configurada para modelo OpenAI")
        sys.exit(1)
        
    # Ejecutar migración
    pipeline = MigrationPipeline(
        pdf_db_path=args.pdf_db,
        chroma_db_path=args.chroma_db,
        batch_size=args.batch_size,
        embedding_model=args.embedding_model,
        resume=args.resume
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()