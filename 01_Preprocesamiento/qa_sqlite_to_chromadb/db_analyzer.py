"""
Analizador de base de datos para extracción y validación de datos de pdf_data.db
"""

import sqlite3
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import hashlib
from migration_logger import MigrationLogger


@dataclass
class QAPair:
    """Estructura de datos para un par pregunta-respuesta"""
    id: int
    instruction: str
    response: str
    context: str
    doc_name: str
    doc_page: int
    type: str
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            'id': self.id,
            'instruction': self.instruction,
            'response': self.response,
            'context': self.context,
            'doc_name': self.doc_name,
            'doc_page': self.doc_page,
            'type': self.type,
            'created_at': self.created_at
        }
    
    def get_content_hash(self) -> str:
        """Genera hash único del contenido"""
        content = f"{self.instruction}{self.response}{self.doc_name}{self.doc_page}"
        return hashlib.md5(content.encode()).hexdigest()


class DatabaseAnalyzer:
    """Analiza y extrae datos de pdf_data.db"""
    
    def __init__(self, db_path: str, logger: MigrationLogger):
        self.db_path = Path(db_path)
        self.logger = logger
        self.connection = None
        self.stats = {
            'total_records': 0,
            'records_by_type': {},
            'records_by_doc': {},
            'avg_lengths': {},
            'duplicates': 0
        }
        
    def connect(self):
        """Establece conexión con la base de datos"""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Base de datos no encontrada: {self.db_path}")
            
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            self.logger.logger.info(f"Conectado a base de datos: {self.db_path}")
        except Exception as e:
            self.logger.logger.error(f"Error conectando a base de datos: {e}")
            raise
            
    def disconnect(self):
        """Cierra la conexión con la base de datos"""
        if self.connection:
            self.connection.close()
            self.logger.logger.info("Conexión a base de datos cerrada")
            
    def analyze_structure(self) -> Dict[str, Any]:
        """Analiza la estructura de la base de datos"""
        cursor = self.connection.cursor()
        
        # Obtener información de tablas
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        structure = {}
        for table in tables:
            # Schema de la tabla
            cursor.execute(f"PRAGMA table_info({table})")
            columns = []
            for row in cursor.fetchall():
                columns.append({
                    'name': row[1],
                    'type': row[2],
                    'not_null': bool(row[3]),
                    'primary_key': bool(row[5])
                })
            
            # Conteo de registros
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            
            structure[table] = {
                'columns': columns,
                'record_count': count
            }
            
        self.logger.logger.info(f"Estructura analizada: {len(tables)} tablas encontradas")
        return structure
        
    def extract_all_records(self) -> List[QAPair]:
        """Extrae todos los registros de qa_pairs"""
        cursor = self.connection.cursor()
        
        query = """
        SELECT id, instruction, response, context, 
               doc_name, doc_page, type, created_at
        FROM qa_pairs
        ORDER BY id
        """
        
        cursor.execute(query)
        records = []
        
        for row in cursor.fetchall():
            record = QAPair(
                id=row['id'],
                instruction=row['instruction'] or '',
                response=row['response'] or '',
                context=row['context'] or '',
                doc_name=row['doc_name'] or '',
                doc_page=row['doc_page'] or 0,
                type=row['type'] or 'text',
                created_at=row['created_at'] or ''
            )
            records.append(record)
            
        self.stats['total_records'] = len(records)
        self.logger.logger.info(f"Extraídos {len(records)} registros de qa_pairs")
        
        return records
        
    def extract_batch(self, offset: int, limit: int) -> List[QAPair]:
        """Extrae un lote de registros"""
        cursor = self.connection.cursor()
        
        query = """
        SELECT id, instruction, response, context, 
               doc_name, doc_page, type, created_at
        FROM qa_pairs
        ORDER BY id
        LIMIT ? OFFSET ?
        """
        
        cursor.execute(query, (limit, offset))
        records = []
        
        for row in cursor.fetchall():
            record = QAPair(
                id=row['id'],
                instruction=row['instruction'] or '',
                response=row['response'] or '',
                context=row['context'] or '',
                doc_name=row['doc_name'] or '',
                doc_page=row['doc_page'] or 0,
                type=row['type'] or 'text',
                created_at=row['created_at'] or ''
            )
            records.append(record)
            
        return records
        
    def analyze_content(self, records: List[QAPair]) -> Dict[str, Any]:
        """Analiza el contenido de los registros"""
        # Estadísticas por tipo
        type_counts = {}
        doc_counts = {}
        length_sums = {'instruction': 0, 'response': 0, 'context': 0}
        
        # Detectar duplicados
        seen_hashes = set()
        duplicates = 0
        
        for record in records:
            # Contar por tipo
            type_counts[record.type] = type_counts.get(record.type, 0) + 1
            
            # Contar por documento
            doc_counts[record.doc_name] = doc_counts.get(record.doc_name, 0) + 1
            
            # Sumar longitudes
            length_sums['instruction'] += len(record.instruction)
            length_sums['response'] += len(record.response)
            length_sums['context'] += len(record.context)
            
            # Detectar duplicados
            content_hash = record.get_content_hash()
            if content_hash in seen_hashes:
                duplicates += 1
            seen_hashes.add(content_hash)
            
        # Calcular promedios
        total = len(records)
        avg_lengths = {
            field: length_sums[field] / total if total > 0 else 0
            for field in length_sums
        }
        
        self.stats.update({
            'records_by_type': type_counts,
            'records_by_doc': doc_counts,
            'avg_lengths': avg_lengths,
            'duplicates': duplicates
        })
        
        self.logger.logger.info(
            f"Análisis de contenido completado: "
            f"{len(type_counts)} tipos, {len(doc_counts)} documentos, "
            f"{duplicates} duplicados potenciales"
        )
        
        return self.stats
        
    def validate_data(self, records: List[QAPair]) -> Tuple[bool, List[str]]:
        """Valida la integridad de los datos"""
        issues = []
        
        for record in records:
            # Validar campos requeridos
            if not record.instruction:
                issues.append(f"Registro {record.id}: instruction vacío")
            if not record.response:
                issues.append(f"Registro {record.id}: response vacío")
            if not record.doc_name:
                issues.append(f"Registro {record.id}: doc_name vacío")
                
            # Validar tipos válidos
            if record.type not in ['text', 'table', 'figure']:
                issues.append(f"Registro {record.id}: tipo inválido '{record.type}'")
                
            # Validar longitudes extremas
            if len(record.response) > 10000:
                issues.append(
                    f"Registro {record.id}: response muy largo "
                    f"({len(record.response)} chars)"
                )
            if len(record.context) > 25000:
                issues.append(
                    f"Registro {record.id}: context muy largo "
                    f"({len(record.context)} chars)"
                )
                
        is_valid = len(issues) == 0
        
        if is_valid:
            self.logger.logger.info("Validación de datos: PASÓ - Sin problemas encontrados")
        else:
            self.logger.logger.warning(
                f"Validación de datos: {len(issues)} problemas encontrados"
            )
            for issue in issues[:10]:  # Mostrar solo los primeros 10
                self.logger.logger.warning(f"  - {issue}")
                
        return is_valid, issues
        
    def get_sample_records(self, n: int = 5, record_type: Optional[str] = None) -> List[QAPair]:
        """Obtiene registros de muestra para pruebas"""
        cursor = self.connection.cursor()
        
        if record_type:
            query = """
            SELECT id, instruction, response, context, 
                   doc_name, doc_page, type, created_at
            FROM qa_pairs
            WHERE type = ?
            LIMIT ?
            """
            cursor.execute(query, (record_type, n))
        else:
            query = """
            SELECT id, instruction, response, context, 
                   doc_name, doc_page, type, created_at
            FROM qa_pairs
            LIMIT ?
            """
            cursor.execute(query, (n,))
            
        records = []
        for row in cursor.fetchall():
            record = QAPair(
                id=row['id'],
                instruction=row['instruction'] or '',
                response=row['response'] or '',
                context=row['context'] or '',
                doc_name=row['doc_name'] or '',
                doc_page=row['doc_page'] or 0,
                type=row['type'] or 'text',
                created_at=row['created_at'] or ''
            )
            records.append(record)
            
        return records