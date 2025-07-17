"""
Script de prueba y validación para la migración
"""

import chromadb
from chromadb.config import Settings
import sqlite3
from typing import List, Dict, Any
import json
from pathlib import Path


class MigrationTester:
    """Realiza pruebas exhaustivas post-migración"""
    
    def __init__(self, pdf_db_path: str = "data/source/pdf_data.db", 
                 chroma_db_path: str = "data/vector"):
        self.pdf_db_path = pdf_db_path
        self.chroma_db_path = chroma_db_path
        self.test_results = []
        
    def run_all_tests(self):
        """Ejecuta todos los tests de validación"""
        print("=" * 60)
        print("EJECUTANDO TESTS DE VALIDACIÓN POST-MIGRACIÓN")
        print("=" * 60)
        
        # Test 1: Verificar integridad de datos
        self._test_data_integrity()
        
        # Test 2: Probar búsquedas semánticas
        self._test_semantic_search()
        
        # Test 3: Verificar metadatos
        self._test_metadata_preservation()
        
        # Test 4: Probar filtros
        self._test_filtered_search()
        
        # Test 5: Rendimiento
        self._test_performance()
        
        # Resumen
        self._print_summary()
        
    def _test_data_integrity(self):
        """Verifica que todos los datos fueron migrados"""
        print("\n[TEST 1] Verificando integridad de datos...")
        
        # Contar registros en SQLite
        conn = sqlite3.connect(self.pdf_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM qa_pairs")
        sqlite_count = cursor.fetchone()[0]
        conn.close()
        
        # Contar documentos en ChromaDB
        client = chromadb.PersistentClient(path=self.chroma_db_path)
        collection = client.get_collection("tech_docs")
        chroma_count = collection.count()
        
        # Verificar
        passed = chroma_count >= sqlite_count
        details = f"SQLite: {sqlite_count} registros, ChromaDB: {chroma_count} documentos"
        
        self.test_results.append({
            'test': 'Integridad de datos',
            'passed': passed,
            'details': details
        })
        
        print(f"  Resultado: {'✓ PASÓ' if passed else '✗ FALLÓ'}")
        print(f"  {details}")
        
    def _test_semantic_search(self):
        """Prueba búsquedas semánticas básicas"""
        print("\n[TEST 2] Probando búsquedas semánticas...")
        
        client = chromadb.PersistentClient(path=self.chroma_db_path)
        collection = client.get_collection("tech_docs")
        
        test_queries = [
            {
                'query': "How to configure servo drive parameters",
                'expected_keywords': ['servo', 'drive', 'parameter', 'configure']
            },
            {
                'query': "Troubleshooting motor errors",
                'expected_keywords': ['error', 'troubleshoot', 'motor']
            },
            {
                'query': "Safety instructions for installation",
                'expected_keywords': ['safety', 'installation']
            }
        ]
        
        all_passed = True
        for test in test_queries:
            results = collection.query(
                query_texts=[test['query']],
                n_results=3,
                include=['documents', 'metadatas']
            )
            
            # Verificar que hay resultados
            if not results['documents'][0]:
                all_passed = False
                print(f"  ✗ Sin resultados para: {test['query']}")
                continue
                
            # Verificar relevancia
            top_result = results['documents'][0][0].lower()
            keywords_found = sum(1 for kw in test['expected_keywords'] 
                               if kw in top_result)
            
            relevance = keywords_found / len(test['expected_keywords'])
            if relevance < 0.5:
                all_passed = False
                print(f"  ✗ Baja relevancia para: {test['query']}")
            else:
                print(f"  ✓ Búsqueda exitosa: {test['query'][:50]}...")
                
        self.test_results.append({
            'test': 'Búsquedas semánticas',
            'passed': all_passed,
            'details': f"Probadas {len(test_queries)} queries"
        })
        
    def _test_metadata_preservation(self):
        """Verifica que los metadatos se preservaron correctamente"""
        print("\n[TEST 3] Verificando preservación de metadatos...")
        
        client = chromadb.PersistentClient(path=self.chroma_db_path)
        collection = client.get_collection("tech_docs")
        
        # Obtener muestra de documentos
        sample = collection.get(limit=10, include=['metadatas'])
        
        required_fields = ['qa_id', 'doc_name', 'doc_page', 'type', 'doc_type']
        all_passed = True
        
        for metadata in sample['metadatas']:
            for field in required_fields:
                if field not in metadata:
                    all_passed = False
                    print(f"  ✗ Campo faltante: {field}")
                    break
                    
        self.test_results.append({
            'test': 'Preservación de metadatos',
            'passed': all_passed,
            'details': f"Verificados {len(required_fields)} campos requeridos"
        })
        
        print(f"  Resultado: {'✓ PASÓ' if all_passed else '✗ FALLÓ'}")
        
    def _test_filtered_search(self):
        """Prueba búsquedas con filtros de metadatos"""
        print("\n[TEST 4] Probando búsquedas filtradas...")
        
        client = chromadb.PersistentClient(path=self.chroma_db_path)
        collection = client.get_collection("tech_docs")
        
        # Test por tipo de documento
        filter_tests = [
            {
                'name': 'Filtro por tipo text',
                'where': {'type': 'text'}
            },
            {
                'name': 'Filtro por documento específico',
                'where': {'doc_name': 'AX5000_SystemManual_V2_5.pdf'}
            },
            {
                'name': 'Filtro por doc_type qa_pair',
                'where': {'doc_type': 'qa_pair'}
            }
        ]
        
        all_passed = True
        for test in filter_tests:
            results = collection.query(
                query_texts=["servo configuration"],
                n_results=5,
                where=test['where'],
                include=['metadatas']
            )
            
            if not results['ids'][0]:
                all_passed = False
                print(f"  ✗ Sin resultados para: {test['name']}")
            else:
                # Verificar que los filtros se aplicaron
                for metadata in results['metadatas'][0]:
                    for key, value in test['where'].items():
                        if metadata.get(key) != value:
                            all_passed = False
                            print(f"  ✗ Filtro no aplicado correctamente: {test['name']}")
                            break
                else:
                    print(f"  ✓ Filtro exitoso: {test['name']}")
                    
        self.test_results.append({
            'test': 'Búsquedas filtradas',
            'passed': all_passed,
            'details': f"Probados {len(filter_tests)} filtros"
        })
        
    def _test_performance(self):
        """Prueba el rendimiento de las búsquedas"""
        print("\n[TEST 5] Probando rendimiento...")
        
        import time
        
        client = chromadb.PersistentClient(path=self.chroma_db_path)
        collection = client.get_collection("tech_docs")
        
        # Realizar múltiples búsquedas
        queries = [
            "motor configuration parameters",
            "error code E1234",
            "safety guidelines",
            "installation procedure",
            "troubleshooting network issues"
        ]
        
        times = []
        for query in queries:
            start = time.time()
            collection.query(query_texts=[query], n_results=10)
            elapsed = time.time() - start
            times.append(elapsed)
            
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Criterio: promedio < 1s, máximo < 2s
        passed = avg_time < 1.0 and max_time < 2.0
        
        self.test_results.append({
            'test': 'Rendimiento',
            'passed': passed,
            'details': f"Promedio: {avg_time:.3f}s, Máximo: {max_time:.3f}s"
        })
        
        print(f"  Resultado: {'✓ PASÓ' if passed else '✗ FALLÓ'}")
        print(f"  Tiempo promedio: {avg_time:.3f}s")
        print(f"  Tiempo máximo: {max_time:.3f}s")
        
    def _print_summary(self):
        """Imprime resumen de todos los tests"""
        print("\n" + "=" * 60)
        print("RESUMEN DE TESTS")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for test in self.test_results if test['passed'])
        
        for test in self.test_results:
            status = "✓ PASÓ" if test['passed'] else "✗ FALLÓ"
            print(f"{test['test']:.<40} {status}")
            print(f"  {test['details']}")
            
        print("\n" + "-" * 60)
        print(f"Total: {passed_tests}/{total_tests} tests pasados")
        
        if passed_tests == total_tests:
            print("\n✅ MIGRACIÓN VALIDADA EXITOSAMENTE")
        else:
            print("\n⚠️  ALGUNOS TESTS FALLARON - REVISAR MIGRACIÓN")
            
        # Guardar resultados
        results_file = Path("migration_logs") / "test_results.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
            
        print(f"\nResultados guardados en: {results_file}")


def main():
    """Ejecuta los tests de validación"""
    tester = MigrationTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()