"""
Модуль работы с векторным хранилищем ChromaDB.
Обрабатывает загрузку документов, chunking и поиск по векторам.
"""

import chromadb
from typing import List, Dict, Any
import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path


# Корень проекта (каталог этого файла) — пути не зависят от cwd
PROJECT_ROOT = Path(__file__).resolve().parent

# Ищем .env рядом со скриптом (корень проекта)
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# Полная диагностика чанков/overlap при загрузке (по умолчанию выключена)
DEBUG_CHUNKS = False


def _resolve_path(path: str) -> Path:
    """Относительный путь разрешается от корня проекта."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


# Соответствие имени файла типу документа
DOCUMENT_TYPE_MAP = {
    "company_overview.txt": "company_overview",
    "temperature_modes.txt": "temperature_instruction",
    "transport_rules.txt": "transport_instruction",
}

# ChromaDB (hnsw:space=cosine) возвращает distance = 1 - cosine_similarity:
#   0.0  — идентичные векторы;
#   ~0.3–0.45 — близкая по смыслу тема;
#   >0.55 — обычно слабо связано с запросом для этой базы знаний.
# Порог 0.50 отсекает шум (например company_description на вопрос о температуре),
# оставляя 1–2 действительно полезных чанка вместо принудительных top_k=3.
MAX_DISTANCE_THRESHOLD = 0.50

# Маркеры смысловых разделов: (section_id, стартовая фраза или None = начало файла)
SECTION_MARKERS = {
    "company_overview.txt": [
        ("company_description", None),
        ("work_principles", "Основные принципы работы компании:"),
        ("transport_process", "Процесс выполнения перевозки:"),
        ("regions", "Компания работает с регионами:"),
        ("temperature_monitoring", "Контроль температуры производится автоматически"),
        ("client_status", "Клиент может получить информацию"),
    ],
    "temperature_modes.txt": [
        ("temperature_ranges", None),
        ("pre_trip_checks", "Перед началом перевозки водитель обязан:"),
        ("movement_restrictions", "Во время движения запрещается:"),
        ("temperature_deviation_actions", "При отклонении температуры водитель обязан:"),
        ("monitoring_frequency", "Контроль осуществляется каждые 15 минут."),
    ],
    "transport_rules.txt": [
        ("logist_before_vehicle", None),
        ("driver_before_loading", "Перед загрузкой водитель обязан:"),
        ("driver_during_transport", "Во время перевозки водитель обязан:"),
        ("incident_actions", "При возникновении проблем:"),
        ("after_unloading", "После выгрузки водитель обязан:"),
    ],
}


class VectorStore:
    """Векторное хранилище на основе ChromaDB."""
    
    def __init__(self, collection_name: str = "rag_collection", persist_directory: str = "./chroma_db"):
        """
        Инициализация векторного хранилища.
        
        Args:
            collection_name: имя коллекции в ChromaDB
            persist_directory: директория для хранения данных
        """
        self.collection_name = collection_name
        persist_path = _resolve_path(persist_directory)
        self.persist_directory = str(persist_path)
        self.last_retrieval_stats = None
        
        # Инициализация ChromaDB клиента
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Получение или создание коллекции
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Коллекция '{collection_name}' загружена. Документов: {self.collection.count()}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Создана новая коллекция '{collection_name}'")
        
        # OpenAI клиент для создания embeddings
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Умное разбиение текста на чанки с учётом семантики.
        
        Стратегия:
        1. Приоритет абзацам (разделение по \n\n)
        2. Разбиение длинных абзацев по предложениям
        3. Сохранение контекста через overlap
        4. Учёт минимального и максимального размера чанка
        
        Args:
            text: исходный текст
            chunk_size: целевой размер чанка в символах
            overlap: размер перекрытия между чанками
            
        Returns:
            список чанков
        """
        # Разделяем текст на абзацы
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Если абзац помещается в текущий чанк
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            # Если текущий чанк не пустой и добавление абзаца превысит размер
            elif current_chunk:
                chunks.append(current_chunk)
                # Добавляем overlap из конца предыдущего чанка
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
            
            # Если абзац слишком большой, разбиваем его на предложения
            else:
                if len(paragraph) > chunk_size:
                    # Разбиваем длинный абзац на предложения
                    sentence_chunks = self._split_long_paragraph(paragraph, chunk_size, overlap)
                    
                    # Добавляем все чанки кроме последнего
                    if sentence_chunks:
                        chunks.extend(sentence_chunks[:-1])
                        current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = paragraph
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk)
        
        # Пост-обработка: фильтруем слишком короткие чанки
        chunks = [chunk for chunk in chunks if len(chunk) >= 50]
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        Получение текста для overlap из конца предыдущего чанка.
        Пытается взять целые предложения.
        
        Args:
            text: текст для извлечения overlap
            overlap_size: желаемый размер overlap
            
        Returns:
            текст overlap
        """
        if len(text) <= overlap_size:
            return text
        
        # Берём последние overlap_size символов
        overlap_candidate = text[-overlap_size:]
        
        # Ищем начало предложения в overlap
        sentence_starts = ['. ', '! ', '? ', '\n']
        best_start = 0
        
        for delimiter in sentence_starts:
            pos = overlap_candidate.find(delimiter)
            if pos != -1 and pos > best_start:
                best_start = pos + len(delimiter)
        
        if best_start > 0:
            return overlap_candidate[best_start:].strip()
        
        return overlap_candidate.strip()
    
    def _split_long_paragraph(self, paragraph: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Разбиение длинного абзаца на чанки по предложениям.
        
        Args:
            paragraph: абзац для разбиения
            chunk_size: целевой размер чанка
            overlap: размер перекрытия
            
        Returns:
            список чанков
        """
        # Разделяем на предложения
        import re
        sentences = re.split(r'([.!?]+\s+)', paragraph)
        
        # Собираем предложения обратно с их разделителями
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i + 1])
            else:
                full_sentences.append(sentences[i])
        
        # Если осталось что-то в конце без разделителя
        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])
        
        chunks = []
        current_chunk = ""
        
        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Если предложение помещается в текущий чанк
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunks.append(current_chunk)
                    # Добавляем overlap
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    # Если одно предложение больше chunk_size, всё равно добавляем его
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _get_document_type(self, filename: str) -> str:
        """Определение типа документа по имени файла."""
        return DOCUMENT_TYPE_MAP.get(filename, Path(filename).stem)

    def _split_into_sections(self, filename: str, text: str) -> List[Dict[str, str]]:
        """
        Разделение документа на смысловые разделы по заголовкам/маркерам.
        Заголовок остаётся вместе с текстом своего раздела.
        """
        markers = SECTION_MARKERS.get(filename)
        if not markers:
            return [{"section": "default", "text": text.strip()}] if text.strip() else []

        positions = []
        for section_id, marker in markers:
            if marker is None:
                positions.append((0, section_id))
                continue
            pos = text.find(marker)
            if pos == -1:
                print(f"  [!] Маркер раздела '{section_id}' не найден в {filename}")
                continue
            positions.append((pos, section_id))

        positions.sort(key=lambda item: item[0])

        sections = []
        for i, (start, section_id) in enumerate(positions):
            end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append({"section": section_id, "text": section_text})
        return sections

    def _chunk_document(
        self,
        filename: str,
        text: str,
        chunk_size: int = 500,
        overlap: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Section-aware chunking:
        - сначала делим документ на смысловые разделы;
        - короткий раздел = один чанк (заголовок + содержание вместе);
        - длинный раздел делится существующим _chunk_text (overlap только внутри раздела).
        """
        sections = self._split_into_sections(filename, text)
        results: List[Dict[str, Any]] = []
        chunk_index = 0

        for section in sections:
            section_id = section["section"]
            section_text = section["text"]

            if len(section_text) <= chunk_size:
                results.append({
                    "text": section_text,
                    "section": section_id,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
                continue

            # Overlap только внутри длинного раздела
            sub_chunks = self._chunk_text(section_text, chunk_size=chunk_size, overlap=overlap)
            for sub_chunk in sub_chunks:
                results.append({
                    "text": sub_chunk,
                    "section": section_id,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1

        return results

    def _find_overlap_fragment(self, prev_chunk: str, next_chunk: str) -> str:
        """Находит общий фрагмент: суффикс prev_chunk = префикс next_chunk."""
        max_check = min(len(prev_chunk), len(next_chunk))
        for length in range(max_check, 0, -1):
            if next_chunk.startswith(prev_chunk[-length:]):
                return prev_chunk[-length:]
        return ""

    def _print_chunk_diagnostics(
        self,
        source: str,
        document_type: str,
        chunk_items: List[Dict[str, Any]],
        overlap: int = 100,
    ):
        """Диагностический вывод чанков, section и metadata."""
        separator = "-" * 60

        for item in chunk_items:
            metadata = {
                "source": source,
                "document_type": document_type,
                "section": item["section"],
                "chunk_index": item["chunk_index"],
            }
            print(separator)
            print(f"Файл: {source}")
            print(f"section: {item['section']}")
            print(f"chunk_index: {item['chunk_index']}")
            print(f"Размер: {len(item['text'])} символов")
            print("Текст:")
            print(item["text"])
            print(f"metadata: {metadata}")
            print(separator)

        print(f"\n[Overlap] Целевое значение внутри длинного раздела: {overlap} символов")
        same_section_pairs = 0
        for i in range(len(chunk_items) - 1):
            prev_item = chunk_items[i]
            next_item = chunk_items[i + 1]
            if prev_item["section"] != next_item["section"]:
                continue

            same_section_pairs += 1
            fragment = self._find_overlap_fragment(prev_item["text"], next_item["text"])
            print(separator)
            print(f"Overlap внутри section '{prev_item['section']}' "
                  f"(chunk {prev_item['chunk_index']} → {next_item['chunk_index']}):")
            if not fragment:
                print("  Общий фрагмент не найден.")
            else:
                print(f"  Размер: {len(fragment)} символов")
                print("  Повторяющийся фрагмент:")
                print(fragment)
            print(separator)

        if same_section_pairs == 0:
            print("[Overlap] Внутри разделов соседних пар нет "
                  "(каждый раздел уместился в один чанк — overlap между разделами не применяется).")

    def _get_loaded_sources(self) -> set:
        """Имена файлов (source), чанки которых уже есть в коллекции."""
        if self.collection.count() == 0:
            return set()

        result = self.collection.get(include=["metadatas"])
        sources = set()
        for meta in result.get("metadatas") or []:
            if meta and meta.get("source"):
                sources.add(meta["source"])
        return sources

    def load_documents(self, data_dir: str):
        """
        Загрузка всех .txt файлов из папки в векторное хранилище.
        Каждый файл читается и обрабатывается отдельно.
        Уже загруженные файлы (по метаданным source) пропускаются.
        
        Args:
            data_dir: путь к папке с документами
        """
        data_path = _resolve_path(data_dir)
        if not data_path.is_dir():
            raise FileNotFoundError(f"Папка {data_path} не найдена")

        txt_files = sorted(data_path.glob("*.txt"))
        print(f"Найдено файлов: {len(txt_files)}")

        if not txt_files:
            print("В папке нет .txt файлов для загрузки")
            return

        loaded_sources = self._get_loaded_sources()
        total_added = 0

        for file_path in txt_files:
            source = file_path.name

            if source in loaded_sources:
                print(f"  {source}: уже загружен, пропуск")
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunk_items = self._chunk_document(source, text)
            print(f"  {source}: создано {len(chunk_items)} чанков")

            if not chunk_items:
                continue

            document_type = self._get_document_type(source)

            # Полная диагностика чанков — только при DEBUG_CHUNKS=True
            if DEBUG_CHUNKS:
                self._print_chunk_diagnostics(source, document_type, chunk_items)

            documents = []
            ids = []
            embeddings = []
            metadatas = []

            for item in chunk_items:
                chunk = item["text"]
                chunk_index = item["chunk_index"]
                embedding = self._create_embedding(chunk)
                documents.append(chunk)
                ids.append(f"{file_path.stem}_{chunk_index}")
                embeddings.append(embedding)
                metadatas.append({
                    "source": source,
                    "document_type": document_type,
                    "section": item["section"],
                    "chunk_index": chunk_index,
                })

                if (chunk_index + 1) % 10 == 0:
                    print(f"    Обработано {chunk_index + 1}/{len(chunk_items)} чанков")

            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )
            total_added += len(chunk_items)

            if DEBUG_CHUNKS:
                stored = self.collection.get(ids=[ids[0]], include=["metadatas"])
                print(f"  Metadata в ChromaDB (id={ids[0]}): {stored['metadatas'][0]}")

        print(f"Всего добавлено чанков в ChromaDB: {total_added}")
    
    def _create_embedding(self, text: str) -> List[float]:
        """
        Создание векторного представления текста через OpenAI.
        
        Args:
            text: текст для векторизации
            
        Returns:
            вектор embeddings
        """
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def _print_retrieval_diagnostics(
        self,
        candidates: List[Dict[str, Any]],
        selected: List[Dict[str, Any]],
    ):
        """Диагностика сырых результатов поиска и отобранного контекста."""
        print("=" * 50)
        print("Retrieval Results")
        print("=" * 50)

        if not candidates:
            print("(нет результатов)")
        else:
            for i, doc in enumerate(candidates, 1):
                meta = doc.get("metadata") or {}
                print(f"\n{i}.")
                print(f"distance: {doc.get('distance')}")
                print(f"source: {meta.get('source')}")
                print(f"section: {meta.get('section')}")
                print(f"chunk_index: {meta.get('chunk_index')}")

        print()
        print("=" * 50)
        print("Selected Context")
        print("=" * 50)

        if not selected:
            print("(после фильтрации документов не осталось)")
        else:
            for doc in selected:
                meta = doc.get("metadata") or {}
                print(f"- source: {meta.get('source')}")
                print(f"  section: {meta.get('section')}")
                print(f"  distance: {doc.get('distance')}")

        print(f"\nДокументов передано в LLM: {len(selected)}")
        print("=" * 50)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов по запросу.
        Берёт top_k кандидатов из ChromaDB, затем отфильтровывает
        слишком далёкие по cosine distance (MAX_DISTANCE_THRESHOLD).
        
        Args:
            query: текст запроса
            top_k: число кандидатов до фильтрации
            
        Returns:
            список документов с метаданными (может быть меньше top_k)
        """
        # Создание embedding для запроса
        query_embedding = self._create_embedding(query)
        
        # Поиск в ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Форматирование сырых кандидатов
        candidates = []
        if results['documents'] and len(results['documents']) > 0:
            metadatas = results.get('metadatas') or [[]]
            distances = results.get('distances') or [[]]
            for i in range(len(results['documents'][0])):
                candidates.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'distance': distances[0][i] if i < len(distances[0]) else None,
                    'metadata': metadatas[0][i] if i < len(metadatas[0]) else None,
                })

        # Фильтрация по cosine distance: чем меньше, тем релевантнее
        selected = [
            doc for doc in candidates
            if doc.get('distance') is not None and doc['distance'] <= MAX_DISTANCE_THRESHOLD
        ]

        selected_distances = [
            doc['distance'] for doc in selected if doc.get('distance') is not None
        ]
        candidate_distances = [
            doc['distance'] for doc in candidates if doc.get('distance') is not None
        ]
        stats_distances = selected_distances or candidate_distances

        self.last_retrieval_stats = {
            'candidates_count': len(candidates),
            'selected_count': len(selected),
            'filtered_count': len(candidates) - len(selected),
            'best_distance': min(stats_distances) if stats_distances else None,
            'avg_distance': (
                sum(stats_distances) / len(stats_distances) if stats_distances else None
            ),
            'threshold': MAX_DISTANCE_THRESHOLD,
        }

        self._print_retrieval_diagnostics(candidates, selected)
        
        return selected
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Получение статистики коллекции.
        
        Returns:
            словарь со статистикой
        """
        return {
            'name': self.collection_name,
            'count': self.collection.count(),
            'persist_directory': self.persist_directory
        }


if __name__ == "__main__":
    # Тестирование векторного хранилища
    import sys
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Ошибка: установите переменную окружения OPENAI_API_KEY")
        sys.exit(1)
    
    vector_store = VectorStore(collection_name="test_collection")
    
    # Загрузка документов из папки data (путь относительно корня проекта)
    vector_store.load_documents("data")
    
    # Поиск
    results = vector_store.search("Какая температура нужна для перевозки замороженной продукции?", top_k=3)
    print("\nРезультаты поиска:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc['text'][:200]}...")
        print(f"   Distance: {doc['distance']}")
    
    # Статистика
    stats = vector_store.get_collection_stats()
    print(f"\nСтатистика: {stats}")

