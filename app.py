"""
Консольное приложение для взаимодействия с RAG ассистентом (API mode).
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

# Загрузка переменных окружения из .env файла
# Ищем .env рядом со скриптом (корень проекта)
env_path = Path(__file__).resolve().parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

def print_banner():
    """Вывод приветственного баннера."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║         RAG Ассистент (API Mode)                        ║
║  Retrieval-Augmented Generation через OpenAI API        ║
╚══════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("Введите 'exit' или 'quit' для выхода")
    print("Введите 'stats' для просмотра статистики")
    print("Введите 'clear' для очистки кеша\n")


def print_sources(context_docs: list):
    """Красивый вывод использованных источников из metadata чанков."""
    separator = "=" * 60
    item_sep = "-" * 60
    preview_len = 200

    print(f"\n{separator}")
    print("📚 Использованные источники")
    print(separator)

    docs = [doc for doc in (context_docs or []) if isinstance(doc, dict) and doc.get("text")]
    if not docs:
        print("\nНет релевантных документов.")
        print(separator)
        return

    for i, doc in enumerate(docs, 1):
        meta = doc.get("metadata") or {}
        text = doc.get("text") or ""
        fragment = text if len(text) <= preview_len else text[:preview_len].rstrip() + "..."
        distance = doc.get("distance")
        distance_str = f"{distance:.3f}" if isinstance(distance, (int, float)) else "—"

        print(f"\nИсточник {i}\n")
        print("Файл:")
        print(meta.get("source", "—"))
        print()
        print("Тип:")
        print(meta.get("document_type", "—"))
        print()
        print("Раздел:")
        print(meta.get("section", "—"))
        print()
        print("Чанк:")
        print(meta.get("chunk_index", "—"))
        print()
        print("Distance:")
        print(distance_str)
        print()
        print("Фрагмент:")
        print()
        print(fragment)
        print()
        print(item_sep)

    print(separator)


def _retrieval_quality(best_distance, selected_count: int, threshold: float) -> str:
    """Оценка качества retrieval по лучшему distance."""
    if selected_count == 0 or best_distance is None or best_distance >= threshold:
        return "Слабое"
    if best_distance < 0.35:
        return "Отличное"
    if best_distance < 0.45:
        return "Хорошее"
    if best_distance < 0.55:
        return "Среднее"
    return "Слабое"


def print_retrieval_summary(stats: dict = None):
    """Краткая статистика retrieval после ответа."""
    separator = "=" * 60
    print(f"\n{separator}")
    print("📊 Retrieval Summary")
    print(separator)

    if not stats:
        print("\nСтатистика retrieval недоступна.")
        print(separator)
        return

    candidates = stats.get("candidates_count", 0)
    selected = stats.get("selected_count", 0)
    filtered = stats.get("filtered_count", candidates - selected)
    best = stats.get("best_distance")
    avg = stats.get("avg_distance")
    threshold = stats.get("threshold")
    quality = _retrieval_quality(best, selected, threshold if threshold is not None else 0.5)

    print()
    print("Найдено кандидатов:")
    print(candidates)
    print()
    print("Передано в LLM:")
    print(selected)
    print()
    print("Отфильтровано:")
    print(filtered)
    print()
    print("Лучший distance:")
    print(f"{best:.3f}" if isinstance(best, (int, float)) else "—")
    print()
    print("Средний distance:")
    print(f"{avg:.3f}" if isinstance(avg, (int, float)) else "—")
    print()
    print("Порог distance:")
    print(f"{threshold:.3f}" if isinstance(threshold, (int, float)) else "—")
    print()
    print("Оценка retrieval:")
    print()
    print(quality)
    print()
    print(separator)


def print_response(result: dict):
    """
    Форматированный вывод ответа, источников и retrieval summary.
    
    Args:
        result: словарь с результатом запроса
    """
    print(f"\n{'─'*60}")
    print(f"📝 Вопрос: {result['query']}")
    print(f"{'─'*60}")
    
    # Индикатор источника ответа
    if result['from_cache']:
        print("💾 Источник: КЕШ")
        if 'cached_at' in result:
            print(f"   Сохранено: {result['cached_at']}")
    elif result.get('llm_skipped'):
        print("ℹ️  Источник: база знаний (LLM не вызывался)")
    else:
        print(f"🌐 Источник: OpenAI API ({result.get('model', 'LLM')})")
        print(f"   Использовано документов: {len(result.get('context_docs', []))}")
    
    print(f"\n💬 Ответ:\n{result['answer']}")

    # Блок источников (только для ответов не из кеша со словарями документов)
    if not result.get('from_cache'):
        print_sources(result.get('context_docs') or [])
        print_retrieval_summary(result.get('retrieval_stats'))
    
    print(f"{'─'*60}\n")


def print_stats(pipeline: RAGPipeline):
    """
    Вывод статистики системы.
    
    Args:
        pipeline: экземпляр RAG pipeline
    """
    stats = pipeline.get_stats()
    
    print(f"\n{'═'*60}")
    print("📊 СТАТИСТИКА СИСТЕМЫ")
    print(f"{'═'*60}")
    
    print("\n🗄️  Векторное хранилище:")
    print(f"   Коллекция: {stats['vector_store']['name']}")
    print(f"   Документов: {stats['vector_store']['count']}")
    print(f"   Директория: {stats['vector_store']['persist_directory']}")
    
    print("\n💾 Кеш:")
    print(f"   Записей: {stats['cache']['total_entries']}")
    print(f"   Размер БД: {stats['cache']['db_size_mb']:.2f} MB")
    if stats['cache']['oldest_entry']:
        print(f"   Первая запись: {stats['cache']['oldest_entry']}")
    if stats['cache']['newest_entry']:
        print(f"   Последняя запись: {stats['cache']['newest_entry']}")
    
    print(f"\n🤖 Модель: {stats['model']}")
    print(f"🌐 Режим: {stats['mode']}")
    print(f"{'═'*60}\n")


def main():
    """Главная функция приложения."""
    print_banner()
    
    # Проверка наличия API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Ошибка: переменная окружения OPENAI_API_KEY не установлена")
        print("\nУстановите её следующим образом:")
        print("  Windows (PowerShell): $env:OPENAI_API_KEY='your-key'")
        print("  Windows (CMD): set OPENAI_API_KEY=your-key")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key'")
        sys.exit(1)
    
    try:
        # Инициализация RAG pipeline
        print("🚀 Инициализация системы...\n")
        pipeline = RAGPipeline(
            collection_name="api_rag_collection",
            cache_db_path="api_rag_cache.db",
            data_dir="data",
            model="gpt-4o-mini"
        )
        print("\n✅ Система готова к работе!\n")
        
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        sys.exit(1)
    
    # Основной цикл взаимодействия
    while True:
        try:
            # Получение запроса от пользователя
            user_input = input("💭 Ваш вопрос: ").strip()
            
            # Обработка специальных команд
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 До свидания!")
                break
            
            if user_input.lower() == 'stats':
                print_stats(pipeline)
                continue
            
            if user_input.lower() == 'clear':
                confirm = input("⚠️  Вы уверены, что хотите очистить кеш? (yes/no): ")
                if confirm.lower() in ['yes', 'y', 'да']:
                    pipeline.cache.clear()
                    print("✅ Кеш очищен")
                continue
            
            if not user_input:
                print("⚠️  Пожалуйста, введите вопрос\n")
                continue
            
            # Обработка запроса через RAG pipeline
            result = pipeline.query(user_input)
            
            # Вывод результата
            print_response(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}\n")


if __name__ == "__main__":
    main()

