# test_pipeline.py
import sys
import json
import logging
import traceback
from pathlib import Path

# === 1. Включаем логирование ДО любых импортов из app ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,  # Гарантируем вывод в консоль
    force=True  # Перезаписываем существующие хендлеры
)

logger = logging.getLogger(__name__)
print("🔍 [DEBUG] Script started", file=sys.stderr)  # Дублируем в stderr

# === 2. Добавляем корень проекта в PATH ===
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"🔍 [DEBUG] Added to path: {project_root}", file=sys.stderr)

# === 3. Импорты с отловом ошибок ===
try:
    from app.core.config import settings
    print("✅ [DEBUG] Config loaded", file=sys.stderr)
except Exception as e:
    print(f"❌ [DEBUG] Config import failed: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

try:
    from app.services.processing import process_video
    print("✅ [DEBUG] process_video imported", file=sys.stderr)
except Exception as e:
    print(f"❌ [DEBUG] process_video import failed: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)


def main():
    """Точка входа."""
    print("🚀 [DEBUG] main() entered", file=sys.stderr)
    
    try:
        video_id = "vidos"
        video_key = f"videos/{video_id}.mp4"
        
        print(f"🎬 Video key: {video_key}", file=sys.stderr)
        print(f"📦 S3 bucket: dddance", file=sys.stderr)
        print(f"🤖 Labeling: {settings.labeling_backend}", file=sys.stderr)
        
        # === ВАЖНО: process_video — синхронная функция, НЕ используем await! ===
        print("⏳ Calling process_video...", file=sys.stderr)
        result = process_video(
            video_key=video_key,
            enable_labeling=True,
        )
        print("✅ process_video returned", file=sys.stderr)
        
        # Вывод результата
        print("\n=== Результат ===", file=sys.stderr)
        print(json.dumps(result, indent=2, ensure_ascii=False), file=sys.stderr)
        
        # Проверка сегментов
        if result and result.get("segments_key"):
            print(f"\n📥 Скачиваем {result['segments_key']}...", file=sys.stderr)
            from app.core import s3 as s3_client
            local_path = Path("test_output") / f"segments_{video_id}.json"
            local_path.parent.mkdir(exist_ok=True, parents=True)
            
            s3_client.download_file(result["segments_key"], str(local_path))
            
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            print(f"\n=== Сегменты ({data.get('num_segments', 0)}) ===", file=sys.stderr)
            for seg in data.get("segments", [])[:3]:
                label = seg.get("label", "???")
                source = seg.get("label_source", "n/a")
                start_s = seg["start_ms"] / 1000
                end_s = seg["end_ms"] / 1000
                print(f"  #{seg['index']}: {label} [{start_s:.1f}-{end_s:.1f}s] ({source})", file=sys.stderr)
        
        print("\n✨ Done!", file=sys.stderr)
        return 0
        
    except Exception as e:
        print(f"\n❌ [ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    print("🔍 [DEBUG] __main__ block entered", file=sys.stderr)
    exit_code = main()
    print(f"🔍 [DEBUG] Exiting with code {exit_code}", file=sys.stderr)
    sys.exit(exit_code)