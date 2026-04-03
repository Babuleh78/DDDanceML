import hashlib
import tempfile
import logging
from pathlib import Path
from app.core import s3 as s3_client

logger = logging.getLogger(__name__)

def download_video_from_url(url: str) -> str:
    """
    Скачивает видео по URL через yt-dlp,
    загружает в S3 и возвращает S3-ключ.
    """
    import yt_dlp

    # Хэш URL → стабильное имя файла (кэш на уровне S3)
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    s3_key = f"videos/url_{url_hash}.mp4"

    # Проверяем — вдруг уже скачивали этот URL
    if s3_client.file_exists(s3_key):
        logger.info(f"URL already in S3: {s3_key}")
        return s3_key

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "video.mp4")

        ydl_opts = {
            "format": "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
            "outtmpl": output_path,
            "quiet": True,
            "no_warnings": True,
            "max_filesize": 200 * 1024 * 1024,  # 200MB лимит
        }

        logger.info(f"Downloading: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if not Path(output_path).exists():
            raise RuntimeError(f"yt-dlp did not produce output for {url}")

        size_mb = Path(output_path).stat().st_size / 1024 / 1024
        logger.info(f"Downloaded {size_mb:.1f}MB, uploading to S3: {s3_key}")
        s3_client.upload_file(output_path, s3_key)

    return s3_key