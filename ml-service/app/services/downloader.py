import hashlib
import tempfile
import logging
import re
import subprocess
from pathlib import Path
from enum import Enum
from app.core import s3 as s3_client

logger = logging.getLogger(__name__)

MAX_SIZE_BYTES = 35 * 1024 * 1024


class Platform(Enum):
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    VK = "vk"
    UNKNOWN = "unknown"


def detect_platform(url: str) -> Platform:
    url_lower = url.lower()
    if re.search(r"(tiktok\.com|vm\.tiktok\.com)", url_lower):
        return Platform.TIKTOK
    if re.search(r"(instagram\.com|instagr\.am)", url_lower):
        return Platform.INSTAGRAM
    if re.search(r"(youtube\.com/shorts|youtu\.be|youtube\.com/watch)", url_lower):
        return Platform.YOUTUBE
    if re.search(r"(vk\.com|vk\.video|vkvideo\.ru)", url_lower):
        return Platform.VK
    return Platform.UNKNOWN


def build_yt_dlp_cmd(platform: Platform, url: str, output_template: str) -> list:
    base_cmd = [
        "yt-dlp",
        "--merge-output-format", "mp4",
        "--recode-video", "mp4",
        "-o", output_template,
    ]

    platform_args = {
        Platform.TIKTOK: [
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--add-header", "Referer:https://www.tiktok.com/",
            "--user-agent", (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
                "Mobile/15E148 Safari/604.1"
            ),
        ],
        Platform.INSTAGRAM: [
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--cookies", "/secrets/instagram_cookies.txt",
            "--user-agent", (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
                "Mobile/15E148 Safari/604.1"
            ),
        ],
        Platform.VK: [
            "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
            "--cookies", "/secrets/vk_cookies.txt",
            "--add-header", "Referer:https://vk.com/",
            "--user-agent", (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        ],
        Platform.UNKNOWN: [
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        ],
    }

    return base_cmd + platform_args[platform] + [url]


def download_youtube_video(url: str, output_dir: str) -> Path: # а нах
    from pytubefix import YouTube

    yt = YouTube(url)
    logger.info(f"YouTube title: {yt.title}")
    stream = (
        yt.streams
        .filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .last()
    )

    if not stream:
        stream = yt.streams.filter(file_extension="mp4").order_by("resolution").last()

    if not stream:
        raise RuntimeError(f"No suitable stream found for {url}")

    logger.info(f"Selected stream: {stream}")
    output_path = stream.download(output_path=output_dir, filename="video.mp4")
    return Path(output_path)


def download_video_from_url(url: str) -> str:
    platform = detect_platform(url)
    logger.info(f"Detected platform: {platform.value} for URL: {url}")

    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    s3_key = f"videos/url_{url_hash}.mp4"

    if s3_client.file_exists(s3_key):
        logger.info(f"URL already in S3: {s3_key}")
        return s3_key

    with tempfile.TemporaryDirectory() as tmpdir:

        if platform == Platform.YOUTUBE:
            output_path = download_youtube_video(url, tmpdir)
        else:
            output_template = str(Path(tmpdir) / "video.%(ext)s")
            cmd = build_yt_dlp_cmd(platform, url, output_template)

            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.stdout:
                logger.info(f"yt-dlp stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"yt-dlp stderr: {result.stderr}")

            if result.returncode != 0:
                raise RuntimeError(
                    f"yt-dlp failed with code {result.returncode}:\n{result.stderr}"
                )

            downloaded_files = list(Path(tmpdir).glob("video.*"))
            if not downloaded_files:
                raise RuntimeError(f"yt-dlp did not produce output for {url}")

            output_path = downloaded_files[0]

        file_size = output_path.stat().st_size
        if file_size > MAX_SIZE_BYTES:
            raise ValueError(
                f"Video too large: {file_size / 1024 / 1024:.1f}MB "
                f"(limit {MAX_SIZE_BYTES / 1024 / 1024:.0f}MB)"
            )

        size_mb = file_size / 1024 / 1024
        logger.info(f"Downloaded {size_mb:.1f}MB, uploading to S3: {s3_key}")
        s3_client.upload_file(str(output_path), s3_key)

    return s3_key