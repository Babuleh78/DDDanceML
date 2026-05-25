import hashlib
from typing import Optional
import tempfile
import logging
import subprocess
from pathlib import Path
from enum import Enum
from urllib.parse import urlparse
from app.core import s3 as s3_client
from app.core.config import settings

logger = logging.getLogger(__name__)

MAX_SIZE_BYTES = 35 * 1024 * 1024

YT_DLP_TIMEOUT_SEC = 120

ALLOWED_URL_SCHEMES = {"http", "https"}

GEO_BLOCK_PHRASES = [
    "Video not available",
    "status code 0",
    "This video is unavailable",
    "Content not available in your area",
    "not available in your country",
    "This video is only available",
    "HTTP Error 403",
    "Unable to download webpage",
    "Sorry, this content isn't available",
]

PLATFORM_PROXY_MAP = {
    "INSTAGRAM": "proxy_instagram",
    "TIKTOK": "proxy_tiktok",
    "VK": "proxy_vk",
    "YOUTUBE": "proxy_youtube",
    "UNKNOWN": "ytdlp_proxy",
}


class Platform(Enum):
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    VK = "vk"
    UNKNOWN = "unknown"


_PLATFORM_HOST_SUFFIXES = {
    Platform.TIKTOK: ("tiktok.com",),
    Platform.INSTAGRAM: ("instagram.com", "instagr.am"),
    Platform.YOUTUBE: ("youtube.com", "youtu.be"),
    Platform.VK: ("vk.com", "vkvideo.ru", "vk.video"),
}


def _host_matches(host: str, suffixes: tuple) -> bool:
    return any(host == s or host.endswith("." + s) for s in suffixes)


def detect_platform(url: str) -> Platform:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return Platform.UNKNOWN

    if not host:
        return Platform.UNKNOWN

    for platform, suffixes in _PLATFORM_HOST_SUFFIXES.items():
        if _host_matches(host, suffixes):
            return platform

    return Platform.UNKNOWN


def validate_video_url(url: str, platform: Platform) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ALLOWED_URL_SCHEMES:
        raise ValueError(f"Недопустимая схема URL: {parsed.scheme!r}")

    if not parsed.hostname:
        raise ValueError("URL не содержит хоста")

    if platform == Platform.UNKNOWN:
        raise ValueError(
            "Поддерживаются только ссылки на TikTok, Instagram, YouTube и VK"
        )


def get_proxy_for_platform(platform: Platform) -> Optional[str]:
    setting_key = PLATFORM_PROXY_MAP.get(platform.name, "ytdlp_proxy")
    proxy = getattr(settings, setting_key, None)
    if not proxy:
        proxy = getattr(settings, "ytdlp_proxy", None)
    return proxy


def build_yt_dlp_cmd(platform: Platform, url: str, output_template: str, proxy: Optional[str] = None) -> list:
    base_cmd = [
        "yt-dlp",
        "--merge-output-format", "mp4",
        "--recode-video", "mp4",
        "-o", output_template,
    ]

    proxy_args = ["--proxy", proxy] if proxy else []

    platform_args = {
        Platform.TIKTOK: [
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--add-header", "Referer:https://www.tiktok.com/",
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

    return base_cmd + proxy_args + platform_args[platform] + [url]


def run_yt_dlp(platform: Platform, url: str, output_template: str) -> subprocess.CompletedProcess:
    proxy = get_proxy_for_platform(platform)

    if proxy:
        cmd = build_yt_dlp_cmd(platform, url, output_template, proxy=proxy)
        logger.info(f"Running with proxy for {platform.value}")
        return subprocess.run(cmd, capture_output=True, text=True, timeout=YT_DLP_TIMEOUT_SEC)

    cmd = build_yt_dlp_cmd(platform, url, output_template)
    logger.info(f"Running without proxy for {platform.value}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=YT_DLP_TIMEOUT_SEC)

    if result.returncode != 0:
        stderr = result.stderr or ""
        is_geo_block = any(phrase in stderr for phrase in GEO_BLOCK_PHRASES)
        fallback_proxy = getattr(settings, "ytdlp_proxy", None)
        if is_geo_block and fallback_proxy:
            logger.warning("Geo-block detected, retrying with fallback proxy")
            cmd = build_yt_dlp_cmd(platform, url, output_template, proxy=fallback_proxy)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=YT_DLP_TIMEOUT_SEC)

    return result


def download_youtube_video(url: str, output_dir: str) -> Path:
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
    if url and "://" not in url:
        url = "https://" + url

    platform = detect_platform(url)
    logger.info(f"Detected platform: {platform.value} for URL: {url}")

    validate_video_url(url, platform)

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
            try:
                result = run_yt_dlp(platform, url, output_template)
            except subprocess.TimeoutExpired as e:
                raise ValueError(
                    f"Скачивание видео превысило лимит {YT_DLP_TIMEOUT_SEC}с"
                ) from e

            if result.stdout:
                logger.info(f"yt-dlp stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"yt-dlp stderr: {result.stderr}")

            if result.returncode != 0:
                stderr = result.stderr or ""
                if any(phrase in stderr for phrase in GEO_BLOCK_PHRASES):
                    raise ValueError(
                        "Видео недоступно — возможно удалено, приватно "
                        "или заблокировано в вашем регионе"
                    )
                raise RuntimeError(
                    f"yt-dlp failed with code {result.returncode}:\n{stderr}"
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