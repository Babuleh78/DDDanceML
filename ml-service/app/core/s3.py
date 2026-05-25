# app/core/s3.py
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from app.core.config import settings
from pathlib import Path
import logging
from typing import Optional
from urllib.parse import urlparse
from boto3.s3.transfer import TransferConfig
logger = logging.getLogger(__name__)


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        config=Config(
            signature_version='s3v4',
            connect_timeout=30,
            read_timeout=300,
            retries={'max_attempts': 3},
            s3={'use_accelerate_endpoint': False},
        ),
    )


def download_file(s3_key: str, local_path: str) -> None:
    client = get_s3_client()
    try:
        client.download_file(settings.s3_bucket, s3_key, local_path)
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'NoSuchKey':
            raise FileNotFoundError(f"S3 key not found: {s3_key}")
        raise RuntimeError(f"S3 download failed for key '{s3_key}': {e}")


def upload_file(local_path: str, s3_key: str) -> None:
    client = get_s3_client()
    local_path_obj = Path(local_path)
    
    if not local_path_obj.exists():
        raise FileNotFoundError(f"File not found: {local_path}")
    
    config = TransferConfig(
        multipart_threshold=100 * 1024 * 1024,
        use_threads=False,
    )

    if s3_key.endswith('.glb'):
        content_type = 'model/gltf-binary'
    elif s3_key.endswith('.mp4'):
        content_type = 'video/mp4'
    elif s3_key.endswith('.json'):
        content_type = 'application/json'
    else:
        content_type = 'application/octet-stream'
    
    try:
        client.upload_file(
            str(local_path),
            settings.s3_bucket,
            s3_key,
            Config=config,
            ExtraArgs={'ContentType': content_type},
        )
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_msg = e.response.get('Error', {}).get('Message', str(e))
        raise RuntimeError(f"S3 upload failed for key '{s3_key}': {error_code} - {error_msg}")
    
def copy_object(src_key: str, dst_key: str) -> None:
    client = get_s3_client()
    try:
        client.copy_object(
            Bucket=settings.s3_bucket,
            CopySource={"Bucket": settings.s3_bucket, "Key": src_key},
            Key=dst_key,
        )
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code in ("NoSuchKey", "404"):
            raise FileNotFoundError(f"S3 source not found: {src_key}")
        raise RuntimeError(
            f"S3 copy failed for '{src_key}' -> '{dst_key}': {error_code} - {e}"
        )


def file_exists(key: str) -> bool:
    try:
        client = get_s3_client()
        client.head_object(Bucket=settings.s3_bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise


def _s3_key_from_url(url_or_key: str) -> str:
    if url_or_key.startswith("s3://"):
        return urlparse(url_or_key).path.lstrip("/")
    if url_or_key.startswith("http"):
        path = urlparse(url_or_key).path.lstrip("/")
        bucket = settings.s3_bucket
        if path.startswith(bucket + "/"):
            path = path[len(bucket) + 1:]
        return path
    return url_or_key


def generate_presigned_url(s3_key_or_url: str, expiry_seconds: int = 86400) -> Optional[str]:
    if not s3_key_or_url:
        return None
    try:
        key = _s3_key_from_url(s3_key_or_url)
        client = get_s3_client()
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": key},
            ExpiresIn=expiry_seconds,
        )
    except Exception as e:
        logger.warning("Could not generate presigned URL for %s: %s", s3_key_or_url, e)
        return None