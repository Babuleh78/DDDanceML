import boto3
from botocore.exceptions import ClientError
from app.core.config import settings


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
    )


def download_file(s3_key: str, local_path: str) -> None:
    """Скачивает файл из S3 по ключу во временный локальный путь."""
    client = get_s3_client()
    try:
        client.download_file(settings.s3_bucket, s3_key, local_path)
    except ClientError as e:
        raise RuntimeError(f"S3 download failed for key '{s3_key}': {e}")


def upload_file(local_path: str, s3_key: str) -> None:
    """Загружает локальный файл в S3."""
    client = get_s3_client()
    try:
        client.upload_file(local_path, settings.s3_bucket, s3_key)
    except ClientError as e:
        raise RuntimeError(f"S3 upload failed for key '{s3_key}': {e}")