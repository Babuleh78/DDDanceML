"""Shared test setup.

`app.core.config` instantiates a `Settings()` singleton at import time, and the
S3 fields are required. We inject dummy values into the environment *before* any
test imports the module so the import succeeds without a real `.env`.
"""

import os

os.environ.setdefault("S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("S3_ACCESS_KEY", "test")
os.environ.setdefault("S3_SECRET_KEY", "test")
os.environ.setdefault("S3_BUCKET", "test-bucket")
