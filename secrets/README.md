# secrets/

This directory holds credentials used by the downloader service (`yt-dlp` cookies
for VK / YouTube / Instagram). **None of these files are committed** — they are
listed in `.gitignore`.

Place the following files here locally (Netscape cookie format, exported per
platform) before running the downloader:

```
secrets/
├── vk_cookies.txt
├── youtube_cookies.txt
└── instagram_cookies.txt
```

The `docker-compose.yaml` mounts this directory read-only into the worker
container at `/secrets`.

> If any cookie file was ever committed by accident, rotate it immediately
> (re-login on the source platform) — a session cookie is a live auth token.
