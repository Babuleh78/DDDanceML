import logging
import os
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

_bot = None
_dp = None

_REASON_LABELS = {
    "animal": "Животное в кадре",
    "nsfw": "NSFW-контент",
    "no_person": "Человек не обнаружен",
    "multiple_persons": "Несколько людей в кадре",
    "other": "Другая причина",
}


def _make_bot():
    from aiogram import Bot
    from aiogram.client.session.aiohttp import AiohttpSession

    proxy = getattr(settings, "telegram_proxy", None) or getattr(settings, "ytdlp_proxy", None)

    if proxy:
        logger.info("Creating Telegram bot with proxy: %s", proxy)
        session = AiohttpSession(proxy=proxy)
        return Bot(token=settings.telegram_bot_token, session=session)

    return Bot(token=settings.telegram_bot_token)


def _ensure_initialized():
    global _bot, _dp
    if _bot is not None:
        return

    from aiogram import Dispatcher
    from app.telegram_bot.handlers import router

    _bot = _make_bot()
    _dp = Dispatcher()
    _dp.include_router(router)


def _build_keyboard(dance_id: str):
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    action_row = [
        InlineKeyboardButton(text="Одобрить", callback_data=f"approve:{dance_id}"),
        InlineKeyboardButton(text="Отклонить", callback_data=f"reject:{dance_id}"),
    ]
    return InlineKeyboardMarkup(inline_keyboard=[action_row])


def _build_caption(
    dance_id: str,
    uploader_user_id: str,
    uploader_login: str,
    reason: str,
    video_link: str = "",
) -> str:
    label = _REASON_LABELS.get(reason, reason)
    lines = [
        "<b>Модерация видео</b>",
        "",
        f"Dance ID: <code>{dance_id}</code>",
    ]
    if uploader_user_id:
        lines.append(f"User ID: <code>{uploader_user_id}</code>")
    if uploader_login:
        lines.append(f"Пользователь: <b>{uploader_login}</b>")
    if not uploader_user_id and not uploader_login:
        lines.append("Пользователь: анонимный")
    lines.append(f"Причина: <b>{label}</b>")
    if video_link:
        lines.append(f'\n<a href="{video_link}">Скачать видео (S3)</a>')
    return "\n".join(lines)


def _make_presigned_url(video_s3_url: str) -> str:
    if not video_s3_url:
        return ""
    if video_s3_url.startswith("http"):
        return video_s3_url
    try:
        from app.core.s3 import generate_presigned_url
        return generate_presigned_url(video_s3_url, expiry_seconds=86400) or ""
    except Exception as e:
        logger.warning("Could not generate presigned URL: %s", e)
        return ""


async def notify_admin(
    dance_id: str,
    reason: str,
    video_path: str,
    uploader_user_id: str,
    uploader_login: str = "",
    video_s3_url: str = "",
) -> None:
    if not settings.telegram_bot_token or not settings.telegram_admin_chat_id:
        logger.warning("Telegram bot not configured, skipping admin notification")
        return

    _ensure_initialized()

    keyboard = _build_keyboard(dance_id)
    presigned = _make_presigned_url(video_s3_url)
    caption = _build_caption(dance_id, uploader_user_id, uploader_login, reason)

    if video_path and os.path.exists(video_path):
        try:
            from aiogram.types import FSInputFile

            await _bot.send_video(
                chat_id=settings.telegram_admin_chat_id,
                video=FSInputFile(video_path),
                caption=caption,
                reply_markup=keyboard,
                parse_mode="HTML",
            )
            logger.info("Admin notified via Telegram for dance_id=%s reason=%s", dance_id, reason)
            return
        except Exception as e:
            logger.warning(
                "Could not send video file to admin (dance_id=%s): %s", dance_id, e
            )

    if presigned:
        try:
            await _bot.send_video(
                chat_id=settings.telegram_admin_chat_id,
                video=presigned,
                caption=caption,
                reply_markup=keyboard,
                parse_mode="HTML",
            )
            logger.info("Admin notified via URL for dance_id=%s reason=%s", dance_id, reason)
            return
        except Exception as e:
            logger.warning(
                "Could not send video URL to admin (dance_id=%s): %s — falling back to text", dance_id, e
            )

    caption_with_link = _build_caption(dance_id, uploader_user_id, uploader_login, reason, presigned)
    try:
        await _bot.send_message(
            chat_id=settings.telegram_admin_chat_id,
            text=caption_with_link,
            reply_markup=keyboard,
            parse_mode="HTML",
        )
    except Exception:
        logger.exception("Failed to send fallback text to admin for dance_id=%s", dance_id)


def notify_admin_sync(
    dance_id: str,
    reason: str,
    video_path: str,
    uploader_user_id: str,
    uploader_login: str = "",
    video_s3_url: str = "",
) -> None:
    import asyncio

    if not settings.telegram_bot_token or not settings.telegram_admin_chat_id:
        logger.warning("Telegram bot not configured, skipping admin notification")
        return

    async def _send() -> None:
        bot = _make_bot()
        try:
            keyboard = _build_keyboard(dance_id)
            presigned = _make_presigned_url(video_s3_url)
            caption = _build_caption(dance_id, uploader_user_id, uploader_login, reason)

            if video_path and os.path.exists(video_path):
                try:
                    from aiogram.types import FSInputFile

                    await bot.send_video(
                        chat_id=settings.telegram_admin_chat_id,
                        video=FSInputFile(video_path),
                        caption=caption,
                        reply_markup=keyboard,
                        parse_mode="HTML",
                    )
                    logger.info("Admin notified (sync) for dance_id=%s reason=%s", dance_id, reason)
                    return
                except Exception as e:
                    logger.warning("Could not send video file (dance_id=%s): %s", dance_id, e)

            if presigned:
                try:
                    await bot.send_video(
                        chat_id=settings.telegram_admin_chat_id,
                        video=presigned,
                        caption=caption,
                        reply_markup=keyboard,
                        parse_mode="HTML",
                    )
                    logger.info("Admin notified via URL (sync) for dance_id=%s reason=%s", dance_id, reason)
                    return
                except Exception as e:
                    logger.warning("Could not send video URL (dance_id=%s): %s", dance_id, e)

            caption_with_link = _build_caption(dance_id, uploader_user_id, uploader_login, reason, presigned)
            try:
                await bot.send_message(
                    chat_id=settings.telegram_admin_chat_id,
                    text=caption_with_link,
                    reply_markup=keyboard,
                    parse_mode="HTML",
                )
            except Exception:
                logger.exception("Fallback text also failed dance_id=%s", dance_id)
        finally:
            await bot.session.close()

    asyncio.run(_send())


async def start_bot() -> None:
    if not settings.telegram_bot_token:
        logger.info("TELEGRAM_BOT_TOKEN not set, bot will not start")
        return

    _ensure_initialized()
    logger.info("Starting Telegram bot polling")

    try:
        await _bot.send_message(
            chat_id=settings.telegram_admin_chat_id,
            text="Работа восстановлена",
        )
    except Exception as e:
        logger.warning("Could not send startup message: %s", e)

    await _dp.start_polling(_bot, allowed_updates=["callback_query"])
