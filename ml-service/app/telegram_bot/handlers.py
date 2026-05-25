import logging

import aiohttp
from aiogram import F, Router
from aiogram.types import CallbackQuery

from app.core.config import settings

logger = logging.getLogger(__name__)
router = Router()


async def _patch_dance_status(dance_id: str, status: str) -> bool:
    url = f"{settings.go_backend_url}/admin/dances/{dance_id}/status"
    logger.info("Patching URL: %s", url)
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.patch(
                    url,
                    json={"status": status},
                    headers={"Authorization": f"Bearer {settings.admin_token}"}
                ) as resp:
                if resp.status >= 300:
                    body = await resp.text()
                    logger.error("Backend rejected status update: %s %s", resp.status, body)
                    return False
                return True
    except Exception:
        logger.exception("Failed to patch dance status dance_id=%s status=%s", dance_id, status)
        return False


async def _update_message_after_action(callback: CallbackQuery, label: str) -> None:
    new_suffix = f"\n\n{label}"
    try:
        if callback.message.caption is not None:
            await callback.message.edit_caption(
                caption=callback.message.caption + new_suffix,
                parse_mode="HTML",
                reply_markup=None,
            )
        else:
            await callback.message.edit_text(
                text=callback.message.text + new_suffix,
                parse_mode="HTML",
                reply_markup=None,
            )
    except Exception:
        logger.warning("Could not edit moderation message after action")


@router.callback_query(F.data.startswith("approve:"))
async def approve_handler(callback: CallbackQuery) -> None:
    dance_id = callback.data.split(":", 1)[1]
    logger.info("Admin approved dance_id=%s", dance_id)
    success = await _patch_dance_status(dance_id, "approved")
    if success:
        await callback.answer(
            f"Одобрено · {dance_id[:8]}… отправлен на обработку",
            show_alert=True,
        )
        await _update_message_after_action(callback, "<b>ОДОБРЕНО</b>")
    else:
        await callback.answer(
            "Бэкенд не принял статус — проверь логи main и admin_token",
            show_alert=True,
        )


@router.callback_query(F.data.startswith("reject:"))
async def reject_handler(callback: CallbackQuery) -> None:
    dance_id = callback.data.split(":", 1)[1]
    logger.info("Admin rejected dance_id=%s", dance_id)
    success = await _patch_dance_status(dance_id, "rejected")
    if success:
        await callback.answer(
            f"Отклонено · {dance_id[:8]}… помечен как rejected",
            show_alert=True,
        )
        await _update_message_after_action(callback, "<b>ОТКЛОНЕНО</b>")
    else:
        await callback.answer(
            "Бэкенд не принял статус — проверь логи main и admin_token",
            show_alert=True,
        )
