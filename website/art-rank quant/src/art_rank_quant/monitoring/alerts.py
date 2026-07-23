from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def alert(message: str, channel: str = "none") -> None:
    logger.warning("alert", extra={"channel": channel, "alert_message": message})
