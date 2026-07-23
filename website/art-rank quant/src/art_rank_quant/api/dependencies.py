from functools import lru_cache

from art_rank_quant.common.config import Settings, get_settings


@lru_cache
def settings() -> Settings:
    return get_settings()
