from art_rank_quant.common.config import Settings, load_yaml


def test_base_configuration_is_localhost() -> None:
    config = load_yaml("configs/base.yaml")
    assert config["api_host"] == "127.0.0.1"
    assert Settings().app_host == "127.0.0.1"
    assert Settings().short_term_history_provider == "eastmoney"
    assert Settings().baostock_enabled is False
