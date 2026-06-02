from mtdata.services.news_text import normalize_news_text


def test_normalize_news_text_repairs_oem_smart_quote_mojibake() -> None:
    assert normalize_news_text("HPE\u00d4\u00c7\u00d6s stock") == "HPE\u2019s stock"


def test_normalize_news_text_repairs_double_encoded_oem_mojibake() -> None:
    garbled = "HPE\u00c3\u201d\u00c3\u2021\u00c3\u2013s stock"

    assert normalize_news_text(garbled) == "HPE\u2019s stock"


def test_normalize_news_text_strips_controls_and_compacts_whitespace() -> None:
    assert normalize_news_text("  Fed\x00  holds\r\nrates  ") == "Fed holds rates"
