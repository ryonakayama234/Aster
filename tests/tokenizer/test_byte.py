from aster.tokenizer.byte import encode, decode


def test_ascii_round_trip():
    text = "hello"
    ids = encode(text)

    assert decode(ids) == text


def test_japanese_round_trip():
    text = "こんにちは、Aster"
    ids = encode(text)

    assert decode(ids) == text


def test_emoji_round_trip():
    text = "Aster 🤖"
    ids = encode(text)

    assert decode(ids) == text


def test_empty_string_round_trip():
    text = ""
    ids = encode(text)

    assert decode(ids) == text