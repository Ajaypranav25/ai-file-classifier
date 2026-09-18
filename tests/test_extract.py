from app.extract import extract


def test_extract_unknown_ext(tmp_path):
    f = tmp_path / "test.unknown"
    f.write_bytes(b"hello")
    res = extract(f)
    assert res.kind == "other"
    assert res.text == "test"
    assert res.raw_bytes is None


def test_extract_text(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    res = extract(f)
    assert res.kind == "text"
    assert res.text == "hello world"
    assert res.raw_bytes is None
