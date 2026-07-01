from text_splitter import normalize_text, split_text


def test_normalize_text_removes_extra_blank_lines():
    text = "Hello\n\n\n\nworld"
    assert normalize_text(text) == "Hello\n\nworld"


def test_split_text_returns_overlapping_chunks():
    text = "A" * 1000
    chunks = split_text(text, chunk_size=300, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)


def test_split_text_empty_input_returns_empty_list():
    assert split_text("   ") == []
