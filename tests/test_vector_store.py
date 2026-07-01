from vector_store import cosine_similarity


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) > 0.99


def test_cosine_similarity_different_dimensions_returns_zero():
    assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0


def test_cosine_similarity_empty_vector_returns_zero():
    assert cosine_similarity([], [1.0]) == 0.0
