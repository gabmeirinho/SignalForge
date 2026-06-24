import uuid

from vector_store import make_vector_id


def test_make_vector_id_is_stable_and_qdrant_compatible():
    first = make_vector_id("0001045810-26-000021", "1A", 4)
    second = make_vector_id("0001045810-26-000021", "1A", 4)
    different = make_vector_id("0001045810-26-000021", "1A", 5)

    assert first == second
    assert first != different
    assert str(uuid.UUID(first)) == first
