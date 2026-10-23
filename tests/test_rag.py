
from src.retriever import Retriever


def test_retriever_returns_relevant_context():
    retriever = Retriever()

    results = retriever.search(
        "How long is the return period?",
        top_k=2,
    )

    assert len(results) > 0

    combined_text = " ".join(
        result["chunk"] for result in results
    ).lower()

    assert "7 days" in combined_text


def test_retriever_returns_similarity_scores():
    retriever = Retriever()

    results = retriever.search(
        "What payment methods are accepted?",
        top_k=2,
    )

    assert len(results) > 0

    for result in results:
        assert "chunk" in result
        assert "score" in result
        assert isinstance(result["score"], float)
