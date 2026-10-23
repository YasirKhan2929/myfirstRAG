
from pathlib import Path
import re

from sentence_transformers import SentenceTransformer, util


# Locate the project directory and document.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENT_PATH = PROJECT_ROOT / "data" / "store_document.txt"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def split_into_chunks(text, max_words=35):
    """Split text into chunks without cutting sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    chunks = []
    current_chunk = []
    current_words = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        if current_chunk and current_words + word_count > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_words = 0

        current_chunk.append(sentence)
        current_words += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


class Retriever:
    """Find document chunks relevant to a user's question."""

    def __init__(self):
        if not DOCUMENT_PATH.exists():
            raise FileNotFoundError(
                f"Document not found: {DOCUMENT_PATH}"
            )

        document = DOCUMENT_PATH.read_text(encoding="utf-8")
        self.chunks = split_into_chunks(document)

        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.chunk_embeddings = self.model.encode(
            self.chunks,
            convert_to_tensor=True
        )

    def search(self, question, top_k=2):
        """Return the most relevant chunks and similarity scores."""
        question_embedding = self.model.encode(
            question,
            convert_to_tensor=True
        )

        scores = util.cos_sim(
            question_embedding,
            self.chunk_embeddings
        )[0]

        count = min(top_k, len(self.chunks))
        top_indices = scores.argsort(descending=True)[:count]

        return [
            {
                "chunk": self.chunks[index],
                "score": round(scores[index].item(), 3)
            }
            for index in top_indices
        ]
