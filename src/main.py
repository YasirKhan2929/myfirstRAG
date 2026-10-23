
from src.retriever import Retriever
from src.generator import Generator


def main():
    retriever = Retriever()
    generator = Generator()

    questions = [
        "How long is the return period?",
        "How long does standard delivery take?",
        "What payment methods are accepted?",
        "Do you offer free shipping?",
        "What is your store's address?",
        "Do you sell bicycles?",
    ]

    for question in questions:
        retrieved_chunks = retriever.search(question, top_k=2)
        answer = generator.answer(question, retrieved_chunks)

        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
