# My First RAG System

A practical Retrieval-Augmented Generation (RAG) project built with Python, Sentence Transformers, and Qwen 2.5.

## Overview

This project answers questions using information retrieved from a local store document. It combines semantic search with a language model to generate context-grounded responses.

When the document does not contain enough relevant information, the system attempts to return:

> I don't know based on the available information.

## Architecture

1. **Document loading:** Reads the store information from `data/store_document.txt`.
2. **Chunking:** Splits the document into smaller text segments.
3. **Embeddings:** Converts document chunks into vectors using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Semantic retrieval:** Ranks chunks by cosine similarity to the user's question.
5. **Answer generation:** Uses `Qwen/Qwen2.5-0.5B-Instruct` to answer using the retrieved context.
6. **Missing-information handling:** Uses a similarity threshold and prompting to reduce unsupported answers.

## Technologies

- Python
- Sentence Transformers
- Hugging Face Transformers
- Qwen 2.5 0.5B Instruct
- PyTorch
- pytest

## Project Structure

```text
myfirstRAG/
├── data/
│   └── store_document.txt
├── src/
│   ├── __init__.py
│   ├── retriever.py
│   ├── generator.py
│   └── main.py
├── tests/
│   └── test_rag.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Python and Git should be installed first.

Create a virtual environment:

```powershell
python -m venv .venv
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the Project

From the `myfirstRAG` directory:

```powershell
.\.venv\Scripts\python.exe -m src.main
```

The first run downloads the embedding model and language model. An internet connection is required for the initial downloads.

## Run Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag.py -v
```

## Example Questions

- How long is the return period?
- How long does standard delivery take?
- What payment methods are accepted?
- Do you offer free shipping?
- What is your store's address?

The expected answers depend on the contents of the local store document.

## Current Limitations

- The project uses a small example document.
- The similarity threshold is a basic safeguard, not a guarantee of answer correctness.
- The language model may still generate unsupported information.
- Automated tests currently focus on retrieval, not the factual accuracy of generated answers.
- The language model may require significant memory and can run slowly on CPU.

## Future Improvements

- Add more source documents.
- Improve chunking and retrieval evaluation.
- Add tests for missing-information questions and generated answers.
- Build an interactive interface using Streamlit.
- Expose the RAG pipeline through a FastAPI endpoint.
- Explore vector databases such as Pinecone.

## Learning Goals

This project demonstrates the core RAG workflow: document processing, embeddings, semantic retrieval, context-grounded generation, and basic automated testing.
