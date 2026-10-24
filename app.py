
import streamlit as st

from src.retriever import Retriever
from src.generator import Generator


st.set_page_config(
    page_title="My First RAG Chatbot",
    page_icon="💬",
    layout="centered",
)

st.title("💬 My First RAG Chatbot")
st.caption("Ask questions about our store. Answers are based on the store document.")

# Load the models once and reuse them between chat interactions.
@st.cache_resource
def load_rag():
    retriever = Retriever()
    generator = Generator()
    return retriever, generator


try:
    with st.spinner("Loading the RAG system..."):
        retriever, generator = load_rag()
except Exception as exc:
    st.error("The RAG system could not be loaded.")
    st.exception(exc)
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask a question about the store...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Finding an answer..."):
            try:
                retrieved_chunks = retriever.search(question, top_k=2)
                answer = generator.answer(question, retrieved_chunks)
                st.markdown(answer)

                with st.expander("View retrieved context"):
                    for result in retrieved_chunks:
                        st.write(f"Similarity score: {result['score']}")
                        st.write(result["chunk"])

            except Exception as exc:
                answer = "Sorry, I couldn't process that question."
                st.error(answer)
                st.exception(exc)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
