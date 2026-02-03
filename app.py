import os
from PyPDF2 import PdfReader
import streamlit as st
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# ---------------- LOAD ENV ----------------
load_dotenv()

# ---------------- HEADER ----------------
def render_header():
    st.markdown(
        """
        <h1 style="text-align:center;">
            🤖 PDF Chatbot
        </h1>
        """,
        unsafe_allow_html=True
    )

# ---------------- EMBEDDINGS ----------------
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # change to 'cuda' if GPU is available
)

# ---------------- LLM ----------------
LLM = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=1024
)

# ---------------- PDF READING ----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# ---------------- TEXT SPLITTING ----------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,     # smaller chunks to avoid 413 errors
        chunk_overlap=100
    )
    return splitter.split_text(text)

# ---------------- VECTOR STORE ----------------
def get_vector_store(chunks):
    if not chunks:
        st.error("No readable text found in PDFs.")
        return False

    vector_store = FAISS.from_texts(chunks, embedding=EMBEDDINGS)
    vector_store.save_local("faiss_index")
    return True

# ---------------- USER QUERY ----------------
def user_input(question):
    if not os.path.exists("faiss_index"):
        return "⚠️ Please upload and process PDFs first."

    db = FAISS.load_local(
        "faiss_index",
        EMBEDDINGS,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(question, k=2)

    # Limit total context size to prevent API errors
    MAX_CONTEXT_CHARS = 3000
    context = ""
    for doc in docs:
        if len(context) + len(doc.page_content) > MAX_CONTEXT_CHARS:
            break
        context += doc.page_content + "\n\n"

    messages = [
        SystemMessage(
            content="Answer the question only from the provided context. "
                    "If the answer is not available in the context, say: "
                    "'answer is not available in the context'."
        ),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion:\n{question}"
        )
    ]

    response = LLM.invoke(messages)
    return response.content

# ---------------- CLEAR CHAT ----------------
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "📄 Upload PDFs and ask me a question"}
    ]

# ---------------- MAIN APP ----------------
def main():
    st.set_page_config(
        page_title="Groq PDF Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    render_header()

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.title("📌 Navigation")
        page = st.radio("Go to", ["🏠 Home", "🤖 PDF Chatbot", "ℹ️ About App"])

        if page == "🤖 PDF Chatbot":
            pdf_docs = st.file_uploader(
                "Upload PDF files",
                accept_multiple_files=True
            )

            if st.button("🚀 Submit & Process"):
                if pdf_docs:
                    with st.spinner("Processing PDFs..."):
                        raw_text = get_pdf_text(pdf_docs)
                        chunks = get_text_chunks(raw_text)
                        if get_vector_store(chunks):
                            st.success("✅ PDFs processed successfully!")
                else:
                    st.error("Please upload at least one PDF.")

            st.button("🧹 Clear Chat History", on_click=clear_chat_history)

    # ---------------- HOME PAGE ----------------
    if page == "🏠 Home":
        st.markdown("## 🏠 Welcome to PDF Chatbot")
        st.markdown("""
### 🚀 What can this app do?

- 📄 **Upload multiple PDFs** and chat with them in real time  
- 🔍 **Semantic search powered by FAISS** for accurate answers  
- 🧠 **Groq + LLaMA-3.1** for ultra-fast responses  
- 🧩 **Chunked document processing** to handle large PDFs safely  
- 🛡️ **Token-safe architecture** (no 413 / TPM errors)  
- 💬 **Chat-style interface** with conversation history  
- ⚡ **Lightweight & local embeddings** (no OpenAI dependency)

---
### 📌 How to use
1. Go to **PDF Chatbot**
2. Upload one or more PDF files
3. Click **Submit & Process**
4. Ask questions in the chat box

👈 Use the sidebar to navigate between pages.
""")

    # ---------------- PDF CHATBOT PAGE ----------------
    elif page == "🤖 PDF Chatbot":
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "📄 Upload PDFs and ask me a question"}
            ]

        # Render chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # User input
        prompt = st.chat_input("Ask a question...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Get model response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = user_input(prompt)
                    st.write(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})

    # ---------------- ABOUT PAGE ----------------
    elif page == "ℹ️ About App":
        st.markdown("""
**Tech Stack**
- Streamlit  
- Groq (LLaMA-3.1-8B)  
- HuggingFace Embeddings  
- FAISS  
        """)

if __name__ == "__main__":

    main()

