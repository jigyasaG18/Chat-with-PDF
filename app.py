import os
import json
import shutil
from PyPDF2 import PdfReader
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    text_data = {}  # {pdf_name: text_content}
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        text_data[pdf.name] = text
    return text_data

# ---------------- TEXT SPLITTING ----------------
def get_text_chunks(text_data):
    """
    text_data: {pdf_name: text_content}
    Returns: [(chunk_text, pdf_name), ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,     # smaller chunks to avoid 413 errors
        chunk_overlap=100
    )
    chunks_with_source = []
    for pdf_name, text in text_data.items():
        chunks = splitter.split_text(text)
        for chunk in chunks:
            chunks_with_source.append((chunk, pdf_name))
    return chunks_with_source

# ---------------- VECTOR STORE ----------------
def get_vector_store(chunks_with_source):
    """
    chunks_with_source: [(chunk_text, pdf_name), ...]
    """
    if not chunks_with_source:
        st.error("No readable text found in PDFs.")
        return False

    chunks_only = [chunk for chunk, _ in chunks_with_source]
    vector_store = FAISS.from_texts(chunks_only, embedding=EMBEDDINGS)
    vector_store.save_local("faiss_index")
    
    # Save metadata mapping chunks to PDF sources
    chunk_metadata = {str(i): pdf_name for i, (_, pdf_name) in enumerate(chunks_with_source)}
    with open("chunk_metadata.json", "w") as f:
        json.dump(chunk_metadata, f)
    
    return True

def rebuild_vector_store():
    """Rebuild vector store from loaded PDFs (excluding deleted ones)"""
    if "loaded_pdfs" not in st.session_state or not st.session_state.loaded_pdfs:
        return False
    
    all_chunks_with_source = []
    for pdf_info in st.session_state.loaded_pdfs.values():
        chunks_with_source = pdf_info.get("chunks_with_source", [])
        all_chunks_with_source.extend(chunks_with_source)
    
    if not all_chunks_with_source:
        return False
    
    return get_vector_store(all_chunks_with_source)

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
    sources = set()
    
    for idx, doc in enumerate(docs):
        if len(context) + len(doc.page_content) > MAX_CONTEXT_CHARS:
            break
        context += doc.page_content + "\n\n"
        
        # Try to get source document name
        if os.path.exists("chunk_metadata.json"):
            try:
                with open("chunk_metadata.json", "r") as f:
                    chunk_metadata = json.load(f)
                    if str(idx) in chunk_metadata:
                        sources.add(chunk_metadata[str(idx)])
            except:
                pass

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
    
    # Format response with sources
    answer = response.content
    if sources:
        answer += f"\n\n📄 *Source: {', '.join(sources)}*"
    
    return answer

# ---------------- CLEAR CHAT ----------------
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "📄 Upload PDFs and ask me a question"}
    ]

# ---------------- DELETE PDF ----------------
def delete_pdf(pdf_name):
    """Delete a specific PDF and rebuild vector store"""
    if "loaded_pdfs" in st.session_state and pdf_name in st.session_state.loaded_pdfs:
        del st.session_state.loaded_pdfs[pdf_name]
        
        # Rebuild vector store without deleted PDF
        if st.session_state.loaded_pdfs:
            rebuild_vector_store()
        else:
            # If no PDFs left, remove indices
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")
            if os.path.exists("chunk_metadata.json"):
                os.remove("chunk_metadata.json")
        
        st.success(f"✅ Deleted '{pdf_name}'")
        st.rerun()

# ---------------- MAIN APP ----------------
def main():
    st.set_page_config(
        page_title="Groq PDF Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state for loaded PDFs
    if "loaded_pdfs" not in st.session_state:
        st.session_state.loaded_pdfs = {}
    
    render_header()

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.title("📌 Navigation")
        page = st.radio("Go to", ["🏠 Home", "🤖 PDF Chatbot", "ℹ️ About App"])

        if page == "🤖 PDF Chatbot":
            st.markdown("---")
            st.subheader("📚 Document Management")
            
            # Show loaded PDFs
            if st.session_state.loaded_pdfs:
                st.markdown("**Loaded Documents:**")
                for pdf_name, pdf_info in st.session_state.loaded_pdfs.items():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        timestamp = pdf_info.get("timestamp", "Unknown")
                        chunks_count = pdf_info.get("chunks_count", 0)
                        st.caption(f"📄 {pdf_name}\n*{timestamp}*\n🔗 {chunks_count} chunks")
                    with col2:
                        if st.button("🗑️", key=f"delete_{pdf_name}", help="Delete this PDF"):
                            delete_pdf(pdf_name)
            else:
                st.info("No PDFs loaded yet")
            
            st.markdown("---")
            st.subheader("📤 Upload PDFs")
            
            pdf_docs = st.file_uploader(
                "Upload PDF files",
                accept_multiple_files=True
            )

            if st.button("🚀 Submit & Process"):
                if pdf_docs:
                    with st.spinner("Processing PDFs..."):
                        text_data = get_pdf_text(pdf_docs)
                        chunks_with_source = get_text_chunks(text_data)
                        
                        # Store metadata for each PDF
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for pdf in pdf_docs:
                            pdf_chunks = [c for c, source in chunks_with_source if source == pdf.name]
                            st.session_state.loaded_pdfs[pdf.name] = {
                                "timestamp": timestamp,
                                "chunks_count": len(pdf_chunks),
                                "chunks_with_source": [(c, pdf.name) for c in pdf_chunks]
                            }
                        
                        if get_vector_store(chunks_with_source):
                            st.success("✅ PDFs processed successfully!")
                        st.rerun()
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
- � **Document management** - track, view, and delete individual PDFs
- 🔗 **Source attribution** - see which PDF each answer comes from
- 🛡️ **Token-safe architecture** (no 413 / TPM errors)  
- 💬 **Chat-style interface** with conversation history  
- ⚡ **Lightweight & local embeddings** (no OpenAI dependency)

---
### 📌 How to use
1. Go to **PDF Chatbot**
2. Upload one or more PDF files
3. Click **Submit & Process**
4. View loaded documents in the sidebar with timestamps and chunk counts
5. Ask questions in the chat box - answers will show source documents
6. Delete individual PDFs using the 🗑️ button without clearing chat history

👈 Use the sidebar to navigate between pages.
""")

    # ---------------- PDF CHATBOT PAGE ----------------
    # ---------------- PDF CHATBOT PAGE ----------------
    elif page == "🤖 PDF Chatbot":
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "📄 Upload PDFs and ask me a question"}
            ]

        # Display loaded documents summary
        if st.session_state.loaded_pdfs:
            with st.expander("📚 Loaded Documents", expanded=False):
                for pdf_name, pdf_info in st.session_state.loaded_pdfs.items():
                    st.caption(f"✅ {pdf_name} ({pdf_info.get('chunks_count', 0)} chunks)")
        
        # 1. Render existing chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # 2. Handle User Input
        prompt = st.chat_input("Ask a question...")
        
        if prompt:
            # Display user message immediately in UI
            with st.chat_message("user"):
                st.write(prompt)
            
            # Add to session state
            st.session_state.messages.append({"role": "user", "content": prompt})

            # 3. Get and Display Model Response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = user_input(prompt)
                    st.write(answer)
            
            # Add assistant response to session state
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Force a rerun to ensure the state is synced
            st.rerun()

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
