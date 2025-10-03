import streamlit as st
from main import install_ollama, get_ollama_instructions, load_documents, create_vector_store, initialize_rag_chain, answer_question
import os
import tempfile
import logging
def main():
    """Main function to run the Streamlit app for RAG-based Q&A."""
    st.set_page_config(page_title="RAG Q&A with Ollama", page_icon="📚", layout="wide")
    st.title("📚 RAG-based Question Answering with Ollama")
    st.markdown("Upload a PDF and ask questions about its content. Powered by local Ollama LLM and FAISS vector search.")

    # Sidebar for instructions and status
    with st.sidebar:
        st.header("Setup & Status")
        try:
            model_name = install_ollama()
            st.success(f"✅ Ollama ready with model: **{model_name}**")
        except SystemExit:
            st.error("❌ Ollama not ready")
            st.markdown(get_ollama_instructions())
            st.stop()  # Halt app execution until fixed

        if 'pdf_uploaded' in st.session_state and st.session_state.pdf_uploaded:
            st.success("✅ PDF processed and RAG chain initialized.")
        else:
            st.info("👆 Upload a PDF to get started.")

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'pdf_uploaded' not in st.session_state:
        st.session_state.pdf_uploaded = False
    if 'vector_store_path' not in st.session_state:
        st.session_state.vector_store_path = None

    # PDF Upload Section
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], help="Choose a PDF to analyze.")
    with col2:
        clear_pdf = st.button("🗑️ Clear PDF", disabled=not st.session_state.pdf_uploaded)

    if clear_pdf:
        st.session_state.chat_history = []
        st.session_state.qa_chain = None
        st.session_state.pdf_uploaded = False
        st.session_state.vector_store_path = None
        st.rerun()

    if uploaded_file is not None and not st.session_state.pdf_uploaded:
        # Get a persistent directory for vector store (e.g., in session or temp)
        persist_dir = st.session_state.get('vector_store_path', tempfile.mkdtemp())
        st.session_state.vector_store_path = persist_dir
        
        with st.spinner("🔄 Processing PDF: Loading, chunking, embedding, and initializing RAG..."):
            try:
                # Load and process PDF
                chunks = load_documents(uploaded_file)
                vector_store = create_vector_store(chunks, persist_dir=persist_dir)
                
                # Initialize RAG chain
                model_name = install_ollama()  # Re-check in case
                st.session_state.qa_chain = initialize_rag_chain(vector_store, model_name=model_name)
                st.session_state.pdf_uploaded = True
                
                # Display PDF info
                st.success(f"✅ PDF processed! Loaded {len(chunks)} chunks from {len(chunks)//10 + 1} pages approx.")
                st.info(f"Vector store saved to: {persist_dir}")
                
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
                logger.error(f"PDF processing error: {e}")  # Assuming logger from main
                st.stop()

    # Chat Interface (only if PDF uploaded)
    if st.session_state.pdf_uploaded:
        # Chat history display
        if st.session_state.chat_history:
            st.subheader("💬 Chat History")
            for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-5:]), 1):  # Show last 5
                with st.expander(f"Q: {q[:50]}..."):
                    st.write(f"**Q:** {q}")
                    st.write(f"**A:** {a}")

        # Question input
        col_q, col_submit = st.columns([4, 1])
        with col_q:
            question = st.text_input("❓ Ask a question about the PDF:", key="question_input", placeholder="e.g., What is the main topic?")
        with col_submit:
            submit_btn = st.button("🚀 Submit", type="primary", disabled=not question.strip())

        if submit_btn:
            with st.spinner("🤖 Generating answer..."):
                try:
                    answer, source_docs = answer_question(
                        st.session_state.qa_chain, question, st.session_state.chat_history
                    )
                    
                    # Update chat history
                    st.session_state.chat_history.append((question, answer))
                    
                    # Display answer with sources
                    st.subheader("📝 Answer")
                    st.markdown(answer)
                    
                    if source_docs:
                        with st.expander(f"📄 Sources ({len(source_docs)} documents)", expanded=True):
                            for i, doc in enumerate(source_docs[:3], 1):  # Limit to top 3
                                with st.expander(f"Source {i}: {doc.metadata.get('source', 'Unknown')[:50]}..."):
                                    st.write(doc.page_content[:800] + "..." if len(doc.page_content) > 800 else doc.page_content)
                            if len(source_docs) > 3:
                                st.info(f"... and {len(source_docs) - 3} more sources.")
                    
                    st.rerun()  # Refresh to update history
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")
                    logger.error(f"Answer generation error: {e}")

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    else:
        st.info("👆 Please upload a PDF to start asking questions.")

if __name__ == "__main__":
    main()