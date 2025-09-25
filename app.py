import streamlit as st
from main import install_ollama, get_ollama_instructions, load_documents, create_vector_store, initialize_rag_chain, answer_question

def main():
    """Main function to run the Streamlit app."""
    st.title("RAG-based Question Answering with Ollama")
    st.write("Upload a PDF file and ask questions about its content.")

    # Check Ollama availability
    try:
        model_name = install_ollama()
        st.success(f"Ollama server is running with model {model_name}.")
    except SystemExit:
        st.error("Ollama server is not running or not installed. Please follow these steps to set it up:")
        st.markdown(get_ollama_instructions())
        return

    # Initialize session state for chat history and QA chain
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'pdf_uploaded' not in st.session_state:
        st.session_state.pdf_uploaded = False

    # File uploader for PDF
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None and not st.session_state.pdf_uploaded:
        with st.spinner("Processing PDF and initializing RAG..."):
            try:
                # Load and process the PDF
                chunks = load_documents(uploaded_file)
                vector_store = create_vector_store(chunks)
                st.session_state.qa_chain = initialize_rag_chain(vector_store, model_name=model_name)
                st.session_state.pdf_uploaded = True
                st.success("PDF processed successfully! You can now ask questions.")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                return

    # Question input
    if st.session_state.pdf_uploaded:
        question = st.text_input("Enter your question:", key="question_input")
        if st.button("Submit"):
            if question:
                with st.spinner("Generating answer..."):
                    try:
                        answer, source_docs = answer_question(
                            st.session_state.qa_chain,
                            question,
                            st.session_state.chat_history
                        )
                        # Display answer
                        st.subheader("Answer")
                        st.write(answer)
                        
                        # Display source documents
                        st.subheader("Source Documents")
                        for i, doc in enumerate(source_docs, 1):
                            with st.expander(f"Source Document {i}"):
                                st.write(doc.page_content[:500] + "...")
                        
                        # Update chat history
                        st.session_state.chat_history.append((question, answer))
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
            else:
                st.warning("Please enter a question.")

        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for i, (q, a) in enumerate(st.session_state.chat_history, 1):
                with st.expander(f"Q{i}: {q}"):
                    st.write(f"**Answer**: {a}")

if __name__ == "__main__":
    main()