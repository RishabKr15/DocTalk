import subprocess
import os
import tempfile
import sys
import re
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

#This main file will give you the answers by using context from LLM alog with pdf you have given as input.

def check_gpu_availability():
    """Check if a GPU is available and Ollama is configured to use it."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        stdout = result.stdout
        if "NVIDIA" in stdout or "Driver Version" in stdout:
            match = re.search(r"(\d+)\s*MiB\s*/\s*(\d+)\s*MiB", stdout)
            if match:
                used_vram = int(match.group(1))
                total_vram = int(match.group(2))
                print(f"GPU detected: {total_vram}MiB total VRAM, {used_vram}MiB used.")
                if total_vram <= 4096:
                    print("Warning: 4GB VRAM detected. Using CPU mode to avoid VRAM contention.")
                    return False, total_vram, used_vram
                return True, total_vram, used_vram
            else:
                print("GPU detected, but VRAM details could not be parsed. Using CPU mode.")
                return False, None, None
        return False, None, None
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("No NVIDIA GPU detected or nvidia-smi not found. Falling back to CPU.")
        return False, None, None

def install_ollama():
    """Check if Ollama is installed and ensure the model is available."""
    try:
        subprocess.run(["ollama", "--version"], check=True, capture_output=True)
        print("Ollama is installed.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Ollama not found. Please install Ollama: https://ollama.com/download")
        sys.exit(1)
    
    model_name = "llama3.2:1b"
    try:
        result = subprocess.run(["ollama", "list"], check=True, capture_output=True, text=True)
        if model_name in result.stdout:
            print(f"Model {model_name} is already available.")
            return model_name
    except subprocess.CalledProcessError:
        pass

    try:
        print(f"Pulling model {model_name}...")
        subprocess.run(["ollama", "pull", model_name], check=True, capture_output=True)
        print(f"Model {model_name} pulled successfully.")
        return model_name
    except subprocess.CalledProcessError as e:
        print(f"Error pulling model {model_name}: {e}")
        alternative_models = ["llama3.2:latest", "llama3.2:3b"]
        for alt_model in alternative_models:
            try:
                print(f"Trying alternative model: {alt_model}")
                subprocess.run(["ollama", "pull", alt_model], check=True, capture_output=True)
                print(f"Successfully pulled model: {alt_model}")
                return alt_model
            except subprocess.CalledProcessError:
                continue
        
        print("Could not pull any model. Please check your Ollama installation.")
        sys.exit(1)

def get_ollama_instructions():
    """Return instructions for installing and starting Ollama."""
    return """
    ### Install and Start Ollama
    **For Windows:**
    1. Download the Ollama installer from [https://ollama.com/download](https://ollama.com/download).
    2. Run the installer and follow the prompts.
    3. Open a command prompt and start the Ollama server:
       ```cmd
       ollama serve
       ```
    4. Pull the llama3.2:1b model:
       ```cmd
       ollama pull llama3.2:1b
       ```
    5. If errors persist, try running in CPU mode:
       ```cmd
       set OLLAMA_NUM_GPU=0
       ollama run llama3.2:1b
       ```
    After completing these steps, refresh the page to continue.
    """

def load_documents(pdf_file):
    """Load and split PDF documents into chunks."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name
        pdf_reader = PyPDFLoader(tmp_file_path)
        documents = pdf_reader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        chunks = text_splitter.split_documents(documents)
        os.unlink(tmp_file_path)
        return chunks
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        raise

def create_vector_store(chunks):
    """Create a FAISS vector store from document chunks."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        raise

def initialize_rag_chain(vector_store, model_name='llama3.2:1b'):
    """Initialize the ConversationalRetrievalChain with Ollama LLM."""
    print("Initializing RAG chain with Ollama LLM...")
    print("Using CPU mode for Ollama to avoid buffer errors.")
    try:
        llm = OllamaLLM(model=model_name, num_gpu=0, base_url="http://127.0.0.1:11434")
        # Test LLM with a simple prompt
        test_response = llm.invoke("Say 'test'")
        print(f"LLM test successful: {test_response[:20]}...")
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        raise Exception(f"Failed to initialize Ollama LLM: {str(e)}")
    
    condense_question_prompt = PromptTemplate.from_template(
        """Given the following conversation history and a follow-up question, rephrase the follow-up question into a clear, concise, and standalone question that preserves the original intent and is answerable based on the provided document context. Avoid adding information not present in the conversation or question.

        Chat History: {chat_history}
        Follow-up Input: {question}
        Standalone question:"""
    )
    answer_prompt = PromptTemplate.from_template(
        """Using the following context from the provided documents, answer the question accurately and concisely, prioritizing information from the context. If the context does not contain enough information to fully answer the question, supplement with a brief general explanation based on common knowledge about the topic, clearly stating when you are supplementing the context. Cite relevant parts of the context where applicable.

        Context: {context}
        Question: {question}
        Answer:"""
    )
    try:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            condense_question_prompt=condense_question_prompt,
            combine_docs_chain_kwargs={"prompt": answer_prompt},
            return_source_documents=True,
            verbose=False
        )
        print("RAG chain initialized successfully.")
        return qa_chain
    except Exception as e:
        print(f"Error initializing RAG chain: {str(e)}")
        raise Exception(f"Failed to initialize RAG chain: {str(e)}")

def answer_question(qa_chain, question, chat_history):
    """Generate an answer to a question using the RAG chain."""
    try:
        result = qa_chain({"question": question, "chat_history": chat_history})
        return result['answer'], result['source_documents']
    except Exception as e:
        print(f"Error answering question: {str(e)}")
        raise