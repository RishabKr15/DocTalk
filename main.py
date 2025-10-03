import subprocess
import os
import tempfile
import sys
import re
import logging
from typing import List, Tuple, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Set up logging for better debugging and production readiness
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_availability() -> Tuple[bool, Optional[int], Optional[int]]:
    """Check if a GPU is available and Ollama is configured to use it.
    
    Prioritizes CPU mode for stability, especially on low-VRAM GPUs.
    """
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        stdout = result.stdout
        if "NVIDIA" in stdout or "Driver Version" in stdout:
            match = re.search(r"(\d+)\s*MiB\s*/\s*(\d+)\s*MiB", stdout)
            if match:
                used_vram = int(match.group(1))
                total_vram = int(match.group(2))
                logger.info(f"GPU detected: {total_vram}MiB total VRAM, {used_vram}MiB used.")
                if total_vram <= 4096:
                    logger.warning("4GB or less VRAM detected. Forcing CPU mode to avoid OOM errors.")
                    return False, total_vram, used_vram
                return True, total_vram, used_vram
            else:
                logger.warning("GPU detected, but VRAM details could not be parsed. Falling back to CPU mode.")
                return False, None, None
        return False, None, None
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("No NVIDIA GPU detected or nvidia-smi not found. Falling back to CPU.")
        return False, None, None

def install_ollama(preferred_model: str = "llama3.2:1b") -> str:
    """Check if Ollama is installed and ensure a suitable model is available.
    
    Tries the preferred model first, then falls back to alternatives. Uses CPU mode for reliability.
    Updated for 2025: Prioritizes newer Llama variants if available.
    """
    try:
        subprocess.run(["ollama", "--version"], check=True, capture_output=True)
        logger.info("Ollama is installed.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Ollama not found. Please install from https://ollama.com/download")
        sys.exit(1)
    
    # Test Ollama server connectivity early
    try:
        test_llm = OllamaLLM(model=preferred_model, num_gpu=0, base_url="http://127.0.0.1:11434")
        test_llm.invoke("test")
        logger.info("Ollama server is running.")
    except Exception as e:
        logger.error(f"Ollama server not accessible: {e}. Please run 'ollama serve' in a terminal.")
        sys.exit(1)
    
    # Updated models list for 2025: Include potential newer releases like Llama 4 if available, but fallback to 3.2
    models_to_try = [preferred_model] + ["llama3.2:3b", "llama3.2:latest", "llama3:8b"]  # llama3 as broader fallback
    
    for model_name in models_to_try:
        try:
            # Check if model is available
            result = subprocess.run(["ollama", "list"], check=True, capture_output=True, text=True)
            if model_name in result.stdout:
                logger.info(f"Model {model_name} is already available.")
            else:
                logger.info(f"Pulling model {model_name}...")
                subprocess.run(["ollama", "pull", model_name], check=True, capture_output=True)
                logger.info(f"Model {model_name} pulled successfully.")
            
            # Test model accessibility in CPU mode
            llm = OllamaLLM(model=model_name, num_gpu=0, base_url="http://127.0.0.1:11434")
            llm.invoke("test")
            logger.info(f"Model {model_name} is accessible in CPU mode.")
            return model_name
            
        except Exception as e:
            logger.warning(f"Error with model {model_name}: {e}")
            continue
    
    logger.error("Could not pull or access any model. Please check your Ollama installation.")
    sys.exit(1)

def get_ollama_instructions() -> str:
    """Return instructions for installing and starting Ollama. Updated for cross-platform in 2025."""
    return """
    ### Install and Start Ollama (Updated for 2025)
    **Windows:**
    1. Download from [ollama.com/download](https://ollama.com/download).
    2. Run installer.
    3. Terminal: `ollama serve`
    4. New terminal: `ollama pull llama3.2:1b`
    5. CPU mode: `set OLLAMA_NUM_GPU=0 && ollama run llama3.2:1b`

    **macOS/Linux:**
    1. `curl -fsSL https://ollama.com/install.sh | sh`
    2. `ollama serve`
    3. `ollama pull llama3.2:1b`
    4. CPU: `OLLAMA_NUM_GPU=0 ollama run llama3.2:1b`

    Refresh page after setup.
    """

def load_documents(pdf_file) -> List[Document]:
    """Load and split PDF documents into chunks for better retrieval."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name
        pdf_reader = PyPDFLoader(tmp_file_path)
        documents = pdf_reader.load()
        # Improved chunking: Larger chunks for 2025-era models with better context windows
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Increased for deeper context
            chunk_overlap=100,  # More overlap for continuity
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        os.unlink(tmp_file_path)
        logger.info(f"Loaded {len(documents)} pages into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        raise

def create_vector_store(chunks: List[Document], persist_dir: Optional[str] = None) -> FAISS:
    """Create or load a FAISS vector store from document chunks. Supports persistence for efficiency."""
    try:
        if persist_dir and os.path.exists(os.path.join(persist_dir, "index.pkl")):
            logger.info(f"Loading existing vector store from {persist_dir}.")
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},  # Explicit CPU for consistency
            )
            vector_store = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
            logger.info("Vector store loaded successfully.")
        else:
            # Updated embedding model: Use a more robust one if available, fallback to original
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},  # Explicit CPU for consistency
                )
                logger.info("Using 'all-MiniLM-L6-v2' embeddings.")
            except Exception as emb_e:
                logger.warning(f"Failed to load 'all-MiniLM-L6-v2': {emb_e}. Falling back to 'sentence-transformers/all-MiniLM-L6-v2'.")
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                )
            
            vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
            logger.info("Vector store created successfully.")
            
            if persist_dir:
                os.makedirs(persist_dir, exist_ok=True)
                vector_store.save_local(persist_dir)
                logger.info(f"Vector store persisted to {persist_dir}.")
        
        return vector_store
    except Exception as e:
        logger.error(f"Error creating/loading vector store: {str(e)}")
        raise

def initialize_rag_chain(vector_store: FAISS, model_name: str = 'llama3.2:1b') -> ConversationalRetrievalChain:
    """Initialize the ConversationalRetrievalChain with Ollama LLM and enhanced prompts."""
    # Ensure model is ready (idempotent)
    working_model = install_ollama(model_name)
    logger.info(f"Using model: {working_model} in CPU mode.")
    
    try:
        llm = OllamaLLM(
            model=working_model,
            num_gpu=0,
            base_url="http://127.0.0.1:11434",
            temperature=0.05,  # Very low for factual, consistent outputs in 2025 standards
        )
        llm.invoke("Test: OK")  # Simple test
        logger.info("LLM ready.")
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        raise
    
    # Enhanced condense prompt: More precise for multi-turn conversations
    condense_question_prompt = PromptTemplate.from_template(
        """You are an expert at reformulating questions for document-based QA. 
        Using the chat history, rephrase the follow-up into a standalone question that captures the full intent, 
        is concise, and relies only on history/question details. Output only the rephrased question.

        Chat History: {chat_history}
        Follow-up: {question}
        Rephrased Question:"""
    )
    
    # Enhanced answer prompt: Stresses structure, citations, and anti-hallucination
    answer_prompt = PromptTemplate.from_template(
        """You are a meticulous QA assistant grounded in the given document context. 
        Answer the question using ONLY the context—be accurate, concise, and use bullet points/tables if helpful. 
        If info is missing, say: "Insufficient context to answer fully." No speculation or external facts.

        Context: {context}
        Question: {question}
        Answer:"""
    )
    
    try:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.7},  # Top 5 with relevance threshold for quality
            ),
            condense_question_prompt=condense_question_prompt,
            combine_docs_chain_kwargs={"prompt": answer_prompt},
            return_source_documents=True,
            verbose=False,
        )
        logger.info("RAG chain initialized.")
        return qa_chain
    except Exception as e:
        logger.error(f"Error initializing RAG chain: {str(e)}")
        raise

def answer_question(qa_chain: ConversationalRetrievalChain, question: str, chat_history: List[Tuple[str, str]]) -> Tuple[str, List[Document]]:
    """Generate an answer using the RAG chain, with chat history support."""
    try:
        result = qa_chain({"question": question, "chat_history": chat_history})
        answer = result['answer']
        sources = result['source_documents']
        logger.info(f"Answered with {len(sources)} sources.")
        return answer, sources
    except Exception as e:
        logger.error(f"Error answering: {str(e)}")
        raise