# DocTalk RAG Q&A with Ollama 

A local Retrieval-Augmented Generation (RAG) system for querying PDF documents using natural language. Built with Streamlit for the UI, LangChain for orchestration, Ollama for lightweight local LLM inference (Llama 3.2), and FAISS for efficient vector search. Everything runs offline on your machine—no cloud APIs required.
Current as of October 2025: Optimized for Llama 3.2 models, with CPU/GPU detection and improved prompts for factual, hallucination-free responses.
<img width="1920" height="900" alt="PreView" src="https://github.com/user-attachments/assets/ce50f929-96c8-4c53-a034-febfb23578f6" />

🚀 Features

* Upload & Query PDFs: Process any PDF into searchable chunks and ask conversational questions.
* Local LLM: Uses Ollama (e.g., llama3.2:1b) for privacy-focused inference. Falls back to alternatives like llama3.2:3b.
* Vector Embeddings: Hugging Face's all-MiniLM-L6-v2 for fast, lightweight text vectorization.
* Conversational Memory: Maintains chat history for follow-up questions.
* Source Citations: Displays relevant document snippets for transparency.
* Robust Setup: Auto-installs/pulls models, checks GPU/VRAM, and uses CPU mode for stability.
* Persistence: Vector stores can be saved to disk for quick reloads.

## 📋 Prerequisites

* Python 3.10+: Tested on 3.12.
* Ollama: Download from ollama.com. Start the server with ollama serve.
* NVIDIA GPU (optional): For acceleration; falls back to CPU if VRAM <4GB or unavailable.
* Dependencies: Installed via pip (see below).

## 🛠️ Installation

* Clone or Download:
* textgit clone <your-repo-url>


### Install Python Dependencies:
* textpip install -r requirements.txt
* requirements.txt contents:
* textstreamlit==1.38.0
* langchain==0.3.1
* langchain-community==0.3.1
* langchain-ollama==0.2.2
* langchain-huggingface==0.1.0
* faiss-cpu==1.8.0  # Use faiss-gpu for NVIDIA
* pypdf==5.1.0
* sentence-transformers==3.1.1

## Setup Ollama:

* Install Ollama (see instructions).
* Run ollama serve in a terminal.
* The app will auto-pull llama3.2:1b (or alternatives) on first run.


## Run the App:
* textstreamlit run app.py
* Open http://localhost:8501 in your browser.

## 📖 Usage

* Launch the App: Run streamlit run app.py.
* Upload a PDF: Select a file via the uploader. The app processes it into chunks and builds a vector index (takes ~10-60s depending on size).
* Ask Questions: Enter a query (e.g., "What is the main topic?") and hit Submit.
* View Responses: Get grounded answers with source previews. Chat history is preserved for follow-ups.
* Reset: Use "Clear PDF" or "Clear Chat" buttons as needed.

## Example Workflow

* Upload: research_paper.pdf
* Question: "Summarize the methodology."
* Answer: Factual summary from relevant sections, with expandable sources.

## Customizing

* Base URL: Edit base_url in main.py for remote Ollama (e.g., Docker: http://host.docker.internal:11434).
* Model: Change preferred_model in install_ollama() to llama3.2:3b for better quality (needs more RAM).
* Chunk Size: Adjust chunk_size in load_documents() for longer/shorter contexts.

## 🔧 Troubleshooting

* Ollama Errors: Ensure ollama serve is running. Check logs for model pulls. Use CPU mode if GPU OOM: Set OLLAMA_NUM_GPU=0.
* PDF Processing Fails: Verify PDF isn't encrypted/corrupted. Test with a simple file.
* No GPU Detected: Install NVIDIA drivers/CUDA; code auto-falls back to CPU.
* Slow Embeddings: Use faiss-gpu for NVIDIA; increase chunk_size for fewer vectors.
* Connection Refused: Confirm Ollama on port 11434: curl http://127.0.0.1:11434/api/tags.
* Logs: Check terminal for logger output (INFO level).

## See main.py for backend details or LangChain Docs for RAG tweaks.
🤝 Contributing

## Fork the repo.
* Create a feature branch (git checkout -b feature/amazing-feature).
* Commit changes (git commit -m 'Add amazing feature').
* Push (git push origin feature/amazing-feature).
* Open a Pull Request.

## Report issues for bugs or enhancements!
* MIT License—use freely, attribute if you like.
* 🙏 Acknowledgments

* Streamlit for the slick UI.
* LangChain for RAG magic.
* Ollama for local LLMs.
* Hugging Face for embeddings.

Built with ❤️ for offline AI in 2025. Questions? Open an issue!
