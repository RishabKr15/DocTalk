

from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="llama3.2:1b", num_gpu=0, base_url="http://127.0.0.1:11434")
print(llm.invoke("Hello, how are you?"))