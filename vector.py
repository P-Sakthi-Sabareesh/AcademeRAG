'''
milvus_uri = "./milvus_local.db"
collection_name = "AcademeRAG"
'''

from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
import os
from pypdf import PdfReader

# List of PDF file paths
pdf_paths = ["Project/RAG project - Sample PDF.pdf", "Project/RAG project sample pdf 2.pdf"]

def pdfs_to_chunks(pdf_paths, chunk_size=1000):
    all_chunks = []
    for path in pdf_paths:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        # Split text into chunks
        for i in range(0, len(text), chunk_size):
            all_chunks.append(text[i:i+chunk_size])
    return all_chunks

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
milvus_uri = "./milvus_local.db"
collection_name = "AcademeRAG"

add_documents = not os.path.exists(milvus_uri)

vector_store = Milvus(
    collection_name=collection_name,
    embedding_function=embeddings,
    connection_args={"uri": milvus_uri},
)

if add_documents:
    chunks = pdfs_to_chunks(pdf_paths)
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_store.add_documents(documents, drop_old=True)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
