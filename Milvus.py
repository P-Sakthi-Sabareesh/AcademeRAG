'''
milvus_uri = "https://cloud.zilliz.com/orgs/org-ylznfpfacshmpcchxifymn/projects/proj-4f83db2c30bfa02ee2c4d5/clusters/in03-200a60c94a793fc/collections/restaurant_reviews"

# Collection name inside Milvus
collection_name = "AcademeRAG"
'''

from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus
import os
from langchain_core.documents import Document

# Embeddings model to generate vector embeddings for text content
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Local Milvus database URI (using Milvus Lite local file-based DB)
milvus_uri = "https://cloud.zilliz.com/orgs/org-ylznfpfacshmpcchxifymn/projects/proj-4f83db2c30bfa02ee2c4d5/clusters/in03-200a60c94a793fc/collections/AcademeRAG"

# Collection name inside Milvus
collection_name = "AcademeRAG"

# Initialize the Milvus vector store with your embeddings and connection info
vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": milvus_uri},
    collection_name=collection_name,
)

# Create retriever interface for querying
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
