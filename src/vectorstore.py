import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_retriever():
    """
    Initializes and returns the ChromaDB retriever.
    """
    db_dir = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 3})

def init_mock_db():
    """
    Deprecated. Use populate_db.py instead.
    """
    pass
