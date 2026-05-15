import os
import sys
import glob
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def compute_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file to uniquely identify it."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def update_db(db_dir: str = None):
    if db_dir is None:
        db_dir = os.getenv("CHROMA_DB_DIR", "./chroma_db")
        
    print(f"Scanning directory '{db_dir}' for .txt files...")
    txt_files = glob.glob(os.path.join(db_dir, "*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in '{db_dir}'.")
        return

    print("Initializing HuggingFace embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load the existing DB
    db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    
    for file_path in txt_files:
        print(f"\nChecking {file_path}...")
        
        # We will use the file hash to determine the document ID, 
        # to avoid re-ingesting the exact same content.
        file_hash = compute_hash(file_path)
        base_id = file_hash[:16] # Use first 16 chars of hash
        
        # Check if this document already exists (simplified check by querying the first chunk)
        first_chunk_id = f"{base_id}_0"
        
        existing_docs = db.get(ids=[first_chunk_id])
        if existing_docs and existing_docs.get('ids') and len(existing_docs['ids']) > 0:
            print(f"  -> Already in the database. Skipping.")
            continue

        print(f"  -> New or modified file. Loading...")
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
        except Exception as e:
            print(f"  -> Failed to load {file_path}: {e}")
            continue

        if not documents:
            continue

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            print(f"  -> No chunks generated for {file_path}. Skipping.")
            continue
        
        # Assign unique IDs based on the hash and chunk index
        ids = [f"{base_id}_{i}" for i in range(len(chunks))]

        print(f"  -> Adding {len(chunks)} chunks to ChromaDB...")
        db.add_documents(documents=chunks, ids=ids)
    
    print("\n✅ Database update complete!")

if __name__ == "__main__":
    load_dotenv()
    update_db()
