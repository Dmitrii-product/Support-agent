import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def populate_chromadb():
    print("Loading documents from chroma_db directory...")
    db_dir = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    txt_files = glob.glob(os.path.join(db_dir, "*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in '{db_dir}'. Please ensure they end in .txt")
        return

    documents = []
    for file_path in txt_files:
        print(f"Loading {file_path}...")
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

    if not documents:
        print("No documents loaded. Exiting.")
        return

    print(f"Loaded {len(documents)} documents. Splitting into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    print("Initializing HuggingFace embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"Saving chunks to ChromaDB at {db_dir}...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )
    
    print("✅ Database successfully populated and persisted!")

if __name__ == "__main__":
    load_dotenv()
    populate_chromadb()
