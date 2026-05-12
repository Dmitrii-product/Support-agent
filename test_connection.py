import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import chromadb

def test_groq():
    print("Testing Groq connection...")
    try:
        model_name = os.getenv("GROQ_MODEL", "qwen-2.5-32b")
        llm = ChatGroq(model=model_name, temperature=0, max_tokens=10)
        # Send a simple ping
        response = llm.invoke([HumanMessage(content="Reply with exactly the word 'OK'")])
        print(f"✅ Groq connection successful! Model '{model_name}' responded with: {response.content}")
    except Exception as e:
        print(f"❌ Groq connection failed: {e}")

def test_chromadb():
    print("\nTesting ChromaDB connection...")
    try:
        db_dir = os.getenv("CHROMA_DB_DIR", "./chroma_db")
        client = chromadb.PersistentClient(path=db_dir)
        # Try to create or get a collection to verify it's working
        collection = client.get_or_create_collection(name="test_collection")
        print(f"✅ ChromaDB connection successful! PersistentClient created at '{db_dir}' with collection '{collection.name}'.")
    except Exception as e:
        print(f"❌ ChromaDB connection failed: {e}")

if __name__ == "__main__":
    load_dotenv()
    print("Starting Connection Tests...\n")
    
    # Check if API key is loaded
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️ Warning: GROQ_API_KEY not found in environment variables.")
        
    test_groq()
    test_chromadb()
    print("\nTests finished.")
