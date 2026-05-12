from langchain_core.documents import Document

class MockRetriever:
    def invoke(self, query: str):
        # Return mock documents
        return [
            Document(page_content=f"Mock document 1 containing info about: {query}"),
            Document(page_content=f"Mock document 2 with further details on: {query}")
        ]

def get_retriever():
    """
    Initializes and returns the mock ChromaDB retriever.
    Replace this with actual ChromaDB setup later.
    """
    return MockRetriever()

def init_mock_db():
    """
    Populates the database with some initial mock support data.
    """
    print("Mock DB initialized (using in-memory MockRetriever for now).")
