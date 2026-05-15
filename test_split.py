from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader('./chroma_db/api_podlychenie.txt', encoding='utf-8')
docs = loader.load()
print(f"Loaded {len(docs)} documents.")
print(f"Doc 0 length: {len(docs[0].page_content) if docs else 0}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_documents(docs)
print(f"Chunks generated: {len(chunks)}")
