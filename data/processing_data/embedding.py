import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

load_dotenv()

DB_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

def store_into_vector_db(chunks: List[Dict[str, Any]], collection_name: str = "golden_docs"):
    """
    Embedding function dùng `text-embedding-3-small` OpenAI.
    Lưu dữ liệu trực tiếp vào Vector Database (ChromaDB).
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("Không tìm thấy OPENAI_API_KEY trong file .env!")

    print(f"✅ Đang lưu các chunks vào Vector DB tại: {DB_DIR}...")
    
    # Init embed config
    openai_ef = OpenAIEmbeddingFunction(
        api_key=openai_key,
        model_name="text-embedding-3-small"
    )

    # Khởi tạo DB Local
    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
    
    # Xoá nếu bị trùng tên test db
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass
        
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=openai_ef
    )
    
    ids = [c["chunk_id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [c["metadata"] for c in chunks]
    
    collection.add(ids=ids, documents=docs, metadatas=metas)
    print(f"✅ Đã dump thành công {len(chunks)} embedded-chunks vào Collection '{collection_name}'.")
    return collection
