import os
from pathlib import Path

from chunking import chunk_document
from embedding import store_into_vector_db

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

def main():
    print("="*60)
    print("🚀 Bắt đầu tiến trình Chuẩn bị Dữ liệu")
    print("="*60)
    
    if not DOCS_DIR.exists():
        print(f"❌ Không tìm thấy thư mục {DOCS_DIR}")
        return
        
    all_chunks = []
    
    print("⏳ Đang tiến hành đọc và Semantic Chunking...")
    
    # Chunking
    for filepath in DOCS_DIR.glob("*.txt"):
        text = filepath.read_text(encoding="utf-8")
        chunks = chunk_document(text, filepath.name)
        all_chunks.extend(chunks)
        
    # Gắn Identifier
    for i, c in enumerate(all_chunks):
        c["chunk_id"] = f"doc_chunk_{i+1:03d}"
        
    print(f"✅ Chunking thành công! (Tạo ra tổng {len(all_chunks)} đoạn meta-data).")
    
    # Embedding DB
    print("\n⏳ Đang kết nối Embeddings và Push lên Vector DB (ChromaDB)...")
    store_into_vector_db(all_chunks, collection_name="golden_docs")
    
    print("\n🎉 Hoàn thành toàn bộ quy trình Data Pipeline. ChromaDB đã lưu dưới local!")

if __name__ == "__main__":
    main()
