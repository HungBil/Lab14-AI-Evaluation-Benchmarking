# Báo cáo Cập nhật Cá nhân - Lab 14

**Họ và tên:** Huỳnh Văn Nghĩa
**MSSV:** 2A202600085
---

## 1. Đóng góp Kỹ thuật (Engineering Contribution)

- **Xây dựng Data Ingestion Pipeline & Vector Database (MỚI):**
  - **Thu thập và chuẩn hóa dữ liệu thô:** Lên kịch bản và tổng hợp 5 tài liệu nghiệp vụ thực địa (`access_control_sop.txt`, `hr_leave_policy.txt`, `it_helpdesk_faq.txt`, `policy_refund_v4.txt`, `sla_p1_2026.txt`) và lưu trữ làm Knowledge Base cốt lõi tại `data/docs`.
  - **Thiết kế quy trình xử lý văn bản tự động hóa:** Chịu trách nhiệm khởi tạo thư mục `data/processing_data/`. 
    - Áp dụng chiến lược **Semantic Chunking kết hợp Đệ quy (Recursive)** tại `chunking.py`: Nhận diện và cắt chia văn bản theo Header (`===`) để bảo toàn ngữ cảnh (Context integrity). Với các đoạn quá dài, tự động chia cắt tiếp theo chiều dài 1200 ký tự và dán đè đoạn nối (overlap) 240 ký tự. Hệ thống tự động kiểm soát gán thẻ ID theo format `doc_chunk_xxx`.
    - Sử dụng mô hình nhúng mới nhất của OpenAI (`text-embedding-3-small`) qua `OpenAIEmbeddingFunction` tại module `embedding.py`.
    - Đẩy mượt mà dữ liệu và khởi tạo instance Vector DB (ChromaDB) chuyên biệt, lưu trữ dưới local database `data/chroma_db` (collection: `golden_docs`).
- **Hoàn thiện Module Synthetic Data Generation (SDG):** 
  - Khởi tạo script `data/synthetic_gen.py` sinh >50 Test Cases tiêu chuẩn (Golden Dataset). Tất cả bao gồm đầy đủ các trường: `question`, `expected_answer`, `context`, và trọng tâm nhất là `expected_retrieval_ids` làm mỏ neo ground truth.
  - Đáp ứng yêu cầu kiểm thử mở rộng (Red Teaming) với 4 loại: Thông thường, Adversarial - hack AI, Edge - thiếu ngữ cảnh, multi-turn.
- **Phát triển Module Retrieval Evaluator:**
  - Hoàn thiện module cốt lõi `engine/retrieval_eval.py`. Triển khai `evaluate_batch()` để đối chiếu `expected_retrieval_ids` gốc so với `retrieved_ids` thực tế trả ra từ RAG agent.
  - Log và tính chính xác 2 thước đo vô cực quan trọng: **Hit Rate** và **MRR**. Đồng bộ hoá việc export tự động điểm số sang `reports/summary.json` phục vụ check lab.

---

## 2. Chiều sâu Kỹ thuật (Technical Depth)

- **Trade-off trong thiết kế Chunking Pipeline:** 
  - Rất nhiều hệ thống RAG đổ vỡ ngay bước tìm kiếm do cắt chuỗi Fixed-size mù quáng khiến một khái niệm luật bị tách thành 2 nửa mất mạch lạc. Việc sử dụng *Semantic Chunking* theo Header Section (`===`) phối hợp *Overlap* 240 char giúp Retrieval engine có góc nhìn đầy đủ theo cụm chủ đề, đại từ nhân xưng nhắc lại không bị trượt bối cảnh.
- **Nắm vững Metrics - MRR và Hit Rate trong RAG:** 
  - *Hit Rate* trả lời câu hỏi: Tài liệu cần thiết có lọt vào tập kết quả không?
  - *MRR (Mean Reciprocal Rank)* đo đạc độ hiệu quả của vị trí Ranking. Vì Gen AI rất dễ bị hiệu ứng "Lost in the Middle" (nếu tài liệu đúng nằm rank 10 thì AI sẽ quên ngay tắp lự). MRR cao chứng minh thuật toán Embedding của file `embedding.py` chạy tốt, khả năng kéo doc chuẩn xác tuyệt đối lên Top 1 rất bền vững.
- **Tối ưu Cost/Performance:** 
  - Thay vì dùng `text-embedding-ada-002`, `text-embedding-3-small` vừa giảm thiểu dung lượng db, tối thiểu hoá chi phí và tăng hiệu năng tìm kiếm tiếng Việt. Model mini (`gpt-4o-mini`) được dùng để tự động cày cuốc gen Data SDG số lượng lớn.

---

## 3. Giải quyết vấn đề (Problem Solving)

- **Quản lý Vector State trùng lặp trên ChromaDB Local:** 
  - *Vấn đề:* Trong quá trình tuning độ dài Chunk file `process.py`, việc chạy lại liên tục sinh ra lượng chunk rác chèn xếp lên nhau trên cùng collection của ChromaDB, gây ngập lụt kết quả Retrieve khiến MRR bị kéo thê thảm.
  - *Cách khắc phục:* Bổ sung cơ chế vòng loại ngoại lệ tự động `try... chroma_client.delete_collection("golden_docs")` ở `embedding.py`. Từ đây DB luôn được dọn sach dứt điểm (fresh copy) mỗi khi chạy lại script, bảo toàn độ tinh sạch của Vector Search.
- **Sự cố Rate Limit Async OpenAI:** 
  - *Vấn đề:* Request SDG đâm 50 luồng đồng thời gây lỗi `HTTP 429`.
  - *Cách khắc phục:* Đã bổ sung logic Semaphore / quản lý batch-size tại Async Request, đảm bảo vừa cân bằng tiêu chí thời gian (< 2 phút) lại không làm nghẽn cổ chai rate-limit.
- **Kiểm soát Data Pipeline Consistency Mạch lạc:** 
  - Do vận hành xuyên suốt "Raw Txt → Chunk → Vector DB" và sau đó đo lường "Expected ID vs Retrieved ID", em đã làm chủ luồng định danh. `doc_chunk_xxx` được thống nhất sinh ra tại `chunking.py`, chèn trơn tru vào ChromaDB và cung cấp như định chuẩn Ground Truth cho prompt của `synthetic_gen.py`. Nó triệt tiêu hoàn toàn lỗi không map được ID khi chạy Retrieval Evaluator.
