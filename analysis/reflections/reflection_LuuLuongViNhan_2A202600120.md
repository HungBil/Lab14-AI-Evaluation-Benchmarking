# 📝 Individual Reflection Report

**Họ và tên:** Lưu Lương Vi Nhân  
**Mã sinh viên:** 2A202600120  
**Vai trò:** Data Engineer: Golden Dataset & Synthetic Data Generation  
**File phụ trách chính:** `data/synthetic_gen.py`  
**Output:** `data/golden_set.jsonl` (55 cases)

---

## 1. Đóng góp kỹ thuật (Engineering Contribution)

### 1.1. Thiết kế và xây dựng pipeline sinh dữ liệu benchmark
Tôi đã xây dựng toàn bộ script `data/synthetic_gen.py` từ đầu để tự động sinh ra bộ **Golden Dataset** phục vụ đánh giá AI Agent. Pipeline hoạt động theo các bước:

1. **Tự động hoá Corpus từ tài liệu thực tế:** Thay vì dùng văn bản mẫu tĩnh (hardcode), tôi đã tích hợp hàm `chunk_document()` từ module `data/processing_data/chunking.py` — module dùng để đưa dữ liệu vào ChromaDB. Script sẽ tự động đọc tất cả file `.txt` trong `data/docs/`, cắt thành 34 chunk với ID tăng dần (`doc_001` đến `doc_034`), và ghép thành `RAW_TEXT_CORPUS` hoàn chỉnh. Điều này đảm bảo tính nhất quán giữa tập dữ liệu benchmark và dữ liệu thực trong Vector DB.

2. **Sinh câu hỏi theo 4 loại chuyên biệt:** Tôi thiết kế 4 prompt riêng biệt với few-shot examples chi tiết để gọi OpenAI API sinh các câu hỏi đa dạng, phân bổ cụ thể:
   - **18 Normal cases:** 12 fact_check + 6 comparison (đối chiếu ≥2 tài liệu)
   - **14 Adversarial cases:** 5 prompt_injection + 5 goal_hijacking + 4 jailbreak
   - **15 Edge cases:** 5 out_of_context + 5 ambiguous + 5 conflicting
   - **8 Multi-turn cases:** 5 carry_over + 3 correction
   - **Tổng kết:** 55 cases sau deduplication (vượt yêu cầu tối thiểu 50 cases)

3. **Schema chuẩn 5 trường bắt buộc:** Mỗi test case được sinh ra đều đảm bảo có đủ 5 trường quan trọng, trong đó đặc biệt là trường `expected_retrieval_ids` — trường này là cơ sở để Người 2 tính toán **Hit Rate** và **MRR** cho hệ thống Retrieval.

4. **Cơ chế validation và deduplication:** Tôi xây dựng hàm `validate_item()` kiểm tra schema đầu ra (đủ trường, kiểu dữ liệu, `doc_id` hợp lệ, v.v.) và hàm `dedupe()` loại bỏ các câu hỏi trùng ý nhau bằng Jaccard similarity trước khi lưu vào file.

5. **Gọi API bất đồng bộ (Async) có retry:** Toàn bộ 4 nhóm câu hỏi được gọi song song bằng `asyncio.gather()`, giúp giảm thời gian sinh data từ ~4 phút xuống còn ~1 phút. Mỗi nhóm có cơ chế retry tối đa 3 lần khi API gặp lỗi.

### 1.2. Git Commits
- Commit: Khởi tạo structure và viết prompt cơ bản cho `synthetic_gen.py`
- Commit: Tích hợp `chunk_document` để thay thế RAW_TEXT_CORPUS tĩnh
- Commit: Thêm validation schema, deduplication và phân phối đồng đều 4 loại case
- Commit: Fix encoding lỗi unicode `→` trên Windows console

---

## 2. Hiểu biết kỹ thuật (Technical Depth)

### 2.1. Hit Rate và MRR — Tại sao `expected_retrieval_ids` quan trọng?

**Hit Rate** đo lường xem hệ thống Retrieval có tìm đúng tài liệu cần thiết hay không:
- Hit Rate = 1.0 nếu ít nhất 1 `expected_retrieval_id` xuất hiện trong top-K kết quả trả về.
- Hit Rate = 0.0 nếu không có doc nào đúng trong top-K.

**MRR (Mean Reciprocal Rank)** đo chất lượng **thứ hạng** của kết quả đúng:
- MRR = 1/vị_trí_đầu_tiên_của_doc_đúng. Ví dụ: doc đúng ở vị trí 1 → MRR = 1.0, ở vị trí 3 → MRR = 0.33.
- MRR cao hơn Hit Rate vì nó phân biệt được "tìm đúng nhưng xếp thứ 1" với "tìm đúng nhưng xếp thứ 5".

Tôi thiết kế `expected_retrieval_ids` bám sát chính xác `doc_id` của từng chunk trong corpus để đảm bảo tính chính xác khi chấm điểm Retrieval.

### 2.2. Các loại Adversarial và Edge Cases

| Loại | Mục đích kiểm tra |
|---|---|
| **Prompt Injection** | Agent có bị lừa bởi lệnh ẩn trong câu hỏi của user không? |
| **Goal Hijacking** | Agent có từ chối yêu cầu off-topic và redirect đúng cách không? |
| **Jailbreak** | Agent có bị "phá khoá" qua role-play hay hypothetical framing không? |
| **Out-of-context** | Agent có thành thật nói "Tôi không biết" hay sẽ ảo giác (hallucinate)? |
| **Ambiguous** | Agent có biết hỏi lại để làm rõ thay vì tự đoán mò không? |
| **Conflicting** | Agent có nhận biết được mâu thuẫn giữa các điều khoản trong tài liệu không? |
| **Multi-turn** | Agent có ghi nhớ ngữ cảnh qua nhiều lượt hội thoại không? |

### 2.3. Trade-off Chất lượng vs. Chi phí

Trong quá trình thực hiện, tôi cân nhắc việc dùng `gpt-4o-mini` vs `gpt-4o`:
- **gpt-4o-mini:** Nhanh hơn, rẻ hơn (~10x), nhưng đôi khi sinh câu hỏi adversarial thiếu sáng tạo và prompt injection quá lộ liễu, dễ bị phát hiện.
- **gpt-4o:** Tốn kém hơn nhưng sinh ra các câu hỏi jailbreak tinh vi hơn, câu hỏi multi-turn có hội thoại tự nhiên hơn, phù hợp hơn cho mục đích Red Teaming chuyên nghiệp.

Quyết định cuối: Dùng `gpt-4o` để đảm bảo chất lượng benchmark, vì đây là "đề thi" — dữ liệu kém chất lượng ở đây sẽ làm cho toàn bộ quá trình đánh giá phía sau bị sai lệch. Tốn 0.27$ :D

---

## 3. Xử lý vấn đề phát sinh (Problem Solving)

### Vấn đề 1: UnicodeEncodeError trên Windows
**Triệu chứng:** Script bị crash khi in ký tự `→` vì terminal Windows (cp1252) không encode được ký tự Unicode này.  
**Giải pháp:** Chạy script với biến môi trường `PYTHONIOENCODING=utf-8` và thay ký tự `→` bằng `->` trong code nguồn để tương thích với mọi môi trường.

### Vấn đề 2: LLM sinh thiếu số lượng yêu cầu
**Triệu chứng:** Yêu cầu 50 cases nhưng LLM chỉ sinh được 46 sau khi dedup.  
**Giải pháp:** Tăng target số lượng yêu cầu lên ~110% so với mục tiêu (55 cases thay vì 50) để sau khi loại bỏ các case không hợp lệ và trùng lặp vẫn đảm bảo đủ ≥50 cases.

### Vấn đề 3: Import module từ thư mục khác
**Triệu chứng:** Script `data/synthetic_gen.py` không thể import `data.processing_data.chunking` vì Python không nhận ra project root là package.  
**Giải pháp:** Dùng `sys.path.append(str(Path(__file__).resolve().parent.parent))` để thêm thư mục gốc của dự án vào Python path trước khi import.

---

## 4. Kết luận

Thông qua việc xây dựng bộ Golden Dataset này, tôi hiểu sâu hơn rằng **chất lượng của bộ "đề thi" quyết định giá trị của toàn bộ hệ thống đánh giá**. Một bộ benchmark kém — thiếu adversarial cases, thiếu `expected_retrieval_ids`, hoặc câu hỏi quá dễ — sẽ cho điểm số cao giả tạo và che giấu điểm yếu thực sự của Agent.

Điều tôi tâm đắc nhất là việc thiết kế các loại câu hỏi theo đúng phương pháp **Red Teaming**: không chỉ kiểm tra xem AI trả lời đúng hay sai, mà còn kiểm tra xem AI có bị lừa, có biết từ chối, và có biết thừa nhận giới hạn của mình hay không — đây mới là tiêu chuẩn của một AI evaluation chuyên nghiệp.
