# Individual Reflection Report - Lab Day 14 (AI Evaluation Benchmarking)

- Họ và tên: Khuất Văn Vương
- Mã sinh viên: 2A202600087

## 1. Engineering Contribution (15 điểm)

### 1.1 Module đã đóng góp

Trong bài lab này, tôi tập trung vào các module kỹ thuật lõi của hệ thống đánh giá:

1. `agent/main_agent.py`

- Triển khai 2 phiên bản agent để benchmark regression:
  - `v1_random`: baseline cố tình retrieval sai (để tạo mốc so sánh thấp, có kiểm soát).
  - `v2_hybrid`: retrieval cải thiện bằng hybrid scoring (dense + sparse fusion theo RRF).
- Chuẩn hóa output của agent để phục vụ evaluator: `answer`, `retrieved_ids`, `metadata.sources`.

2. `engine/retrieval_eval.py`

- Hoàn thiện tính toán retrieval metrics cho từng case:
  - Hit Rate@k: có ít nhất 1 ground-truth ID trong top-k kết quả retrieval.
  - MRR: nghịch đảo vị trí tài liệu liên quan đầu tiên.
- Bổ sung hàm `score()` để tích hợp trực tiếp vào pipeline benchmark cho mỗi test case.

3. `engine/llm_judge.py`

- Xây dựng multi-judge consensus với 2 judge model:
  - Judge A: GPT-4o (remote).
  - Judge B: local Phi-3-mini-4k-instruct-GGUF.
- Xây dựng logic xử lý kết quả:
  - Parse JSON robust (kể cả output có text thừa).
  - Conflict resolution khi 2 judge lệch điểm.
  - Tính `agreement_rate` và `individual_scores`.
  - Tính Cohen's Kappa ở tầng tổng hợp để đo độ đồng thuận toàn batch.

4. `engine/runner.py`

- Bổ sung log tiến trình theo batch/case trong async runner:
  - Hiển thị phần trăm hoàn thành và thời gian elapsed.
  - Giúp phân biệt rõ trạng thái đang chạy và đang treo I/O.

5. `main.py`

- Tích hợp đầy đủ pipeline benchmark V1 vs V2.
- Tạo summary metrics và release gate dựa trên delta chất lượng + retrieval quality.
- Xuất `reports/summary.json` và `reports/benchmark_results.json`.

### 1.2 Bằng chứng kỹ thuật

Các file chỉnh sửa chính:

- `agent/main_agent.py`
- `engine/retrieval_eval.py`
- `engine/llm_judge.py`
- `engine/runner.py`
- `main.py`
- `data/golden_set.jsonl`

## 2. Technical Depth (15 điểm)

### 2.1 MRR và ý nghĩa

- MRR (Mean Reciprocal Rank) phản ánh tốc độ tìm đúng tài liệu liên quan trong ranking retrieval.
- Nếu tài liệu đúng đứng ở top 1, MRR = 1.0; top 2 thì 0.5; không có thì 0.
- Vì vậy MRR nhạy với chất lượng ranking hơn Hit Rate (Hit Rate chỉ biết có/không).

### 2.2 Cohen's Kappa và độ tin cậy judge

- Agreement thô có thể cao do trùng ngẫu nhiên.
- Cohen's Kappa loại trừ phần đồng thuận ngẫu nhiên theo phân phối nhãn, nên phản ánh đúng hơn độ nhất quán giữa 2 judge.
- Trong pipeline, Kappa giúp đánh giá reliability của hệ thống chấm điểm tự động.

### 2.3 Position Bias

- Position bias xảy ra khi model chấm bị thiên vị câu trả lời ở vị trí A/B thay vì chất lượng nội dung.
- Cách kiểm tra là hoán đổi vị trí hai response và so sánh score delta.
- Nếu delta lớn, cần điều chỉnh prompt hoặc phương pháp chấm để giảm bias.

### 2.4 Trade-off chi phí - chất lượng

- Dùng 2 judge giúp tăng độ tin cậy nhưng làm tăng latency/cost.
- Phiên bản local judge (Phi-3 GGUF) giảm phụ thuộc API, nhưng cần tối ưu tài nguyên local và ổn định parser/output.
- Batch async tăng throughput, nhưng cần quan sát I/O và rate-limit để tránh treo batch.

## 3. Problem Solving (10 điểm)

### 3.1 Vấn đề gặp phải

1. Pipeline bị kẹt lâu ở giữa batch dù không crash.
2. JSON parse lỗi kiểu `Extra data` do output judge trả thêm text.
3. Conflict khi merge `main.py`.
4. Dataset ban đầu chưa chuẩn hóa hoàn toàn theo guide (thiếu technical constraints, thiếu `question` cho multi-turn, có rule mâu thuẫn).
5. Lỗi môi trường (`_cffi_backend`) và lỗi gọi provider judge.

### 3.2 Cách xử lý

1. Thêm progress logs trong runner để quan sát chính xác batch/case đang chạy.
2. Viết parser JSON an toàn (`raw_decode` + bỏ code fences) để chống output nhiễu.
3. Resolve conflict `main.py` theo hướng giữ cấu trúc khung bài nhưng dùng logic judge/eval thật.
4. Chuẩn hóa dataset:

- Bổ sung `question` cho multi-turn case.
- Sửa các `must_not_contain` mâu thuẫn với `expected_answer`.
- Thêm nhóm `technical_constraint` gồm `latency_stress` và `cost_efficiency`.

5. Chuyển judge thứ 2 từ Gemini sang local Phi-3 GGUF để giảm phụ thuộc mạng/API.

### 3.3 Bài học rút ra

- Với pipeline async nhiều API calls, khả năng quan sát (observability) quan trọng không kém thuật toán.
- Dữ liệu benchmark cần nhất quán schema và tiêu chí chấm để tránh false fail.
- Multi-judge chỉ có giá trị khi parse/output ổn định và có chỉ số đồng thuận định lượng.

## 4. Kế hoạch cải tiến tiếp theo

1. Bổ sung timeout và retry cho từng judge call để tránh treo vô hạn.
2. Thêm cơ chế cache kết quả judge để giảm chi phí chạy lại.
3. Mở rộng evaluation cho safety và abstention accuracy ở các case adversarial/out-of-context.
4. Bổ sung dashboard trực quan (pass/fail clusters, latency histogram, cost breakdown).

## 5. Tự đánh giá theo rubric cá nhân

- Engineering Contribution: 14/15
- Technical Depth: 13/15
- Problem Solving: 9/10
- Tổng tự đánh giá: 36/40
