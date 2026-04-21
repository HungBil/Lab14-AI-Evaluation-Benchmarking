# Báo cáo Cá nhân

**Họ và tên:** Nguyễn Đông Hưng  
**MSSV:** 2A202600392  
**Vai trò trong nhóm:** Thành viên số 5 - `Evaluate retrieve + judge`

## 1. Tôi được giao nhiệm vụ gì?
Tôi phụ trách phần đánh giá chất lượng retrieval và generation của agent. Mục tiêu chính là:
- triển khai retrieval evaluation thật bằng `Hit Rate` và `MRR`
- triển khai multi-judge để chấm chất lượng câu trả lời
- bổ sung chỉ số độ đồng thuận nâng cao giữa 2 judge
- tổng hợp benchmark thành report có thể nộp và so sánh giữa `V1` và `V2`

## 2. Tôi đã thực hiện những công việc cụ thể nào?
Tôi đã thực hiện các phần sau:
- triển khai `RetrievalEvaluator` để tính `Hit Rate` và `MRR` từ `expected_retrieval_ids` và `metadata.sources`
- triển khai `LLMJudge` để chấm song song bằng 2 model judge, lấy `final_score`, `agreement_rate`, và lưu reasoning của từng judge
- bổ sung `Cohen's Kappa` để đo độ đồng thuận giữa 2 judge sau khi loại trừ đồng thuận ngẫu nhiên
- sửa lỗi mismatch `doc_id` giữa benchmark dataset và agent retrieval output để retrieval metric phản ánh đúng hơn
- chuẩn hóa lại `summary.json` và `benchmark_results.json` theo format gần mẫu của giảng viên
- chạy benchmark end-to-end, phân tích lỗi, và tổng hợp báo cáo nhóm

## 3. Khó khăn lớn nhất tôi gặp là gì?
Khó khăn lớn nhất là pipeline benchmark chạy được nhưng số retrieval ban đầu ra `0.0` toàn bộ. Khi kiểm tra kỹ, tôi phát hiện `expected_retrieval_ids` dùng namespace `doc_***`, trong khi agent lại trả `case_*`, nên evaluator luôn tính sai. Ngoài ra, phần multi-judge ban đầu dùng Gemini bị quota/free-tier giới hạn, khiến benchmark dễ rơi vào fallback.

## 4. Tôi đã giải quyết các khó khăn đó như thế nào?
- Với retrieval eval, tôi lần theo đường dữ liệu từ `golden_set.jsonl` -> `MainAgent` -> `RetrievalEvaluator`, sau đó sửa agent để load corpus chunk thật từ `data/docs` và giữ nguyên `doc_001..doc_034`.
- Với multi-judge, tôi chuyển môi trường test sang 2 judge OpenAI (`gpt-4o` và `gpt-4o-mini`) để tránh quota Gemini trong lúc benchmark.
- Với report, tôi đối chiếu file sample của giảng viên với output hiện tại rồi sửa phần export report thay vì chỉnh tay file JSON.

## 5. Kết quả đầu ra của phần tôi phụ trách là gì?
- Benchmark chạy end-to-end và sinh được:
  - `reports/summary.json`
  - `reports/benchmark_results.json`
- V2 cải thiện so với V1:
  - `V1 score = 1.2458`
  - `V2 score = 1.8644`
  - `Delta = +0.62`
- Retrieval của V2 đã tăng lên:
  - `Hit Rate = 0.5085`
  - `MRR = 0.3983`
- Multi-judge có chỉ số đồng thuận:
  - `agreement_rate = 0.8305`
  - `Cohen's Kappa = 0.5453`

## 6. Tôi học được gì sau bài lab này?
Tôi rút ra ba bài học quan trọng:
- metric chỉ có ý nghĩa khi các namespace và schema dữ liệu khớp tuyệt đối; sai `doc_id` là đủ làm toàn bộ retrieval eval vô nghĩa
- multi-judge không chỉ là gọi 2 model, mà còn phải nghĩ đến conflict resolution, reliability metric, và quota/cost thực tế
- report đầu ra cần được thiết kế nhất quán từ code thay vì sửa tay, nếu không rất dễ lệch format khi chạy lại benchmark

## 7. Nếu có thêm thời gian, tôi sẽ cải thiện gì?
Nếu có thêm thời gian, tôi sẽ:
- thêm lớp `policy/refusal` để xử lý tốt hơn các case `prompt_injection`, `goal_hijacking`, và `jailbreak`
- thêm ngưỡng confidence để agent biết khi nào nên nói “không có thông tin” hoặc hỏi lại
- cải thiện answer synthesis cho các case `comparison`, `conflicting`, và `multi_turn`
- tối ưu cost/performance bằng rate limiting, batching hợp lý, và log chi phí chi tiết cho từng lần eval
