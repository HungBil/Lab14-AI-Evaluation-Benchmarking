# Báo cáo Cá nhân

**Họ và tên:** Nguyễn Đông Hưng  
**MSSV:** 2A202600392

## 1. Tôi đã làm được gì trong bài lab này?
Tôi đã tham gia hoàn thiện phần benchmark cho agent, bao gồm chạy evaluation, kiểm tra kết quả retrieval, chuẩn hóa report đầu ra và đối chiếu với sample để file nộp đúng format. Tôi cũng hỗ trợ sửa phần multi-judge để benchmark chạy ổn định hơn trong môi trường test.

## 2. Khó khăn lớn nhất tôi gặp là gì và tôi đã xử lý như thế nào?
Khó khăn lớn nhất là benchmark chạy được nhưng `hit_rate` ban đầu bằng 0 và kết quả judge dễ rơi vào fallback. Tôi đã lần theo luồng dữ liệu giữa dataset, agent và evaluator để tìm ra lỗi lệch `doc_id`, sau đó sửa lại để `expected_retrieval_ids` khớp với `sources`. Đồng thời tôi kiểm tra lại phần judge để tránh lỗi quota/runtime trong lúc chạy benchmark.

## 3. Tôi học được gì sau bài lab này?
Tôi học được rằng khi làm evaluation cho AI agent, metric chỉ có ý nghĩa nếu schema dữ liệu và pipeline khớp hoàn toàn. Ngoài ra, việc so sánh nhiều judge model giúp kết quả khách quan hơn, nhưng cũng phải tính đến quota, cost và độ ổn định khi triển khai thực tế.
