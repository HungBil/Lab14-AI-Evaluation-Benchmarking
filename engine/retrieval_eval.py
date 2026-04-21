from typing import List, Dict


class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        Tính toán xem ít nhất 1 trong expected_ids có nằm trong top_k của retrieved_ids không.
        """
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Tính Mean Reciprocal Rank.
        Tìm vị trí đầu tiên của một expected_id trong retrieved_ids.
        MRR = 1 / position (vị trí 1-indexed). Nếu không thấy thì là 0.
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def score(self, test_case: Dict, response: Dict) -> Dict:
        """
        Tính retrieval metrics cho 1 test case đơn lẻ.
        Được gọi bởi BenchmarkRunner.run_single_test().

        - test_case: cần có trường 'expected_retrieval_ids'
        - response: output từ Agent, cần có 'metadata.sources' chứa danh sách doc IDs
        """
        expected_ids = test_case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("metadata", {}).get("sources", [])

        hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids)
        mrr = self.calculate_mrr(expected_ids, retrieved_ids)

        # Faithfulness và relevancy ước tính đơn giản dựa trên retrieval quality.
        # (Nếu cần chính xác hơn, tích hợp thư viện RAGAS để đánh giá bằng LLM.)
        faithfulness = 1.0 if hit_rate > 0 else 0.4
        relevancy = hit_rate

        return {
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": mrr
            }
        }

    async def evaluate_batch(self, dataset: List[Dict], agent=None) -> Dict:
        """
        Chạy eval cho toàn bộ bộ dữ liệu.
        Dataset cần có trường 'expected_retrieval_ids' và Agent trả về 'retrieved_ids'.
        """
        hit_rates = []
        mrrs = []
        failed_cases = []

        for case in dataset:
            expected_ids = case.get("expected_retrieval_ids", [])

            if agent:
                response = await agent.query(case["question"])
                retrieved_ids = response.get("metadata", {}).get("sources", [])
            else:
                retrieved_ids = []

            hr = self.calculate_hit_rate(expected_ids, retrieved_ids)
            mrr = self.calculate_mrr(expected_ids, retrieved_ids)
            hit_rates.append(hr)
            mrrs.append(mrr)

            if hr == 0:
                failed_cases.append({
                    "question": case["question"],
                    "expected": expected_ids,
                    "retrieved": retrieved_ids
                })

        total = len(hit_rates)
        return {
            "avg_hit_rate": sum(hit_rates) / total if total else 0,
            "avg_mrr": sum(mrrs) / total if total else 0,
            "failed_cases": failed_cases,
            "total": total
        }
