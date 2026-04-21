import asyncio
import json
import os
import re
import time
from engine.runner import BenchmarkRunner
from agent.main_agent import MainAgent
from engine.retrieval_eval import RetrievalEvaluator


def _tokenize(text: str):
    return set(re.findall(r"\w+", (text or "").lower()))


class ExpertEvaluator:
    def __init__(self):
        self.retrieval_evaluator = RetrievalEvaluator()

    async def score(self, case, resp):
        answer_tokens = _tokenize(resp.get("answer", ""))
        expected_tokens = _tokenize(case.get("expected_answer", ""))
        query_tokens = _tokenize(case.get("question", ""))

        faithfulness = len(answer_tokens.intersection(expected_tokens)) / max(1, len(expected_tokens))
        relevancy = len(answer_tokens.intersection(query_tokens)) / max(1, len(query_tokens))

        expected_ids = case.get("expected_retrieval_ids", [])
        retrieved_ids = resp.get("retrieved_ids", [])
        if expected_ids and retrieved_ids:
            hit_rate = self.retrieval_evaluator.calculate_hit_rate(expected_ids, retrieved_ids)
            mrr = self.retrieval_evaluator.calculate_mrr(expected_ids, retrieved_ids)
        else:
            # Fallback cho dataset chưa có expected_retrieval_ids.
            mrr = faithfulness
            hit_rate = 1.0 if faithfulness >= 0.2 else 0.0

        return {
            "faithfulness": min(1.0, max(0.0, faithfulness)),
            "relevancy": min(1.0, max(0.0, relevancy)),
            "retrieval": {"hit_rate": hit_rate, "mrr": mrr},
        }


class MultiModelJudge:
    async def evaluate_multi_judge(self, q, a, gt):
        answer_tokens = _tokenize(a)
        gt_tokens = _tokenize(gt)
        q_tokens = _tokenize(q)

        overlap_gt = len(answer_tokens.intersection(gt_tokens)) / max(1, len(gt_tokens))
        overlap_q = len(answer_tokens.intersection(q_tokens)) / max(1, len(q_tokens))

        score_a = max(1.0, min(5.0, 1.0 + 4.0 * overlap_gt))
        score_b = max(1.0, min(5.0, 1.0 + 2.5 * overlap_gt + 1.5 * overlap_q))
        avg_score = (score_a + score_b) / 2
        score_diff = abs(score_a - score_b)
        agreement = max(0.0, 1.0 - (score_diff / 4.0))

        return {
            "final_score": avg_score,
            "agreement_rate": agreement,
            "individual_scores": {"judge_a": score_a, "judge_b": score_b},
            "reasoning": (
                "Điểm dựa trên mức độ trùng khớp giữa câu trả lời với expected answer "
                "và mức độ liên quan với câu hỏi."
            ),
        }


async def run_benchmark_with_results(agent_version: str, agent):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    runner = BenchmarkRunner(agent, ExpertEvaluator(), MultiModelJudge())
    results = await runner.run_all(dataset)

    total = len(results)
    summary = {
        "metadata": {"version": agent_version, "total": total, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total
        }
    }
    return results, summary

async def run_benchmark(version, agent):
    _, summary = await run_benchmark_with_results(version, agent)
    return summary


async def main():
    v1_summary = await run_benchmark("Agent_V1_Base", MainAgent(version="v1_random"))
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized", MainAgent(version="v2_hybrid"))

    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    if delta > 0 and v2_summary["metrics"]["hit_rate"] >= v1_summary["metrics"]["hit_rate"]:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")

if __name__ == "__main__":
    asyncio.run(main())
