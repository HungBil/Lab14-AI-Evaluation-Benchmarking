import asyncio
import json
import os
import time

from dotenv import load_dotenv

from agent.main_agent import MainAgent
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner


load_dotenv()


class ExpertEvaluator:

    def __init__(self):
        self._retrieval_evaluator = RetrievalEvaluator()

    async def score(self, case, resp):
        return await self._retrieval_evaluator.score(case, resp)


class MultiModelJudge:

    def __init__(self):
        self._judge = LLMJudge()

    async def evaluate_multi_judge(self, q, a, gt):
        return await self._judge.evaluate_multi_judge(q, a, gt)

    def calculate_cohens_kappa(self, scores_a, scores_b):
        return self._judge.calculate_cohens_kappa(scores_a, scores_b)


def _format_single_result(result):
    judge = result.get("judge", {})
    ragas = result.get("ragas", {})
    individual_scores = judge.get("individual_scores", {})
    individual_reasoning = judge.get("individual_reasoning", {})
    judge_b_key = next((k for k in individual_scores.keys() if k != "gpt-4o"), "gpt-4o-mini")

    return {
        "test_case": result.get("test_case"),
        "agent_response": result.get("agent_response"),
        "latency": result.get("latency"),
        "ragas": {
            "hit_rate": ragas.get("retrieval", {}).get("hit_rate", 0.0),
            "mrr": ragas.get("retrieval", {}).get("mrr", 0.0),
            "faithfulness": ragas.get("faithfulness", 0.0),
            "relevancy": ragas.get("relevancy", 0.0),
        },
        "judge": {
            "final_score": judge.get("final_score", 0.0),
            "agreement_rate": judge.get("agreement_rate", 0.0),
            "individual_results": {
                "gpt-4o": {
                    "score": individual_scores.get("gpt-4o", 0),
                    "reasoning": individual_reasoning.get("gpt-4o", ""),
                },
                judge_b_key: {
                    "score": individual_scores.get(judge_b_key, 0),
                    "reasoning": individual_reasoning.get(judge_b_key, ""),
                },
            },
            "status": "conflict" if judge.get("conflict") else "consensus",
        },
        "status": result.get("status", "fail"),
    }


def _build_summary_report(v1_summary, v2_summary, delta):
    v1_metrics = v1_summary["metrics"]
    v2_metrics = v2_summary["metrics"]
    decision = "APPROVE" if delta > 0 and v2_metrics["hit_rate"] >= v1_metrics["hit_rate"] else "BLOCK"

    return {
        "metadata": {
            "total": v2_summary["metadata"]["total"],
            "version": "BASELINE (V1)",
            "timestamp": v2_summary["metadata"]["timestamp"],
            "versions_compared": ["V1", "V2"],
        },
        "metrics": {
            "avg_score": v1_metrics["avg_score"],
            "hit_rate": v1_metrics["hit_rate"],
            "agreement_rate": v1_metrics["agreement_rate"],
        },
        "regression": {
            "v1": {
                "score": v1_metrics["avg_score"],
                "hit_rate": v1_metrics["hit_rate"],
                "judge_agreement": v1_metrics["agreement_rate"],
            },
            "v2": {
                "score": v2_metrics["avg_score"],
                "hit_rate": v2_metrics["hit_rate"],
                "judge_agreement": v2_metrics["agreement_rate"],
            },
            "decision": decision,
        },
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
    judge_instance = MultiModelJudge()

    # Thu thập điểm riêng từng judge để tính Cohen's Kappa trên toàn batch
    scores_gpt = [
        r["judge"]["individual_scores"].get("gpt-4o", 3)
        for r in results
        if "individual_scores" in r.get("judge", {})
    ]
    scores_gpt_mini = [
        r["judge"]["individual_scores"].get("gpt-4o-mini", 3)
        for r in results
        if "individual_scores" in r.get("judge", {})
    ]
    kappa_result = judge_instance.calculate_cohens_kappa(scores_gpt, scores_gpt_mini)

    print(f"\n🔬 Cohen's Kappa: {kappa_result['kappa']} — {kappa_result['interpretation']}")

    summary = {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total,
            "cohens_kappa": kappa_result,
        },
    }
    return results, summary


async def run_benchmark(version, agent):
    _, summary = await run_benchmark_with_results(version, agent)
    return summary


async def main():
    v1_results, v1_summary = await run_benchmark_with_results(
        "Agent_V1_Base", MainAgent(version="v1_random")
    )
    v2_results, v2_summary = await run_benchmark_with_results(
        "Agent_V2_Optimized", MainAgent(version="v2_hybrid")
    )

    if not v1_results or not v1_summary or not v2_results or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")

    summary_report = _build_summary_report(v1_summary, v2_summary, delta)
    benchmark_report = {
        "v1": [_format_single_result(r) for r in v1_results],
        "v2": [_format_single_result(r) for r in v2_results],
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, ensure_ascii=False, indent=2)

    if delta > 0 and v2_summary["metrics"]["hit_rate"] >= v1_summary["metrics"]["hit_rate"]:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")


if __name__ == "__main__":
    asyncio.run(main())
