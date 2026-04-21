import asyncio
import json
import os
from typing import Dict, Any, List


class LLMJudge:
    def __init__(self):
        # Rubrics đánh giá cho Judge
        self.rubrics = {
            "accuracy": "Chấm điểm từ 1-5 dựa trên độ chính xác so với Ground Truth...",
            "tone": "Chấm điểm từ 1-5 dựa trên sự chuyên nghiệp của ngôn ngữ..."
        }

    def _build_judge_prompt(self, question: str, answer: str, ground_truth: str) -> str:
        """Tạo prompt chung cho cả 2 judge model."""
        return (
            "Bạn là giám khảo AI. Hãy chấm điểm câu trả lời từ 1 đến 5.\n\n"
            "Tiêu chí chấm:\n"
            "- 5: Hoàn hảo, chính xác và đầy đủ so với đáp án chuẩn\n"
            "- 4: Tốt, gần đúng nhưng thiếu chi tiết nhỏ\n"
            "- 3: Trung bình, đúng một phần\n"
            "- 2: Kém, sai nhiều thông tin\n"
            "- 1: Rất kém, hoàn toàn sai hoặc không liên quan\n\n"
            f"Câu hỏi: {question}\n"
            f"Đáp án chuẩn (Ground Truth): {ground_truth}\n"
            f"Câu trả lời cần chấm: {answer}\n\n"
            'Trả về JSON duy nhất: {"score": <1-5>, "reasoning": "<giải thích ngắn>"}'
        )

    async def _call_gpt4o(self, prompt: str) -> Dict:
        """Gọi OpenAI GPT-4o làm Judge A."""
        from openai import AsyncOpenAI
        client = AsyncOpenAI()  # đọc OPENAI_API_KEY từ env

        resp = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(resp.choices[0].message.content)

    async def _call_gemini(self, prompt: str) -> Dict:
        """Gọi Google Gemini 2.5 Flash làm Judge B."""
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        resp = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0
            )
        )
        return json.loads(resp.text)

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        EXPERT TASK: Gọi 2 model judge (GPT-4o + Gemini 2.5 Flash) song song.
        Tính toán sự sai lệch. Nếu lệch > 1 điểm, áp dụng conflict resolution.
        """
        prompt = self._build_judge_prompt(question, answer, ground_truth)

        # Gọi 2 judge song song bằng asyncio.gather
        score_a = 3  # fallback mặc định
        score_b = 3
        reasoning_a = ""
        reasoning_b = ""

        try:
            gpt_result, gemini_result = await asyncio.gather(
                self._call_gpt4o(prompt),
                self._call_gemini(prompt)
            )
            score_a = int(gpt_result.get("score", 3))
            score_b = int(gemini_result.get("score", 3))
            reasoning_a = gpt_result.get("reasoning", "")
            reasoning_b = gemini_result.get("reasoning", "")
        except Exception as e:
            print(f"⚠️ Lỗi khi gọi Judge: {e}. Dùng điểm fallback = 3.")

        # Giới hạn score trong khoảng 1-5
        score_a = max(1, min(5, score_a))
        score_b = max(1, min(5, score_b))

        # --- Conflict Resolution ---
        diff = abs(score_a - score_b)
        if diff > 1:
            # Xung đột lớn: lấy điểm thấp hơn (chiến lược bảo thủ / conservative)
            final_score = float(min(score_a, score_b))
            agreement = 0.0
        elif diff == 1:
            # Lệch nhẹ: lấy trung bình
            final_score = (score_a + score_b) / 2
            agreement = 0.5
        else:
            # Đồng thuận hoàn toàn
            final_score = float(score_a)
            agreement = 1.0

        return {
            "final_score": final_score,
            "agreement_rate": agreement,
            "individual_scores": {"gpt-4o": score_a, "gemini-2.5-flash": score_b},
            "individual_reasoning": {"gpt-4o": reasoning_a, "gemini-2.5-flash": reasoning_b},
            "conflict": diff > 1
        }

    async def check_position_bias(self, question: str, response_a: str, response_b: str) -> Dict:
        """
        Nâng cao: Đổi chỗ response A và B để xem Judge có thiên vị vị trí không.
        Gọi judge 2 lần: lần 1 (A trước B), lần 2 (B trước A).
        Nếu điểm thay đổi nhiều → judge có position bias.
        """
        score_original = await self.evaluate_multi_judge(question, response_a, response_b)
        score_swapped = await self.evaluate_multi_judge(question, response_b, response_a)

        bias = abs(score_original["final_score"] - score_swapped["final_score"])
        return {
            "original_score": score_original["final_score"],
            "swapped_score": score_swapped["final_score"],
            "bias_delta": bias,
            "has_position_bias": bias > 0.5
        }

    def calculate_cohens_kappa(
        self,
        scores_a: List[int],
        scores_b: List[int],
        categories: List[int] = None
    ) -> Dict[str, Any]:
        """
        Tính Cohen's Kappa — độ đồng thuận giữa 2 judge model trên toàn bộ batch,
        có loại trừ yếu tố đồng thuận ngẫu nhiên.

        Công thức: κ = (P_observed - P_chance) / (1 - P_chance)

        - P_observed: tỷ lệ case mà 2 judge cho cùng điểm
        - P_chance:   tỷ lệ đồng thuận ngẫu nhiên (dựa trên phân phối điểm của mỗi judge)

        Thang đọc kết quả:
          κ < 0.0  → Kém hơn ngẫu nhiên
          0.0–0.2  → Gần như không đồng thuận
          0.2–0.4  → Đồng thuận yếu
          0.4–0.6  → Đồng thuận trung bình
          0.6–0.8  → Đồng thuận tốt
          0.8–1.0  → Đồng thuận rất tốt / gần hoàn hảo
        """
        if len(scores_a) != len(scores_b) or len(scores_a) == 0:
            return {"kappa": 0.0, "interpretation": "Không đủ dữ liệu"}

        if categories is None:
            categories = list(range(1, 6))  # thang điểm 1-5

        n = len(scores_a)

        # P_observed: tỷ lệ 2 judge đồng ý
        p_observed = sum(1 for a, b in zip(scores_a, scores_b) if a == b) / n

        # P_chance: xác suất tình cờ đồng thuận
        # = Σ_k [ (số lần judge A chọn k / n) × (số lần judge B chọn k / n) ]
        p_chance = sum(
            (scores_a.count(k) / n) * (scores_b.count(k) / n)
            for k in categories
        )

        # Tránh chia cho 0 (khi P_chance = 1, tức là 2 judge luôn chọn cùng 1 nhãn)
        if p_chance >= 1.0:
            kappa = 1.0
        else:
            kappa = (p_observed - p_chance) / (1 - p_chance)

        # Diễn giải
        if kappa < 0:
            interpretation = "Kém hơn ngẫu nhiên — 2 judge không đáng tin"
        elif kappa < 0.2:
            interpretation = "Gần như không đồng thuận"
        elif kappa < 0.4:
            interpretation = "Đồng thuận yếu"
        elif kappa < 0.6:
            interpretation = "Đồng thuận trung bình"
        elif kappa < 0.8:
            interpretation = "Đồng thuận tốt"
        else:
            interpretation = "Đồng thuận rất tốt / gần hoàn hảo"

        return {
            "kappa": round(kappa, 4),
            "p_observed": round(p_observed, 4),
            "p_chance": round(p_chance, 4),
            "n_samples": n,
            "interpretation": interpretation
        }
