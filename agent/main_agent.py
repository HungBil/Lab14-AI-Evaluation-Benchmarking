import asyncio
import json
import random
import re
from pathlib import Path
from typing import Dict, List

class MainAgent:
    """
    Đây là Agent mẫu sử dụng kiến trúc RAG đơn giản.
    Sinh viên nên thay thế phần này bằng Agent thực tế đã phát triển ở các buổi trước.
    def __init__(
    """
    """
    Agent hỗ trợ 2 phiên bản để benchmark:
    - v1_random: baseline cố tình retrieval sai bằng random.
    - v2_hybrid: retrieval cải thiện bằng hybrid (dense+sparse RRF).
    """
    def __init__(
        self,
        version: str = "v1_random",
        data_path: str = "data/golden_set.jsonl",
        top_k_search: int = 10,
        top_k_select: int = 3,
    ):
        self.version = version
        self.name = f"SupportAgent-{version}"
        self.top_k_search = top_k_search
        self.top_k_select = top_k_select
        self.docs = self._load_docs(Path(data_path))

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower())

    def _load_docs(self, path: Path) -> List[Dict]:
        if not path.exists():
            return []

        docs: List[Dict] = []
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get("context") or row.get("expected_answer") or ""
                docs.append(
                    {
                        "doc_id": f"case_{idx}",
                        "text": text,
                        "metadata": row.get("metadata", {}),
                    }
                )
        return docs

    def _retrieve_v1_random(self, question: str) -> List[Dict]:
        if not self.docs:
            return []
        q_tokens = set(self._tokenize(question))
        # Cố tình lấy chunk có độ trùng từ thấp (ưu tiên overlap=0).
        wrong_pool = []
        for d in self.docs:
            overlap = len(q_tokens.intersection(self._tokenize(d["text"])))
            if overlap == 0:
                wrong_pool.append(d)
        pool = wrong_pool if wrong_pool else self.docs
        return random.sample(pool, k=min(self.top_k_select, len(pool)))

    def _dense_score(self, q_set: set, text: str) -> float:
        t_set = set(self._tokenize(text))
        if not q_set or not t_set:
            return 0.0
        inter = len(q_set.intersection(t_set))
        union = len(q_set.union(t_set))
        return inter / union if union else 0.0

    def _sparse_score(self, q_tokens: List[str], text: str) -> float:
        t_set = set(self._tokenize(text))
        if not q_tokens or not t_set:
            return 0.0
        hits = sum(1 for t in q_tokens if t in t_set)
        return hits / len(q_tokens)

    def _retrieve_v2_hybrid(self, question: str) -> List[Dict]:
        if not self.docs:
            return []

        q_tokens = self._tokenize(question)
        q_set = set(q_tokens)

        dense_ranked = sorted(
            self.docs,
            key=lambda d: self._dense_score(q_set, d["text"]),
            reverse=True,
        )[: self.top_k_search]
        sparse_ranked = sorted(
            self.docs,
            key=lambda d: self._sparse_score(q_tokens, d["text"]),
            reverse=True,
        )[: self.top_k_search]

        rrf_k = 60
        dense_weight = 0.6
        sparse_weight = 0.4

        dense_pos = {d["doc_id"]: i + 1 for i, d in enumerate(dense_ranked)}
        sparse_pos = {d["doc_id"]: i + 1 for i, d in enumerate(sparse_ranked)}
        scores = {}
        for doc_id in set(dense_pos.keys()).union(sparse_pos.keys()):
            score = 0.0
            if doc_id in dense_pos:
                score += dense_weight * (1.0 / (rrf_k + dense_pos[doc_id]))
            if doc_id in sparse_pos:
                score += sparse_weight * (1.0 / (rrf_k + sparse_pos[doc_id]))
            scores[doc_id] = score

        doc_map = {d["doc_id"]: d for d in self.docs}
        top_ids = [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        top_ids = top_ids[: self.top_k_select]
        return [doc_map[doc_id] for doc_id in top_ids if doc_id in doc_map]

    def _build_answer(self, selected_docs: List[Dict], question: str) -> str:
        if not selected_docs:
            return "Không đủ dữ liệu để trả lời."
        if self.version == "v1_random":
            wrong_templates = [
                "Thông tin truy xuất cho thấy chính sách hoàn toàn khác, chưa thể kết luận.",
                "Nội dung hiện có không liên quan trực tiếp, cần chuyển bộ phận khác xử lý.",
                "Theo nguồn ngẫu nhiên, điều kiện xử lý là 999 giờ và cần 4 cấp duyệt.",
            ]
            return random.choice(wrong_templates)
        return "Dựa trên ngữ cảnh truy xuất, câu trả lời là: " + selected_docs[0]["text"][:220]

    async def query(self, question: str) -> Dict:
        await asyncio.sleep(0.12 if self.version == "v2_hybrid" else 0.15)

        if self.version == "v1_random":
            selected = self._retrieve_v1_random(question)
            model_name = "baseline-random"
        elif self.version == "v2_hybrid":
            selected = self._retrieve_v2_hybrid(question)
            model_name = "hybrid-rrf"
        else:
            raise ValueError(f"Version không hợp lệ: {self.version}")

        answer = self._build_answer(selected, question)
        return {
            "answer": answer,
            "contexts": [d["text"] for d in selected],
            "retrieved_ids": [d["doc_id"] for d in selected],
            "metadata": {
                "model": model_name,
                "tokens_used": 100 + 35 * len(selected),
                "sources": [d["doc_id"] for d in selected],
            },
        }

if __name__ == "__main__":
    agent = MainAgent(version="v1_random")
    async def test():
        resp = await agent.query("Làm thế nào để đổi mật khẩu?")
        print(resp)
    asyncio.run(test())
