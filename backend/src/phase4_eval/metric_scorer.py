# src/phase4_eval/metric_scorer.py
"""
PHASE 4 — Metric Scorer: Cascade 2 Tầng 100% Local (spec V8.0 §4.1, §4.2)
           + Đối chiếu Ground Truth JSON

Tầng 1 (CPU, không tốn VRAM):
    score_relevance = BGE-Reranker(query, context)      -> Context Relevance
    score_ground     = NLI(context, answer)  [P(Entailment)] -> Groundedness
    score_answer_rel = BGE-Reranker(query, answer)       -> Answer Relevance
    (RAG Triad đầy đủ 3 tiêu chí, tách biệt Reranker khỏi NLI theo §Tóm tắt #17)

    Luồng quyết định:
        score > 0.7 (cả relevance & ground)  -> PASS  -> eval_report.jsonl
        0.4 <= score <= 0.7 (bất kỳ tiêu chí nào)      -> eval_queue_deep.jsonl
        score < 0.4                          -> FAIL  -> eval_report.jsonl

Tầng 2 (Prometheus-2 GGUF 4-bit, VRAM Swapping — chỉ case biên §4.2):
    del qwen_model (nếu đang ghim) -> gc.collect() -> empty_cache()
    -> load Prometheus-2 -> chấm toàn bộ deep queue -> del judge_model
    -> gc.collect() -> empty_cache()

Ground Truth Comparison: đối chiếu `trake`/`answer` sinh ra với file JSON
Ground Truth (`[{query, expected_keyframe, expected_answer}, ...]`), tính
Keyframe Hit-Rate + Reciprocal Rank + đơn giản hoá Answer Match.
"""

from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import torch 
from backend.src.common.config_loader import config

# Lấy cấu hình động từ settings.yaml
_EVAL_CFG = config.get("phase4_eval", {})
DEFAULT_THRESHOLD_LOW = float(_EVAL_CFG.get("threshold_low", 0.4))
DEFAULT_THRESHOLD_HIGH = float(_EVAL_CFG.get("threshold_high", 0.7))
DEFAULT_RERANKER = _EVAL_CFG.get("reranker_model", "BAAI/bge-reranker-base")
DEFAULT_NLI = _EVAL_CFG.get("nli_model", "cross-encoder/nli-deberta-v3-base")
DEFAULT_REPORT_PATH = _EVAL_CFG.get("eval_report_path", "backend/data/eval_logs/eval_report.jsonl")
DEFAULT_QUEUE_PATH = _EVAL_CFG.get("eval_queue_deep_path", "backend/data/eval_logs/eval_queue_deep.jsonl")


def _sigmoid(x: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class TierOneScore:
    query: str
    context: str
    answer: str
    score_relevance: float
    score_ground: float
    score_answer_relevance: float
    verdict: str  # "pass" | "fail" | "deep_queue"


class MetricScorer:
    def __init__(self):
        self._reranker = None
        self._nli_tokenizer = None
        self._nli_model = None
        self._judge_model = None  # Prometheus-2 (llama_cpp)

    # ------------------------------------------------------------------ #
    # Tầng 1 — CPU models lifecycle
    # ------------------------------------------------------------------ #
    def _load_tier1_models(
        self,
        reranker_name: str = "BAAI/bge-reranker-base",
        nli_name: str = "cross-encoder/nli-deberta-v3-base",
    ) -> None:
        from sentence_transformers import CrossEncoder
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self._reranker = CrossEncoder(reranker_name, device="cpu")
        self._nli_tokenizer = AutoTokenizer.from_pretrained(nli_name)
        self._nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name)
        self._nli_model.eval()

    def _unload_tier1_models(self) -> None:
        self._reranker = None
        self._nli_tokenizer = None
        self._nli_model = None
        gc.collect()  # CPU-only, nhưng vẫn dọn tham chiếu để giải phóng RAM

    def _nli_entailment_prob(self, premise: str, hypothesis: str) -> float:
        inputs = self._nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = self._nli_model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        id2label = {int(k): v.lower() for k, v in self._nli_model.config.id2label.items()}
        entail_idx = next((idx for idx, label in id2label.items() if "entail" in label), None)
        if entail_idx is None:
            entail_idx = int(torch.argmax(probs).item())
        return float(probs[entail_idx].item())

    def score_tier1(self, query: str, context: str, answer: str) -> TierOneScore:
        raw_relevance = float(self._reranker.predict([(query, context)])[0])
        score_relevance = _sigmoid(raw_relevance)

        score_ground = self._nli_entailment_prob(premise=context, hypothesis=answer)

        raw_answer_rel = float(self._reranker.predict([(query, answer)])[0])
        score_answer_relevance = _sigmoid(raw_answer_rel)

        min_score = min(score_relevance, score_ground, score_answer_relevance)
        max_deciding = min(score_relevance, score_ground)  # 2 tiêu chí chính theo §4.1

        if max_deciding > DEFAULT_THRESHOLD_HIGH and score_answer_relevance > DEFAULT_THRESHOLD_HIGH:
            verdict = "pass"
        elif min_score < DEFAULT_THRESHOLD_LOW:
            verdict = "fail"
        else:
            verdict = "deep_queue"

        return TierOneScore(
            query=query, context=context, answer=answer,
            score_relevance=round(score_relevance, 4),
            score_ground=round(score_ground, 4),
            score_answer_relevance=round(score_answer_relevance, 4),
            verdict=verdict,
        )

    def run_cascade_tier1(
        self,
        records: Sequence[Dict[str, str]],
        eval_report_path: str,
        eval_queue_deep_path: str,
    ) -> Dict[str, int]:
        """
        Chạy Tầng 1 cho toàn bộ `records` ([{query, context, answer}, ...]),
        ghi trực tiếp PASS/FAIL vào eval_report.jsonl, đẩy case biên vào
        eval_queue_deep.jsonl. Trả về thống kê số lượng mỗi nhóm.
        """
        Path(eval_report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(eval_queue_deep_path).parent.mkdir(parents=True, exist_ok=True)

        stats = {"pass": 0, "fail": 0, "deep_queue": 0}
        self._load_tier1_models()
        try:
            with open(eval_report_path, "a", encoding="utf-8") as report_f, \
                 open(eval_queue_deep_path, "a", encoding="utf-8") as deep_f:
                for rec in records:
                    scored = self.score_tier1(rec["query"], rec["context"], rec["answer"])
                    stats[scored.verdict] += 1
                    line = json.dumps(asdict(scored), ensure_ascii=False)
                    if scored.verdict == "deep_queue":
                        deep_f.write(line + "\n")
                    else:
                        report_f.write(line + "\n")
        finally:
            self._unload_tier1_models()

        return stats

    # ------------------------------------------------------------------ #
    # Tầng 2 — Prometheus-2 Deep Judge (VRAM Swapping, §4.2)
    # ------------------------------------------------------------------ #
    def deep_judge_tier2(
        self,
        eval_queue_deep_path: str,
        eval_report_path: str,
        prometheus_gguf_path: str,
        qwen_model_ref: Optional[Any] = None,
        n_gpu_layers: int = -1,
    ) -> Dict[str, int]:
        """
        1. Gỡ Qwen 2.5 VL khỏi VRAM nếu caller truyền tham chiếu qua
           `qwen_model_ref` (§4.2 bước 1).
        2. Nạp Prometheus-2 GGUF 4-bit qua llama-cpp-python.
        3. Chấm RAG Triad cho từng record trong eval_queue_deep.jsonl.
        4. Giải phóng VRAM sau khi chấm xong toàn bộ deep queue.
        """
        # --- 1) Gỡ Qwen (nếu có) ---
        if qwen_model_ref is not None:
            del qwen_model_ref
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- 2) Nạp Prometheus-2 ---
        from llama_cpp import Llama
        self._judge_model = Llama(
            model_path=prometheus_gguf_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=4096,
            verbose=False,
        )

        deep_path = Path(eval_queue_deep_path)
        if not deep_path.is_file():
            return {"judged": 0}

        stats = {"pass": 0, "fail": 0}
        judged_lines: List[str] = []

        with open(deep_path, "r", encoding="utf-8") as f:
            deep_records = [json.loads(line) for line in f if line.strip()]

        try:
            for rec in deep_records:
                verdict, feedback_score = self._prometheus_judge_one(
                    query=rec["query"], context=rec["context"], answer=rec["answer"]
                )
                rec["tier2_verdict"] = verdict
                rec["tier2_score"] = feedback_score
                stats[verdict] += 1
                judged_lines.append(json.dumps(rec, ensure_ascii=False))
        finally:
            # --- 4) Giải phóng VRAM sau khi chấm xong TOÀN BỘ deep queue ---
            del self._judge_model
            self._judge_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        with open(eval_report_path, "a", encoding="utf-8") as report_f:
            for line in judged_lines:
                report_f.write(line + "\n")

        # Xoá deep queue đã xử lý xong (đã ghi kết quả cuối vào eval_report)
        deep_path.unlink(missing_ok=True)

        return stats

    def _prometheus_judge_one(self, query: str, context: str, answer: str) -> tuple[str, int]:
        """
        Prompt Prometheus-2 theo định dạng Absolute Grading (1-5), chấm
        đồng thời Groundedness + Relevance của (query, context, answer).
        Điểm >= 4 -> pass, ngược lại -> fail.
        """
        rubric_prompt = (
            "###Task Description:\n"
            "An instruction (containing a query, retrieved context, and a "
            "generated answer), and a score rubric are given.\n"
            "1. Write feedback assessing whether the answer is faithfully "
            "grounded in the context AND relevant to the query.\n"
            "2. After the feedback, write a score from 1 to 5.\n"
            "3. Output format: 'Feedback: (write feedback) [RESULT] (an integer number between 1 and 5)'\n\n"
            f"###Query:\n{query}\n\n###Context:\n{context}\n\n###Answer:\n{answer}\n\n"
            "###Score Rubric:\n"
            "1: Hoàn toàn không liên quan hoặc bịa đặt thông tin ngoài context.\n"
            "3: Liên quan một phần, có vài chi tiết không được context hỗ trợ.\n"
            "5: Hoàn toàn liên quan và được context hỗ trợ đầy đủ.\n\n"
            "###Feedback:"
        )

        output = self._judge_model(
            rubric_prompt, max_tokens=256, temperature=0.0, stop=["###"],
        )
        text_out = output["choices"][0]["text"]

        match = re.search(r"\[RESULT\]\s*(\d)", text_out)
        score = int(match.group(1)) if match else 1
        verdict = "pass" if score >= 4 else "fail"
        return verdict, score

    # ------------------------------------------------------------------ #
    # Ground Truth Comparison
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @classmethod
    def _answer_overlap_ratio(cls, generated: str, expected: str) -> float:
        """Tỉ lệ trùng lặp token đơn giản (Jaccard) — dùng khi không có model NLI sẵn."""
        gen_tokens = set(cls._normalize_text(generated).split())
        exp_tokens = set(cls._normalize_text(expected).split())
        if not exp_tokens:
            return 0.0
        return len(gen_tokens & exp_tokens) / len(exp_tokens)

    def compare_with_ground_truth(
        self,
        generated_results: Sequence[Dict[str, Any]],
        ground_truth_path: str,
    ) -> Dict[str, Any]:
        """
        Đối chiếu kết quả sinh ra (mỗi item: {query, answer, trake:[{keyframe_path,...}]})
        với Ground Truth JSON: [{query, expected_keyframe, expected_answer}, ...].
        Trả về Keyframe Hit-Rate, Mean Reciprocal Rank (keyframe), và
        Answer Overlap Ratio trung bình.
        """
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

        gt_by_query = {self._normalize_text(g["query"]): g for g in ground_truth}

        hit_count = 0
        reciprocal_ranks: List[float] = []
        answer_overlaps: List[float] = []
        matched = 0

        for result in generated_results:
            gt = gt_by_query.get(self._normalize_text(result.get("query", "")))
            if gt is None:
                continue
            matched += 1

            trake = result.get("trake", [])
            keyframe_paths = [t.get("keyframe_path") for t in trake if t.get("keyframe_path")]
            expected_kf = gt.get("expected_keyframe")

            if expected_kf and expected_kf in keyframe_paths:
                hit_count += 1
                rank = keyframe_paths.index(expected_kf) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

            if gt.get("expected_answer"):
                overlap = self._answer_overlap_ratio(result.get("answer", ""), gt["expected_answer"])
                answer_overlaps.append(overlap)

        return {
            "matched_queries": matched,
            "total_ground_truth": len(ground_truth),
            "keyframe_hit_rate": round(hit_count / matched, 4) if matched else 0.0,
            "mean_reciprocal_rank": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
            "mean_answer_overlap": round(sum(answer_overlaps) / len(answer_overlaps), 4) if answer_overlaps else 0.0,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4 Metric Scorer (standalone test).")
    parser.add_argument("--records_json", required=True, help='JSON list [{"query","context","answer"}]')
    parser.add_argument("--eval_report_path", default="data/eval_logs/eval_report.jsonl")
    parser.add_argument("--eval_queue_deep_path", default="data/eval_logs/eval_queue_deep.jsonl")
    parser.add_argument("--prometheus_gguf_path", default=None)
    parser.add_argument("--ground_truth_path", default=None)
    args = parser.parse_args()

    with open(args.records_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    scorer = MetricScorer()
    tier1_stats = scorer.run_cascade_tier1(records, args.eval_report_path, args.eval_queue_deep_path)
    output = {"tier1_stats": tier1_stats}

    if args.prometheus_gguf_path:
        tier2_stats = scorer.deep_judge_tier2(
            args.eval_queue_deep_path, args.eval_report_path, args.prometheus_gguf_path
        )
        output["tier2_stats"] = tier2_stats

    if args.ground_truth_path:
        generated_results = [{"query": r["query"], "answer": r["answer"], "trake": r.get("trake", [])} for r in records]
        output["ground_truth_metrics"] = scorer.compare_with_ground_truth(generated_results, args.ground_truth_path)

    print(json.dumps(output, ensure_ascii=False, indent=2))