# src/phase3_search/vlm_generator.py
"""
PHASE 3 — VLM Generator: Qwen 2.5 VL + 0/1 Knapsack Cut-off (spec V8.0 §3.5)

Chỉ Qwen 2.5 VL (3B INT4) được phép GHIM CỐ ĐỊNH trên GPU trong khâu suy
luận (§0, nguyên tắc bất biến toàn cục) — khác với các model Phase 1/2
phải Load->Xử lý->Unload theo Tiered Caching. Tuy nhiên module vẫn cung
cấp `unload()` tường minh (del + gc.collect + empty_cache) để Phase 4
có thể "gỡ Qwen — nạp Judge" khi cần swap VRAM (§4.2).

0/1 Knapsack DP (thay Greedy, §3.5):
    - Capacity  W = --max_context_images (mặc định 4)
    - Item = Logical Chunk:
        + Chunk đơn (không vắt cảnh):  Weight=1, Value=điểm RRF/rerank của chunk.
        + Chunk cặp (vắt cảnh, có `parent_chunk_id`): Weight=2 (tính GỘP
          cho cả cặp — 2 ảnh), Value = điểm của sub-chunk A (partner B
          không có điểm riêng, không tự đóng góp Value, chỉ "đi kèm").
    - DP đảm bảo tối ưu TOÀN CỤC (không bỏ sót slot như Greedy), tránh
      tràn context/VRAM khi nhét ảnh vào Qwen 2.5 VL.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image


@dataclass
class KnapsackItem:
    item_id: str
    weight: int
    value: float
    text: str
    keyframe_paths: List[str] = field(default_factory=list)


# -------------------------------------------------------------------------- #
# 0/1 Knapsack DP — đúng chuẩn, có backtrack chọn item
# -------------------------------------------------------------------------- #
def knapsack_select(items: Sequence[KnapsackItem], capacity: int) -> Tuple[List[KnapsackItem], float]:
    """
    DP bảng 2 chiều dp[i][w] = giá trị tối đa dùng i item đầu tiên với
    capacity w. Backtrack để lấy đúng tập item được chọn (khác Greedy vì
    xét toàn cục thay vì tham lam theo value/weight ratio).
    """
    n = len(items)
    if n == 0 or capacity <= 0:
        return [], 0.0

    dp = [[0.0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w_i, v_i = items[i - 1].weight, items[i - 1].value
        for w in range(0, capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if w_i <= w:
                candidate_value = dp[i - 1][w - w_i] + v_i
                if candidate_value > dp[i][w]:
                    dp[i][w] = candidate_value

    selected: List[KnapsackItem] = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(items[i - 1])
            w -= items[i - 1].weight
    selected.reverse()

    return selected, dp[n][capacity]


def build_knapsack_items(
    reranked_chunks: Sequence[Dict[str, Any]],
    partner_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    score_field: str = "rerank_score",
) -> List[KnapsackItem]:
    """
    Chuyển danh sách chunk đã rerank (reranker.py) thành item cho Knapsack.
    `partner_lookup`: map parent_chunk_id -> payload của sub-chunk B
    (partner không tự match query, được graph_router.py lấy về qua
    CROSS_SCENE_PAIR edge).
    """
    items: List[KnapsackItem] = []
    processed_parents = set()

    for chunk in reranked_chunks:
        payload = chunk.get("payload", chunk)
        parent_id = payload.get("parent_chunk_id")
        is_pair = bool(parent_id) and payload.get("is_cross_scene_subchunk", False)

        if is_pair:
            if parent_id in processed_parents:
                continue
            processed_parents.add(parent_id)

            keyframe_paths = [payload.get("keyframe_path")] if payload.get("keyframe_path") else []
            partner = (partner_lookup or {}).get(parent_id)
            if partner and partner.get("keyframe_path"):
                keyframe_paths.append(partner["keyframe_path"])

            combined_text = payload.get("text", "")
            if partner and partner.get("text"):
                combined_text = f"{combined_text} {partner['text']}".strip()

            items.append(KnapsackItem(
                item_id=parent_id,
                weight=2,
                value=float(chunk.get(score_field, chunk.get("s_total", 0.0))),
                text=combined_text,
                keyframe_paths=[p for p in keyframe_paths if p],
            ))
        else:
            chunk_id = payload.get("chunk_id") or chunk.get("point_id", "")
            keyframe_paths = [payload.get("keyframe_path")] if payload.get("keyframe_path") else []
            items.append(KnapsackItem(
                item_id=str(chunk_id),
                weight=1,
                value=float(chunk.get(score_field, chunk.get("s_total", 0.0))),
                text=payload.get("text", ""),
                keyframe_paths=[p for p in keyframe_paths if p],
            ))

    return items


class VLMGenerator:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_name: Optional[str] = None,
    ):
        from backend.src.common.config_loader import config
        self.device = device
        
        # Lấy từ config, ưu tiên biến model_name nếu được truyền vào
        _VLM_CFG = config.get("phase3_search", {})
        self.model_name = model_name or _VLM_CFG.get("vlm_model_id", "Qwen/Qwen2.5-VL-3B-Instruct-AWQ")
        
        self._model = None
        self._processor = None

    def load(self) -> None:
        """
        Nạp Qwen 2.5 VL 3B INT4 (AWQ) — GHIM CỐ ĐỊNH trên GPU theo nguyên
        tắc toàn cục, KHÔNG unload sau mỗi câu hỏi (khác các model Phase1/2).
        """
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_name)

    def unload(self) -> None:
        """
        Gỡ Qwen khỏi VRAM tường minh — dùng khi Phase 4 cần swap sang
        Prometheus-2 Judge (§4.2): del tham chiếu + gc.collect() +
        empty_cache(), vì chỉ gọi empty_cache() không đủ để giải phóng
        VRAM triệt để nếu còn tham chiếu treo.
        """
        if self._model is not None:
            del self._model
            self._model = None
        self._processor = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _build_messages(self, query: str, selected_items: Sequence[KnapsackItem]) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        context_texts = []
        images: List[Image.Image] = []
        content: List[Dict[str, Any]] = []

        for item in selected_items:
            for kf_path in item.keyframe_paths:
                if kf_path and Path(kf_path).is_file():
                    img = Image.open(kf_path).convert("RGB")
                    images.append(img)
                    content.append({"type": "image", "image": kf_path})
            context_texts.append(item.text)

        joined_context = "\n---\n".join(t for t in context_texts if t)
        prompt = (
            "Bạn là trợ lý trả lời câu hỏi dựa trên ngữ cảnh video được cung cấp "
            "(văn bản bóc băng + hình ảnh keyframe). Chỉ trả lời dựa trên ngữ cảnh, "
            "không bịa đặt thông tin ngoài phạm vi.\n\n"
            f"Ngữ cảnh:\n{joined_context}\n\nCâu hỏi: {query}"
        )
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        return messages, images

    def generate(
        self,
        query: str,
        reranked_chunks: Sequence[Dict[str, Any]],
        max_context_images: int = 4,
        partner_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Pipeline hoàn chỉnh: build Knapsack item -> chọn tối ưu bằng DP ->
        dựng prompt đa phương thức -> gọi Qwen 2.5 VL -> trả JSON tinh gọn
        (status, answer, trake) theo chuẩn I/O của spec §0.
        """
        if self._model is None:
            self.load()

        items = build_knapsack_items(reranked_chunks, partner_lookup)
        selected_items, total_value = knapsack_select(items, capacity=max_context_images)

        messages, images = self._build_messages(query, selected_items)

        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_prompt],
            images=images if images else None,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        answer_text = self._processor.batch_decode(
            generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

        # Giải phóng activation tạm thời của lượt suy luận này (không unload model)
        del inputs, generated_ids, generated_trimmed
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        trake = [
            {
                "chunk_id": item.item_id,
                "timestamp": None,  # gắn lại timestamp thực từ payload gốc ở tầng gọi nếu cần
                "keyframe_path": item.keyframe_paths[0] if item.keyframe_paths else None,
                "weight_used": item.weight,
                "knapsack_value": round(item.value, 5),
            }
            for item in selected_items
        ]

        return {
            "status": "ok",
            "answer": answer_text,
            "trake": trake,
            "knapsack_total_value": round(total_value, 5),
            "images_slots_used": sum(i.weight for i in selected_items),
            "images_slots_capacity": max_context_images,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3 VLM Generator (standalone test).")
    parser.add_argument("--query", required=True)
    parser.add_argument("--reranked_json", required=True, help="JSON list output từ reranker.py")
    parser.add_argument("--max_context_images", type=int, default=4)
    args = parser.parse_args()

    with open(args.reranked_json, "r", encoding="utf-8") as f:
        reranked_chunks = json.load(f)

    generator = VLMGenerator()
    result = generator.generate(args.query, reranked_chunks, max_context_images=args.max_context_images)
    print(json.dumps(result, ensure_ascii=False, indent=2))