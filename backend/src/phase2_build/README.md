# Phase 2 — Vector & Graph Construction

## 1. Mục Tiêu

Phase 2 chuyển các chunk văn bản/ảnh từ Phase 1 thành vector số học, đẩy vào Qdrant để phục vụ tìm kiếm ngữ nghĩa, đồng thời xây dựng Knowledge Graph (Kuzu) để nắm bắt quan hệ nhân quả/thời gian giữa các sự kiện trong video.

**Input:** chunk từ `data/temp_workspace/` (Phase 1).
**Output:** vector trong Qdrant Collection theo domain, đồ thị `.kuzu` trong `data/kuzu_graph/`.

## 2. Sơ Đồ Luồng Xử Lý

```
Chunks (từ Phase 1)
         │
         ▼
┌───────────────────────────────────────────────────────┐
│  Strict Sequential Batching (bắt buộc)                │
│                                                       │
│  Load SigLIP 2 → Encode toàn bộ ảnh trong batch       │
│  → Unload + empty_cache()                             │
│              ↓ (hoàn tất 100% trước khi qua bước sau) │
│  Load BGE-M3 → Encode toàn bộ text trong batch        │
│  → Unload + empty_cache()                             │
└─────────────────────┬─────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│  Upsert vào Qdrant                          │
│  payload.commit_status = "pending"          │
└─────────────────────┬───────────────────────┘
                       ▼
              Toàn batch ghi xong?
                 │            │
               Có            Không (sập giữa chừng)
                 │            │
                 ▼            ▼
     commit_status = "committed"   giữ "pending"
                 │                  → Phase 0 dọn ở lần resume sau
                 ▼
      ┌───────────────────────────────────┐
      │ Kuzu đọc chunk đã committed       │
      │ → tạo Node (Thực thể/Sự kiện)     │
      │ → tạo Edge (Thời gian/Không gian) │
      │ → lưu incremental vào .kuzu       │
      └───────────────────────────────────┘
```

## 3. Công Nghệ & Thuật Toán Áp Dụng

| Thành phần | Công nghệ | Ghi chú kỹ thuật |
|---|---|---|
| Embedding ảnh | SigLIP 2 | Chỉ chạy ở Phase 2, không xen kẽ với BGE-M3 |
| Embedding text | BGE-M3 | Sinh cả Dense Vector lẫn Sparse Vector (BM25-style) |
| Vector DB | Qdrant (local) | Kiến trúc Tri-Search: Dense + Sparse + Metadata trong cùng payload |
| Cô lập dữ liệu | Qdrant Collection theo domain | Mỗi `--save_domain` là một Collection vật lý riêng biệt — không dùng payload filter chung |
| Graph DB | Kuzu | Thay thế NetworkX in-memory; hỗ trợ update gia tăng (incremental), lưu trực tiếp trên đĩa |
| Chống fragmentation | Strict Sequential Batching | `empty_cache()` chỉ gọi ở biên giới chuyển giao model, không gọi trong inner-loop |
| Toàn vẹn giao dịch | `commit_status` payload field | Graph chỉ đọc dữ liệu đã `committed`, không build song song với ingest |

## 4. Cấu Trúc Payload Qdrant

```json
{
  "chunk_id": "c_0042",
  "parent_chunk_id": "c_0042",
  "vid_hash": "a1b2c3d4e5f6...",
  "timestamp": "01:29:03",
  "keyframe_path": "data/temp_workspace/kf_0042.jpg",
  "dense_vector_text": "[...]",
  "dense_vector_image": "[...]",
  "sparse_vector_bm25": "[...]",
  "metadata_yolo": ["person", "ball", "goal"],
  "commit_status": "committed"
}
```

## 5. Cách Chạy Độc Lập

```bash
# Build Vector và Graph cho dữ liệu đã extract ở Phase 1
python cli_pipeline.py --run_phase 2 --build_graph

# Chỉ định đường dẫn Qdrant local cụ thể
python cli_pipeline.py --run_phase 2 --qdrant_path ./data/qdrant_db --build_graph
```

**Output mẫu (JSON):**
```json
{
  "status": "success",
  "vectors_upserted": 214,
  "vectors_committed": 214,
  "vectors_pending_rolled_back": 0,
  "graph": {
    "nodes_created": 58,
    "edges_created": 121,
    "storage_path": "data/kuzu_graph/football_analytics.kuzu"
  }
}
```

## 6. Lưu Ý Vận Hành

- Nếu `vectors_pending_rolled_back` khác 0 sau một lần chạy, nghĩa là lần chạy trước đó đã bị gián đoạn — đây là hành vi rollback đúng như thiết kế, không phải lỗi.
- Domain mới (`--save_domain` chưa từng dùng) sẽ tự động tạo Collection Qdrant mới, không cần khởi tạo thủ công.
- Nếu cần rebuild toàn bộ Graph từ đầu (ví dụ đổi schema Node/Edge), xóa file `.kuzu` tương ứng trong `data/kuzu_graph/` rồi chạy lại `--run_phase 2 --build_graph` — vector trong Qdrant không bị ảnh hưởng vì 2 hệ lưu trữ độc lập nhau.
