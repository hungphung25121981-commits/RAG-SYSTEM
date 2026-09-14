import os
import time
import json
import asyncio
from typing import List, Dict,Any,Optional
from fastapi import FastAPI,Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app=FastAPI(title="VIRAGPT",description="ENGINEE API BACKEND FOR VIDEO RAG SYSTEM",version="1.0.0")




# cau hinh CORS de cho phep truy cap tu cac domain khac nhau
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các domain truy cập
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức HTTP
    allow_headers=["*"],  # Cho phép tất cả các header
)



class ChatMessage(BaseModel):
   model: str="video-rag-v1"
   messages: List[ChatMessage]
   stream: Optional[bool]
   temperature: Optional[float] = 0.7

@app.get("/")
def health_check():
    """Kiểm tra server sống hay chết."""
    return {"status": "ok", "service": "Video RAG API Engine"}

@app.get("/v1/models")
def list_models():
    """Endpoint trả về danh sách Models để Open WebUI hiển thị trong dropdown select."""
    return {
        "object": "list",
        "data": [
            {
                "id": "video-rag-v1",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "video-rag-admin"
            }
        ]
    }

async def dummy_rag_streamer(prompt_text: str):
    """
    Generator giả lập Server gửi phản hồi dạng Stream (SSE - Server-Sent Events) 
    chuẩn OpenAI cho Open WebUI nhận chữ nào hiện chữ đó.
    (Sau này sẽ nối logic Phase 3 RAG & Qwen-2.5-VL vào đây).
    """
    model_id = "video-rag-v1"
    created_time = int(time.time())
    
    # Giả lập response mẫu để test giao diện
    dummy_response = f"Tao đã nhận được câu hỏi của mày: '{prompt_text}'. Hạt nhân RAG đang khởi động để truy vấn Keyframes từ Qdrant..."
    words = dummy_response.split(" ")

    for word in words:
        chunk = {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": word + " "},
                    "finish_reason": None
                }
            ]
        }
        # Format chuẩn SSE: `data: <json>\n\n`
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.08) # Giả lập delay sinh word

    # Chunk kết thúc
    final_chunk = {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Endpoint chính xử lý Chat Completion từ Open WebUI.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")
    
    user_prompt = request.messages[-1].content

    # Mặc định trả về Streaming Response theo chuẩn SSE
    return StreamingResponse(
        dummy_rag_streamer(user_prompt),
        media_type="text/event-stream"
    )