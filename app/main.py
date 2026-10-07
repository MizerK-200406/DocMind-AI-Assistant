"""Ứng dụng FastAPI chính của DocMind.

Bao gồm các tính năng cốt lõi:
- Quản lý phiên trò chuyện và lưu trữ lịch sử tin nhắn trong SQLite.
- Endpoint POST /chat: Hỏi đáp đồng bộ với LLM, tự động ghi nhận hội thoại vào DB.
- Endpoint POST /chat/stream: Truyền phát phản hồi theo thời gian thực qua Server-Sent Events (SSE).
- Endpoint GET /conversations/{conv_id}: Truy vấn toàn bộ lịch sử tin nhắn của một cuộc hội thoại.
- Endpoint POST /extract: Phân tích và trích xuất dữ liệu có cấu trúc từ văn bản theo Pydantic schema.
- Endpoint POST /ask: Hỏi đáp RAG trên tài liệu đã nạp, kèm trích dẫn số trang và nguồn tham khảo.
- Endpoint GET /documents: Liệt kê danh sách các tài liệu đã được nạp vào vector store.
- Endpoint GET /health: Health-check endpoint phục vụ giám sát container và uptime.
- Bộ Exception Handlers toàn diện: Bắt và chuẩn hóa các mã lỗi từ phía OpenAI/NVIDIA API.
"""

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from pydantic import BaseModel, Field

from app import db, llm, structured
from app.config import settings
from app.rag import pipeline, store
from app.schemas import DocumentInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docmind")

SYSTEM_PROMPT = settings.SYSTEM_PROMPT
# Giới hạn số tin nhắn lịch sử tối đa đưa vào context window của LLM
MAX_HISTORY = settings.MAX_HISTORY


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời (Lifespan) của ứng dụng FastAPI."""
    db.init_db()
    yield


app = FastAPI(
    title="DocMind",
    description="Hệ thống trợ lý AI thông minh hỗ trợ trò chuyện ngữ cảnh và trích xuất tài liệu.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RateLimitError)
async def rate_limit_handler(request: Request, exc: RateLimitError):
    """Xử lý lỗi vượt quá giới hạn tần suất gọi API (Rate Limit) sau khi đã retry."""
    logger.warning("LLM rate limit sau khi đã retry: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"error": "llm_busy", "detail": "Hệ thống đang quá tải, vui lòng thử lại sau."},
        headers={"Retry-After": "10"},
    )


@app.exception_handler(APITimeoutError)
async def timeout_handler(request: Request, exc: APITimeoutError):
    """Xử lý lỗi timeout khi LLM không phản hồi kịp trong khoảng thời gian quy định."""
    return JSONResponse(
        status_code=504,
        content={"error": "llm_timeout", "detail": "LLM phản hồi khá lâu, yêu cầu hết thời gian chờ."},
    )


@app.exception_handler(APIConnectionError)
async def connection_handler(request: Request, exc: APIConnectionError):
    """Xử lý lỗi mất kết nối mạng hoặc không thể liên lạc tới endpoint của LLM."""
    return JSONResponse(
        status_code=502,
        content={"error": "llm_unreachable", "detail": "Không kết nối được tới dịch vụ LLM."},
    )


@app.exception_handler(AuthenticationError)
async def auth_handler(request: Request, exc: AuthenticationError):
    """Xử lý lỗi sai API Key hoặc tài khoản không được cấp phép truy cập dịch vụ LLM."""
    logger.error("API key LLM không hợp lệ hoặc hết hạn: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "server_misconfigured", "detail": "Lỗi cấu hình hệ thống (API Key)."},
    )


class ChatRequest(BaseModel):
    """Lược đồ dữ liệu yêu cầu cho endpoint trò chuyện."""
    message: str = Field(min_length=1, max_length=4000, description="Nội dung tin nhắn người dùng gửi")
    conversation_id: str | None = Field(default=None, description="ID cuộc hội thoại (để trống nếu bắt đầu hội thoại mới)")


def prepare_conversation(req: ChatRequest) -> tuple[str, list[dict]]:
    """Tạo hoặc xác thực phiên hội thoại, lấy lịch sử và ghép thành payload gửi LLM.
    
    Lưu ý: Tin nhắn của user và assistant CHƯA được lưu vào DB ở bước này.
    Chúng chỉ được ghi nhận đồng thời thông qua `db.add_exchange` sau khi LLM phản hồi thành công,
    nhằm đảm bảo không gây ô nhiễm lịch sử hội thoại nếu xảy ra lỗi kết nối.
    
    Returns:
        tuple[str, list[dict]]: (conv_id, danh sách tin nhắn bao gồm system + history + new user message).
    """
    if req.conversation_id:
        if not db.conversation_exists(req.conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        conv_id = req.conversation_id
    else:
        conv_id = db.create_conversation()

    # Dành 1 slot cho tin nhắn mới của người dùng trong context window
    history = db.get_messages(conv_id, limit=max(MAX_HISTORY - 1, 0))
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": req.message}]
    )
    return conv_id, messages


@app.post("/chat")
def chat(req: ChatRequest):
    """Endpoint trò chuyện tiêu chuẩn (Đồng bộ - Blocking)."""
    conv_id, messages = prepare_conversation(req)
    resp = llm.chat_completion(messages, temperature=0.2)
    answer = resp.choices[0].message.content
    db.add_exchange(conv_id, req.message, answer)
    return {
        "conversation_id": conv_id,
        "answer": answer,
        "usage": resp.usage.model_dump() if resp.usage else None,
    }


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Endpoint trò chuyện truyền phát thời gian thực (Streaming qua Server-Sent Events)."""
    conv_id, messages = prepare_conversation(req)

    def event_generator():
        parts: list[str] = []
        try:
            yield f"data: {json.dumps({'conversation_id': conv_id})}\n\n"
            stream = llm.chat_completion(messages, temperature=0.2, stream=True)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    parts.append(token)
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            # Lưu lại câu trả lời tích lũy ngay cả khi client ngắt kết nối giữa chừng
            if parts:
                db.add_exchange(conv_id, req.message, "".join(parts))

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/conversations/{conv_id}")
def get_conversation(conv_id: str):
    """Lấy danh sách toàn bộ tin nhắn thuộc về một cuộc hội thoại cụ thể."""
    if not db.conversation_exists(conv_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation_id": conv_id, "messages": db.get_messages(conv_id)}


class ExtractRequest(BaseModel):
    """Lược đồ dữ liệu yêu cầu cho endpoint trích xuất thông tin tài liệu."""
    text: str = Field(min_length=1, max_length=20000, description="Đoạn văn bản cần trích xuất thông tin")


@app.post("/extract", response_model=DocumentInfo)
def extract_document(req: ExtractRequest):
    """Phân tích văn bản tự do và trích xuất thành đối tượng có cấu trúc (DocumentInfo).
    
    Bao gồm: tiêu đề, phân loại tài liệu, ngôn ngữ, tóm tắt, ý chính và cảm xúc.
    """
    try:
        return structured.extract(
            req.text,
            DocumentInfo,
            instruction="Bạn là hệ thống phân tích văn bản chuyên nghiệp. Đọc kỹ văn bản và trích xuất chính xác thông tin.",
        )
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/health")
def health():
    """Endpoint kiểm tra sức khỏe hệ thống (Health Check)."""
    return {"status": "OK"}


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)
    source: str | None = None


class SourceItem(BaseModel):
    id: int
    source: str
    page: int
    page_label: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    answered: bool
    sources: list[SourceItem]

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Hỏi đáp trên tài liệu dựa trên pipeline RAG.
    
    Hỗ trợ lọc theo tài liệu cụ thể thông qua trường `source`.
    Trả về câu trả lời kèm danh sách các đoạn trích dẫn nguồn, số trang và điểm tin cậy.
    """
    source = req.source or None
    if source and source not in store.list_sources():
        raise HTTPException(
            status_code=404,
            detail=f"Không có tài liệu '{source}'. Xem danh sách ở GET /documents.",
        )
    return pipeline.answer_question(req.question, k=req.top_k, source=source)


@app.get("/documents")
def list_documents():
    """Liệt kê danh sách tất cả các tài liệu hiện có trong vector store."""
    return {"documents": store.list_sources()}
