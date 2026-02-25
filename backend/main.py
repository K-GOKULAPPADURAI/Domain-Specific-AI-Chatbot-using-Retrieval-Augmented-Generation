import os
import shutil
import json
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from rag_engine import get_rag_engine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    file_paths = []
    saved_names = []
    for file in files:
        if not file.filename:
            continue
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in (".pdf", ".txt"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.filename}. Only PDF and TXT files are accepted.",
            )
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(file_path)
        saved_names.append(file.filename)

    if not file_paths:
        raise HTTPException(status_code=400, detail="No valid files provided.")

    try:
        engine = get_rag_engine()
        result = engine.load_documents(file_paths)
        message = f"Successfully indexed {result['indexed']} chunks from {result['files']} file(s)."
        response = {"message": message, "files": saved_names}
        if "warnings" in result:
            response["warnings"] = result["warnings"]
        return response
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.post("/chat")
async def chat(question: str = Form(...)):
    try:
        engine = get_rag_engine()
        answer = engine.query(question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(question: str = Form(...)):
    engine = get_rag_engine()

    def sse_data(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def generate():
        try:
            # SSE padding helps some clients/proxies flush the stream immediately.
            yield ":" + (" " * 2048) + "\n\n"
            yield sse_data({"status": "started"})
            await asyncio.sleep(0)
            for token in engine.query_stream(question):
                yield sse_data({"token": token})
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield sse_data({"error": str(e)})
            yield "data: [DONE]\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # Helps disable proxy buffering (notably on nginx-based deployments).
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(generate(), media_type="text/event-stream", headers=headers)


@app.post("/chat/clear")
async def clear_chat():
    engine = get_rag_engine()
    engine.clear_history()
    return {"message": "Chat history cleared."}


# --- Document management ---


@app.get("/documents")
async def list_documents():
    if not os.path.exists(UPLOAD_DIR):
        return {"files": []}
    files = []
    for name in sorted(os.listdir(UPLOAD_DIR)):
        path = os.path.join(UPLOAD_DIR, name)
        if os.path.isfile(path):
            files.append({"name": name, "size": os.path.getsize(path)})
    return {"files": files}


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found.")
    os.remove(path)
    return {"message": f"Deleted {filename}."}


@app.delete("/documents")
async def delete_all_documents():
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR)
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
    # Reset the engine
    import rag_engine as re_mod
    re_mod.rag_engine = None
    return {"message": "All documents and index cleared."}


# --- Debug endpoints ---


@app.get("/debug/chunks")
async def get_debug_chunks():
    try:
        if os.path.exists("chunks_debug.json"):
            with open("chunks_debug.json", "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            return {"status": "success", "total_chunks": len(chunks_data), "chunks": chunks_data}
        return {"status": "info", "message": "No chunks created yet.", "chunks": []}
    except Exception as e:
        return {"status": "error", "message": str(e), "chunks": []}


@app.get("/debug/chunk-summary")
async def get_chunk_summary():
    try:
        if os.path.exists("chunks_debug.json"):
            with open("chunks_debug.json", "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            total_chars = sum(chunk["length"] for chunk in chunks_data)
            avg_length = total_chars // len(chunks_data) if chunks_data else 0
            return {
                "status": "success",
                "total_chunks": len(chunks_data),
                "total_characters": total_chars,
                "average_chunk_length": avg_length,
                "min_chunk_length": min(chunk["length"] for chunk in chunks_data) if chunks_data else 0,
                "max_chunk_length": max(chunk["length"] for chunk in chunks_data) if chunks_data else 0,
                "unique_sources": list(set(chunk["source"] for chunk in chunks_data)),
            }
        return {"status": "info", "message": "No chunks created yet.", "total_chunks": 0}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# --- Serve frontend static files (production) ---

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

if os.path.isdir(STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the React SPA for any non-API route."""
        file_path = os.path.join(STATIC_DIR, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
