# Domain-Specific AI Chatbot (RAG)

A full-stack Retrieval-Augmented Generation (RAG) chatbot that lets you upload domain documents (PDF/TXT) and ask grounded questions against that content.

## What This Project Does

- Uploads PDF/TXT files from a React UI.
- Extracts text and chunks it on the backend.
- Builds/updates a FAISS vector index using Hugging Face embeddings.
- Retrieves relevant chunks per user question.
- Sends retrieved context to an OpenRouter-hosted LLM for final answers.

## 🎥 Demo & Results

### 🖼️ Demo Images

#### 1️⃣ Demo Image 1
![Demo 1](demo/pic%20(1).jpeg)

#### 2️⃣ Demo Image 2
![Demo 2](demo/pic%20(2).jpeg)

#### 3️⃣ Demo Image 3
![Demo 3](demo/pic%20(3).jpeg)

---

### 🎬 Demo Video

<video width="700" controls>
  <source src="demo/demo-video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Tech Stack

- Frontend: React + Vite
- Backend: FastAPI
- RAG: LangChain + FAISS + sentence-transformers
- LLM Provider: OpenRouter (via `ChatOpenAI` with custom `base_url`)

## Repository Structure

```text
.
├── backend/
│   ├── main.py              # FastAPI app (/upload, /chat)
│   ├── rag_engine.py        # RAG pipeline + OpenRouter LLM config
│   ├── requirements.txt
│   └── static/              # Built frontend served in deployment
├── frontend/
│   ├── src/App.jsx          # Chat UI + upload flow
│   ├── src/index.css        # Main app styling
│   └── package.json
├── render.yaml              # Render deployment config
└── README.md
```

## How It Works

1. User uploads documents from the UI.
2. Backend loads files (`PyPDFLoader`/`TextLoader`).
3. Text is split into chunks (`RecursiveCharacterTextSplitter`).
4. Chunks are embedded (`all-MiniLM-L6-v2`) and stored in FAISS.
5. On chat, top-k chunks are retrieved and passed to the LLM prompt.
6. LLM returns a concise answer grounded in retrieved context.

## Prerequisites

- Python 3.11+
- Node.js 20+
- npm
- OpenRouter API key

## Local Development

### 1. Backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in `backend/`:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
```

Run backend:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend setup

In a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## API Endpoints

### `POST /upload`

- Form field: `files` (supports multiple files)
- Accepts: `.pdf`, `.txt`
- Response: indexing success message

Example:

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@/absolute/path/to/doc1.pdf" \
  -F "files=@/absolute/path/to/doc2.txt"
```

### `POST /chat`

- Form field: `question`
- Response: `{ "answer": "..." }`

Example:

```bash
curl -X POST http://localhost:8000/chat \
  -F "question=What are the key points from the uploaded docs?"
```

## Deployment (Render)

This repo includes `render.yaml` that:

1. Builds frontend with Vite.
2. Copies `frontend/dist` into `backend/static`.
3. Starts FastAPI via Uvicorn.

Important config note:

- Main RAG backend (`backend/rag_engine.py`) uses `OPENROUTER_API_KEY`.
- `render.yaml` currently defines `XAI_API_KEY`, which is used by `backend/testxapi.py` (alternate test endpoint), not by the main RAG flow.
- For production RAG deployment, add/set `OPENROUTER_API_KEY` in Render environment variables.

## Known Limitations

- Scanned/image-only PDFs without extractable text will fail unless OCR is added.
- FAISS index is local filesystem state (not a managed vector database).
- Current CORS is open (`allow_origins=["*"]`) for development convenience.

## Suggested Next Improvements

- Add OCR fallback for scanned PDFs.
- Add source citations/chunk references in responses.
- Add authentication and per-user document isolation.
- Persist vectors in a managed store (pgvector, Pinecone, Weaviate, etc.).
- Add tests for `/upload`, `/chat`, and retrieval quality.
