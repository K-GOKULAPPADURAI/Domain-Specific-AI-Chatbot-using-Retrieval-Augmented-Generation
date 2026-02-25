import os
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("XAI_API_KEY")

client = None
if API_KEY:
    client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1")


@app.post("/chat")
async def chat(question: str = Form(...)):
    if not API_KEY or not client:
        return {"error": "XAI_API_KEY not set. Please set the XAI_API_KEY environment variable."}

    response = client.chat.completions.create(
        model="grok-3-latest",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer the user's question clearly and concisely."},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
    )

    answer = response.choices[0].message.content
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
