from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.model import TinyLlamaModel
from src.prompts import SYSTEM_PROMPT

app = FastAPI()
model = TinyLlamaModel()

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
def index():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
def chat(req: ChatRequest):
    prompt = f"""
{SYSTEM_PROMPT}

User:
{req.message}

Assistant:
"""
    output = model.generate(prompt)
    response = output.split("Assistant:")[-1].strip()
    return {"response": response}
