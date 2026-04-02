"""
Serveur FastAPI pour l'inference NER en temps reel.
Usage : uvicorn app:app --reload --port 8000
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from inference import load_model, predict
from model import TAG_NAMES

app = FastAPI(title="NER Francophone", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modele au demarrage
model, vocab = load_model()


class NERRequest(BaseModel):
    text: str


class TokenPrediction(BaseModel):
    token: str
    tag: str


class NERResponse(BaseModel):
    predictions: list[TokenPrediction]
    tags: list[str] = TAG_NAMES


@app.post("/predict", response_model=NERResponse)
def predict_ner(request: NERRequest):
    preds = predict(request.text, model, vocab)
    return NERResponse(predictions=[TokenPrediction(**p) for p in preds])


@app.get("/")
def serve_frontend():
    return FileResponse(Path(__file__).parent / "index.html")
