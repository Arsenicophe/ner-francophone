"""
Inference NER : charge le modele entraine et predit les entites nommees.
Usage : python inference.py "Emmanuel Macron a visite la Tour Eiffel a Paris"
"""

import sys
from pathlib import Path

import torch

from model import BiLSTMNER, ID_TO_TAG

SAVE_PATH = Path("ner_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vocabulary:
    """Vocabulaire minimal pour l'inference."""

    def __init__(self):
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.word2idx.get(t.lower(), 1) for t in tokens]

    def __len__(self) -> int:
        return len(self.word2idx)


def load_model(path: Path = SAVE_PATH) -> tuple[BiLSTMNER, Vocabulary]:
    """Charge le modele et le vocabulaire depuis un checkpoint."""
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    vocab = Vocabulary()
    # Supporte les deux formats de checkpoint (local et Colab)
    if "vocab_word2idx" in checkpoint:
        vocab.word2idx = checkpoint["vocab_word2idx"]
    elif "vocab_state" in checkpoint:
        vocab.word2idx = checkpoint["vocab_state"]["word2idx"]
    vocab.idx2word = {i: w for w, i in vocab.word2idx.items()}
    hp = checkpoint["hyperparams"]
    model = BiLSTMNER(
        vocab_size=hp.get("vocab_size", len(vocab)),
        embedding_dim=hp["embedding_dim"],
        hidden_dim=hp["hidden_dim"],
        num_tags=hp["num_tags"],
        dropout=0.0,  # pas de dropout en inference
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, vocab


def predict(sentence: str, model: BiLSTMNER, vocab: Vocabulary) -> list[dict]:
    """Predit les tags NER pour une phrase.

    Returns:
        Liste de dicts {"token": str, "tag": str} pour chaque mot.
    """
    tokens = sentence.split()
    if not tokens:
        return []
    token_ids = torch.tensor([vocab.encode(tokens)], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = model(token_ids)
    pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()
    return [
        {"token": tok, "tag": ID_TO_TAG[tag_id]}
        for tok, tag_id in zip(tokens, pred_ids)
    ]


def format_predictions(predictions: list[dict]) -> str:
    """Affichage colore des predictions dans le terminal."""
    colors = {"PER": "\033[94m", "LOC": "\033[92m", "ORG": "\033[93m", "MISC": "\033[95m"}
    reset = "\033[0m"
    parts = []
    for p in predictions:
        tag = p["tag"]
        if tag != "O":
            color = colors.get(tag, "")
            parts.append(f"{color}[{p['token']}]({tag}){reset}")
        else:
            parts.append(p["token"])
    return " ".join(parts)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python inference.py \"Votre phrase ici\"")
        sys.exit(1)

    sentence = " ".join(sys.argv[1:])
    model, vocab = load_model()
    preds = predict(sentence, model, vocab)
    print(format_predictions(preds))
