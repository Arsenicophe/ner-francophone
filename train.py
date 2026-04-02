"""
Pipeline d'entrainement du modele BiLSTM-NER sur WikiNER-FR.
Usage : python train.py
"""

import json
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from model import BiLSTMNER, PAD_TAG_ID, TAG_TO_ID

# ─── Hyperparametres ─────────────────────────────────────────────────────────

EMBEDDING_DIM = 128
HIDDEN_DIM = 128
DROPOUT = 0.3
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 15
PATIENCE = 3          # early stopping
MIN_FREQ = 2          # frequence minimale pour inclure un mot dans le vocabulaire
MAX_SEQ_LEN = 128     # longueur maximale des sequences
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data")
SAVE_PATH = Path("ner_model.pt")


# ─── Vocabulaire ─────────────────────────────────────────────────────────────

class Vocabulary:
    """Vocabulaire mot -> index avec tokens speciaux."""

    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self):
        self.word2idx: dict[str, int] = {self.PAD: 0, self.UNK: 1}
        self.idx2word: dict[int, str] = {0: self.PAD, 1: self.UNK}

    def build(self, sentences: list[list[str]], min_freq: int = 2) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for tokens in sentences:
            counter.update(t.lower() for t in tokens)
        for word, freq in counter.items():
            if freq >= min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        return self

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.word2idx.get(t.lower(), 1) for t in tokens]

    def __len__(self) -> int:
        return len(self.word2idx)

    def state_dict(self) -> dict:
        return {"word2idx": self.word2idx}

    def load_state_dict(self, state: dict) -> "Vocabulary":
        self.word2idx = state["word2idx"]
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        return self


# ─── Dataset ─────────────────────────────────────────────────────────────────

class NERDataset(Dataset):
    """Dataset NER : chaque element est (token_ids, tag_ids)."""

    def __init__(self, path: Path, vocab: Vocabulary, max_len: int = MAX_SEQ_LEN):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for item in raw:
            tokens = item["tokens"][:max_len]
            tags = item["tags"][:max_len]
            token_ids = torch.tensor(vocab.encode(tokens), dtype=torch.long)
            tag_ids = torch.tensor(tags, dtype=torch.long)
            self.samples.append((token_ids, tag_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """Padding dynamique par batch."""
    tokens, tags = zip(*batch)
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=PAD_TAG_ID)
    return tokens_padded, tags_padded


# ─── Metriques ───────────────────────────────────────────────────────────────

def compute_metrics(
    model: BiLSTMNER, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Calcule la loss et l'accuracy masquee (hors padding)."""
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TAG_ID)
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for tokens, tags in loader:
            tokens, tags = tokens.to(device), tags.to(device)
            logits = model(tokens)
            loss = criterion(logits.view(-1, logits.size(-1)), tags.view(-1))
            total_loss += loss.item() * tokens.size(0)
            preds = logits.argmax(dim=-1)
            mask = tags != PAD_TAG_ID
            correct += (preds[mask] == tags[mask]).sum().item()
            total += mask.sum().item()
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": correct / total if total > 0 else 0.0,
    }


# ─── Entrainement ───────────────────────────────────────────────────────────

def train() -> None:
    print(f"Device : {DEVICE}")

    # Charger les donnees brutes pour construire le vocabulaire
    with open(DATA_DIR / "train.json", encoding="utf-8") as f:
        train_raw = json.load(f)

    vocab = Vocabulary().build([item["tokens"] for item in train_raw], min_freq=MIN_FREQ)
    print(f"Vocabulaire : {len(vocab)} mots")

    # Datasets et DataLoaders
    train_ds = NERDataset(DATA_DIR / "train.json", vocab)
    val_ds = NERDataset(DATA_DIR / "val.json", vocab)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # Modele
    model = BiLSTMNER(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametres : {total_params:,}")
    print(model)

    # Optimiseur et loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TAG_ID)

    # Boucle d'entrainement
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for tokens, tags in train_loader:
            tokens, tags = tokens.to(DEVICE), tags.to(DEVICE)
            optimizer.zero_grad()
            logits = model(tokens)
            loss = criterion(logits.view(-1, logits.size(-1)), tags.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item() * tokens.size(0)

        train_loss = epoch_loss / len(train_ds)
        val_metrics = compute_metrics(model, val_loader, DEVICE)
        scheduler.step(val_metrics["loss"])
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab_state": vocab.state_dict(),
                    "hyperparams": {
                        "embedding_dim": EMBEDDING_DIM,
                        "hidden_dim": HIDDEN_DIM,
                        "dropout": DROPOUT,
                        "num_tags": len(TAG_TO_ID),
                    },
                },
                SAVE_PATH,
            )
            print(f"  -> Modele sauvegarde ({SAVE_PATH})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping a l'epoch {epoch}")
                break

    # Evaluation finale sur le test
    test_ds = NERDataset(DATA_DIR / "test.json", vocab)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = compute_metrics(model, test_loader, DEVICE)
    print(
        f"\nResultats sur le test set :\n"
        f"  Loss     : {test_metrics['loss']:.4f}\n"
        f"  Accuracy : {test_metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    train()
