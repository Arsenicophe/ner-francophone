"""
BiLSTM pour la Reconnaissance d'Entites Nommees (NER) en francais.
Entraine sur le corpus WikiNER-FR (Wikipedia francophone).
"""

import torch
import torch.nn as nn


TAG_NAMES = ["O", "LOC", "PER", "MISC", "ORG"]
TAG_TO_ID = {tag: i for i, tag in enumerate(TAG_NAMES)}
ID_TO_TAG = {i: tag for tag, i in TAG_TO_ID.items()}
PAD_TAG_ID = -100  # ignore_index natif de CrossEntropyLoss


class BiLSTMNER(nn.Module):
    """Modele BiLSTM pour le NER.

    Architecture :
        Embedding -> Dropout -> BiLSTM (2 couches) -> Dropout -> Linear

    Args:
        vocab_size: taille du vocabulaire (incluant <PAD> a l'index 0)
        embedding_dim: dimension des embeddings
        hidden_dim: dimension cachee du LSTM (par direction)
        num_tags: nombre de classes NER
        dropout: taux de dropout
        padding_idx: index du token de padding dans l'embedding
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_tags: int = len(TAG_NAMES),
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.dropout_emb = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.dropout_lstm = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor d'indices de tokens, shape (batch, seq_len)
        Returns:
            logits: shape (batch, seq_len, num_tags)
        """
        emb = self.dropout_emb(self.embedding(x))
        lstm_out, _ = self.lstm(emb)
        lstm_out = self.dropout_lstm(lstm_out)
        logits = self.classifier(lstm_out)
        return logits
