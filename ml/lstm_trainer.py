import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from ml.lstm_model import LSTMRiskModel
from utils.logger import get_logger

logger = get_logger(__name__)

MODELS_DIR = "models"
SEQUENCE_LENGTH = 7   # use last 7 days as input sequence
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
FEATURES = ["sentiment_compound", "risk_score", "keyword_risk"]


class SequenceDataset(Dataset):
    """
    Builds sliding window sequences from daily time series.

    Given 30 days of data with sequence_length=7:
        - Sequence 1: days 1-7   → label for day 8
        - Sequence 2: days 2-8   → label for day 9
        - etc.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_sequences(df: pd.DataFrame, seq_len: int = SEQUENCE_LENGTH):
    """
    Converts article-level DataFrame into sequences for LSTM.

    Steps:
    1. Aggregate to daily averages
    2. Build sliding window sequences
    3. Return X (sequences) and y (labels)
    """
    df = df.copy()
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published", "price_label"] + FEATURES)
    df["date"] = df["published"].dt.date

    # Daily aggregation
    daily = df.groupby("date").agg(
        sentiment_compound=("sentiment_compound", "mean"),
        risk_score=("risk_score", "mean"),
        keyword_risk=("keyword_risk", "mean"),
        price_label=("price_label", lambda x: x.mode()[0])
    ).reset_index()

    daily = daily.sort_values("date").reset_index(drop=True)

    if len(daily) < seq_len + 1:
        return None, None, None, None

    # Encode labels
    le = LabelEncoder()
    daily["label_encoded"] = le.fit_transform(daily["price_label"])

    # Scale features
    scaler = StandardScaler()
    feature_vals = scaler.fit_transform(daily[FEATURES])

    # Build sequences
    X_seqs, y_seqs = [], []
    for i in range(len(daily) - seq_len):
        seq = feature_vals[i: i + seq_len]
        label = daily["label_encoded"].iloc[i + seq_len]
        X_seqs.append(seq)
        y_seqs.append(label)

    X_arr = np.array(X_seqs)
    y_arr = np.array(y_seqs)

    logger.info(f"Sequences built: {len(X_seqs)} | Sequence length: {seq_len}")
    logger.info(f"Classes: {le.classes_}")

    return X_arr, y_arr, le, scaler


def train_lstm(df: pd.DataFrame) -> dict:
    """
    Full LSTM training pipeline.
    Returns metrics dict and saves model + scaler + encoder.
    """
    logger.info("\n" + "=" * 40)
    logger.info("Training: LSTM (PyTorch)")
    logger.info("=" * 40)

    X, y, le, scaler = build_sequences(df)

    if X is None:
        logger.error(f"Not enough daily data for LSTM sequences (need {SEQUENCE_LENGTH + 1}+ days)")
        return {"success": False, "error": "Not enough sequential data"}

    # Train/test split (80/20 — no shuffle, preserve time order)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if len(X_train) == 0 or len(X_test) == 0:
        logger.error("Not enough data for train/test split")
        return {"success": False, "error": "Insufficient data after split"}

    logger.info(f"Train sequences: {len(X_train)} | Test sequences: {len(X_test)}")

    train_dataset = SequenceDataset(X_train, y_train)
    test_dataset = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    num_classes = len(le.classes_)
    model = LSTMRiskModel(
        input_size=len(FEATURES),
        hidden_size=64,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.2
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    best_val_loss = float("inf")
    best_model_state = None
    train_losses = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step()

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    report = classification_report(
        all_labels, all_preds,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )

    logger.info(f"\nLSTM Results:")
    logger.info(f"  Accuracy : {accuracy:.4f}")
    logger.info(f"  F1       : {f1:.4f}")

    # Save model, scaler, encoder
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "lstm_price_movement.pt")
    scaler_path = os.path.join(MODELS_DIR, "lstm_scaler.pkl")
    encoder_path = os.path.join(MODELS_DIR, "lstm_label_encoder.pkl")

    torch.save({
        "model_state": best_model_state,
        "input_size": len(FEATURES),
        "hidden_size": 64,
        "num_layers": 2,
        "num_classes": num_classes,
        "dropout": 0.2,
        "sequence_length": SEQUENCE_LENGTH,
        "features": FEATURES,
    }, model_path)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(le, f)

    logger.info(f"Saved: {model_path}")
    logger.info(f"Saved: {scaler_path}")
    logger.info(f"Saved: {encoder_path}")

    return {
        "success": True,
        "model": "LSTM",
        "accuracy": round(accuracy, 4),
        "f1_weighted": round(f1, 4),
        "classification_report": report,
        "classes": le.classes_.tolist(),
        "epochs_trained": EPOCHS,
        "best_train_loss": round(best_val_loss, 4),
        "model_path": model_path,
    }
