import torch
import numpy as np
import pickle
from typing import Optional, Dict
from ml.lstm_model import LSTMRiskModel
from utils.logger import get_logger

logger = get_logger(__name__)


class LSTMPredictor:
    """
    Loads trained LSTM model and runs inference on a single feature vector.

    Note: LSTM ideally needs a sequence of past days.
    For single-article prediction via API, we replicate the
    single article's features across the sequence length as a fallback.
    This is less accurate than a true sequence but still functional.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoder = None
        self.config = None
        self._ready = False

    def load(self) -> None:
        try:
            checkpoint = torch.load("models/lstm_price_movement.pt",
                                    map_location=torch.device("cpu"))
            self.config = checkpoint

            self.model = LSTMRiskModel(
                input_size=checkpoint["input_size"],
                hidden_size=checkpoint["hidden_size"],
                num_layers=checkpoint["num_layers"],
                num_classes=checkpoint["num_classes"],
                dropout=checkpoint["dropout"]
            )
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()

            with open("models/lstm_scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            with open("models/lstm_label_encoder.pkl", "rb") as f:
                self.encoder = pickle.load(f)

            self._ready = True
            logger.info("LSTM model loaded")

        except FileNotFoundError:
            logger.warning("LSTM model not found — run Phase 3B first")
            self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    def predict(self, sentiment_compound: float,
                risk_score: float, keyword_risk: float) -> Optional[Dict]:
        """
        Single-article LSTM prediction.
        Replicates features across sequence length as fallback.
        """
        if not self._ready:
            return None

        try:
            seq_len = self.config.get("sequence_length", 7)
            features = self.scaler.transform(
                [[sentiment_compound, risk_score, keyword_risk]]
            )
            # Replicate across sequence length
            sequence = np.repeat(features, seq_len, axis=0)
            X = torch.FloatTensor(sequence).unsqueeze(0)

            with torch.no_grad():
                output = self.model(X)
                proba = torch.softmax(output, dim=1).numpy()[0]
                pred_idx = np.argmax(proba)
                label = self.encoder.inverse_transform([pred_idx])[0]
                confidence = round(float(np.max(proba)), 4)

            return {
                "label": label,
                "confidence": confidence,
                "probabilities": {
                    cls: round(float(p), 4)
                    for cls, p in zip(self.encoder.classes_, proba)
                }
            }

        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return None
