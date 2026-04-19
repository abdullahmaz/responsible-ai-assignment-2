"""Three-layer moderation pipeline.

Layer 1 — regex blocklist (cheap, auditable).
Layer 2 — calibrated DistilBERT (isotonic calibration of the best mitigated model from Part 4).
Layer 3 — human review queue for the uncertainty band.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BLOCKLIST: dict[str, list[re.Pattern[str]]] = {
    "direct_threat": [
        # capturing group over the threat verb — required by the spec
        re.compile(r"\bi\s*(?:'?ll|will|am\s+going\s+to|gonna)\s+(kill|murder|shoot|stab|hurt|beat|strangle)\s+you\b", re.IGNORECASE),
        re.compile(r"\byou\s*(?:'?re|are)\s+(?:going\s+to|gonna)\s+(?:die|get\s+hurt|bleed)\b", re.IGNORECASE),
        re.compile(r"\bsomeone\s+should\s+(?:kill|shoot|stab|hurt)\s+(?:you|him|her|them)\b", re.IGNORECASE),
        re.compile(r"\bi\s*(?:'?ll|will)\s+find\s+(?:where|out\s+where)\s+you\s+live\b", re.IGNORECASE),
        re.compile(r"\b(?:i\s+hope|hope)\s+you\s+(?:die|get\s+killed|get\s+hit)\b", re.IGNORECASE),
        re.compile(r"\bi'?m\s+coming\s+(?:for|after)\s+you\b", re.IGNORECASE),
    ],
    "self_harm_directed": [
        # second-person subject, self-harm verb phrase
        re.compile(r"\byou\s+should\s+(?:kill|off|hang|cut)\s+yourself\b", re.IGNORECASE),
        re.compile(r"\bgo\s+(?:kill|off|hang)\s+yourself\b", re.IGNORECASE),
        re.compile(r"\bnobody\s+(?:would|will)\s+miss\s+you\s+if\s+you\s+(?:died|were\s+dead)\b", re.IGNORECASE),
        re.compile(r"\bdo\s+everyone\s+a\s+(?:favou?r|solid)\s+and\s+(?:disappear|die|kys)\b", re.IGNORECASE),
        re.compile(r"\bthe\s+world\s+would\s+be\s+better\s+(?:off\s+)?without\s+you\b", re.IGNORECASE),
        re.compile(r"\bkys\b", re.IGNORECASE),
    ],
    "doxxing_stalking": [
        re.compile(r"\bi\s+know\s+where\s+you\s+(?:live|work|go\s+to\s+school)\b", re.IGNORECASE),
        re.compile(r"\bi\s*(?:'?ll|will|am\s+going\s+to|gonna)\s+post\s+your\s+(?:address|phone|number|email|workplace)\b", re.IGNORECASE),
        re.compile(r"\bi\s+found\s+your\s+(?:real\s+name|address|workplace|employer|family)\b", re.IGNORECASE),
        re.compile(r"\beveryone\s+will\s+know\s+who\s+you\s+really\s+are\b", re.IGNORECASE),
        re.compile(r"\bi\s*(?:'?ve|have)\s+got\s+your\s+(?:ip|home|school)\b", re.IGNORECASE),
    ],
    "dehumanization": [
        # must use the required (?:human|people|person) alternation
        re.compile(r"\b\w+\s+are\s+not\s+(?:human|people|person)s?\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+are\s+(?:animals|vermin|rats|cockroaches|parasites)\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+should\s+be\s+(?:exterminated|eradicated|wiped\s+out|gassed)\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+are\s+a\s+(?:disease|cancer|plague|infestation)\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+don'?t\s+deserve\s+to\s+(?:exist|live|be\s+(?:human|people|person)s?)\b", re.IGNORECASE),
    ],
    "coordinated_harassment": [
        # at least one lookahead
        re.compile(r"\beveryone\s+report\s+(?=\S)", re.IGNORECASE),
        re.compile(r"\blet'?s\s+all\s+(?:go\s+after|dogpile|pile\s+on|bury)\s+\S+", re.IGNORECASE),
        re.compile(r"\braid\s+(?:their|his|her|that)\s+(?:profile|account|dm|inbox)\b", re.IGNORECASE),
        re.compile(r"\bmass\s+report\s+this\s+(?:account|user|profile)\b", re.IGNORECASE),
    ],
}


def input_filter(text: str) -> dict | None:
    """Run the regex blocklist. Return a block decision dict if matched, else None."""
    for category, patterns in BLOCKLIST.items():
        for pattern in patterns:
            if pattern.search(text):
                return {
                    "decision": "block",
                    "layer": "input_filter",
                    "category": category,
                    "confidence": 1.0,
                }
    return None


class _DistilBertScorer(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper around a HuggingFace DistilBERT sequence classifier.

    Only needed so that `CalibratedClassifierCV` can wrap the model via `predict_proba`.
    """

    def __init__(self, model_dir: str, device: str | None = None, max_length: int = 128, batch_size: int = 64):
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self.classes_ = np.array([0, 1])
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(self.device).eval()

    def fit(self, X, y=None):
        self._load()
        return self

    @torch.no_grad()
    def predict_proba(self, X: Iterable[str]) -> np.ndarray:
        self._load()
        texts = list(X)
        out = np.zeros((len(texts), 2), dtype=np.float32)
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            out[i : i + len(batch)] = probs
        return out

    def predict(self, X: Iterable[str]) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


@dataclass
class _CalibrationData:
    texts: list[str]
    labels: np.ndarray


class ModerationPipeline:
    """Three-layer content moderation pipeline.

    Usage:
        pipe = ModerationPipeline.from_artifacts('artifacts/best_mitigated.json', calibration_data)
        pipe.predict("some comment")
    """

    def __init__(
        self,
        model_dir: str,
        calibrator: CalibratedClassifierCV,
        block_threshold: float = 0.6,
        allow_threshold: float = 0.4,
    ):
        if not (0.0 <= allow_threshold < block_threshold <= 1.0):
            raise ValueError("Thresholds must satisfy 0 <= allow < block <= 1.")
        self.model_dir = model_dir
        self.calibrator = calibrator
        self.block_threshold = block_threshold
        self.allow_threshold = allow_threshold

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_artifacts(
        cls,
        best_mitigated_json: str | Path,
        calibration_texts: Iterable[str],
        calibration_labels: Iterable[int],
        block_threshold: float = 0.6,
        allow_threshold: float = 0.4,
    ) -> "ModerationPipeline":
        meta = json.loads(Path(best_mitigated_json).read_text())
        model_dir = meta["model_dir"]
        scorer = _DistilBertScorer(model_dir=model_dir)
        scorer.fit([])  # eager-load weights
        calibrator = CalibratedClassifierCV(estimator=scorer, method="isotonic", cv="prefit")
        calibrator.fit(list(calibration_texts), np.asarray(list(calibration_labels)))
        return cls(
            model_dir=model_dir,
            calibrator=calibrator,
            block_threshold=block_threshold,
            allow_threshold=allow_threshold,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _model_probs(self, texts: list[str]) -> np.ndarray:
        return self.calibrator.predict_proba(texts)[:, 1]

    def predict(self, text: str) -> dict:
        pre = input_filter(text)
        if pre is not None:
            return pre
        conf = float(self._model_probs([text])[0])
        if conf >= self.block_threshold:
            return {"decision": "block", "layer": "model", "confidence": conf}
        if conf <= self.allow_threshold:
            return {"decision": "allow", "layer": "model", "confidence": conf}
        return {"decision": "review", "layer": "model", "confidence": conf}

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Batched path that avoids a per-text model call.

        Useful for the Part 5 demonstration over 1,000 comments.
        """
        results: list[dict | None] = [input_filter(t) for t in texts]
        need_model = [i for i, r in enumerate(results) if r is None]
        if need_model:
            probs = self._model_probs([texts[i] for i in need_model])
            for idx, conf in zip(need_model, probs):
                conf_f = float(conf)
                if conf_f >= self.block_threshold:
                    results[idx] = {"decision": "block", "layer": "model", "confidence": conf_f}
                elif conf_f <= self.allow_threshold:
                    results[idx] = {"decision": "allow", "layer": "model", "confidence": conf_f}
                else:
                    results[idx] = {"decision": "review", "layer": "model", "confidence": conf_f}
        return results  # type: ignore[return-value]


__all__ = ["BLOCKLIST", "input_filter", "ModerationPipeline"]


if __name__ == "__main__":
    # Smoke-test the regex layer without touching the model so the file can be
    # run on any machine (no GPU / no checkpoint required).
    samples = [
        "I will kill you tomorrow",
        "you should kill yourself",
        "I know where you live",
        "those people are not human",
        "everyone report user123 for spam",
        "thanks for the thoughtful comment",
    ]
    for s in samples:
        hit = input_filter(s)
        if hit is None:
            print(f"[pass-through] {s!r}")
        else:
            print(f"[{hit['category']:<22}] {s!r}")
