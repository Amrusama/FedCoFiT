from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AccStats:
    correct: int = 0
    total: int = 0

    def update(self, pred: Optional[str], gold: str) -> None:
        if pred is None:
            self.total += 1
            return
        self.correct += int(pred == gold)
        self.total += 1

    def accuracy(self) -> float:
        return float(self.correct) / float(max(1, self.total))


def summarize_client_accuracies(accs: List[float]) -> Dict[str, Any]:
    import numpy as np

    a = np.array(accs, dtype=np.float64)
    return {
        "mean": float(a.mean()) if a.size else 0.0,
        "std": float(a.std(ddof=0)) if a.size else 0.0,
        "min": float(a.min()) if a.size else 0.0,
        "max": float(a.max()) if a.size else 0.0,
    }


