from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class StageTimer:
    stage: str
    client_id: Optional[int]
    round_idx: Optional[int]
    t0: float = 0.0

    def __enter__(self) -> "StageTimer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        pass

    def seconds(self) -> float:
        return time.perf_counter() - self.t0


def timing_record(stage: str, seconds: float, client_id: Optional[int], round_idx: Optional[int]) -> Dict[str, Any]:
    return {
        "stage": stage,
        "client_id": client_id,
        "round": round_idx,
        "seconds": float(seconds),
    }


