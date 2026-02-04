from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CommTotals:
    bytes_up: int = 0
    bytes_down: int = 0

    @property
    def bytes_total(self) -> int:
        return int(self.bytes_up + self.bytes_down)


def comm_record(
    *,
    round_idx: Optional[int],
    stage: str,
    bytes_up: int,
    bytes_down: int,
    cumulative: CommTotals,
    add_to_cumulative: bool = True,
) -> Dict[str, Any]:
    """
    Record a communication event.
    
    Args:
        round_idx: Round number
        stage: Event stage (e.g., "client_comm", "round_total")
        bytes_up: Bytes uploaded in this event
        bytes_down: Bytes downloaded in this event
        cumulative: Running totals tracker
        add_to_cumulative: If True, adds bytes to cumulative totals.
                          Set to False for summary/total lines that are just
                          aggregations of already-counted events.
    """
    if add_to_cumulative:
        cumulative.bytes_up += int(bytes_up)
        cumulative.bytes_down += int(bytes_down)
    return {
        "round": round_idx,
        "stage": stage,
        "bytes_up": int(bytes_up),
        "bytes_down": int(bytes_down),
        "bytes_total": int(bytes_up + bytes_down),
        "cumulative_bytes_up": int(cumulative.bytes_up),
        "cumulative_bytes_down": int(cumulative.bytes_down),
        "cumulative_bytes_total": int(cumulative.bytes_total),
    }


def bytes_for_embeddings_upload(num_clients: int, num_samples: int, emb_dim: int, bytes_per_elem: int = 2) -> int:
    # float16 => 2 bytes
    return int(num_clients * num_samples * emb_dim * bytes_per_elem)


def bytes_for_labels_download(num_clients: int, labels: list[str]) -> int:
    # per spec: num_clients * sum(len(label_utf8)) for the round
    per_round = sum(len(l.encode("utf-8")) for l in labels)
    return int(num_clients * per_round)


