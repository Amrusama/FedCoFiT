from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer

from fedcofit.prompts import normalize_label
from fedcofit.timing import timing_record


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # assumes normalized
    return float(np.dot(a, b))


def _medoid_by_avg_cosine(embs_norm: np.ndarray) -> int:
    # embs_norm shape [n, d], already normalized.
    sims = embs_norm @ embs_norm.T
    avg = sims.mean(axis=1)
    return int(np.argmax(avg))


def semantic_consensus_one_sample(
    *,
    embeddings: List[Optional[np.ndarray]],
    labels: List[Optional[str]],
    dbscan_eps: float,
    dbscan_min_samples: int,
    dbscan_metric: str,
    task_type: str = "classification",
    sample_idx: int = -1,
    verbose: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Compute semantic consensus for one sample across all clients.
    
    For classification tasks:
        labels are "positive" or "negative", returns the consensus label
    For instruction tasks:
        labels are full response texts, returns the response closest to cluster centroid
    
    embeddings: list length num_clients with float16/float32 arrays shape [d] or None for invalid
    labels: list length num_clients with label/response string or None for invalid
    
    Returns:
        (consensus_label, debug_info) where debug_info contains clustering details
    """
    debug_info: Dict[str, Any] = {
        "sample_idx": sample_idx,
        "num_clients": len(embeddings),
        "valid_clients": [],
        "cluster_labels": [],
        "num_clusters": 0,
        "majority_cluster_size": 0,
        "selection_method": "none",
        "selected_client": -1,
        "selected_response_preview": "",
    }
    
    valid_idx = [i for i, e in enumerate(embeddings) if e is not None and labels[i] is not None and labels[i] != ""]
    debug_info["valid_clients"] = valid_idx
    
    if len(valid_idx) == 0:
        debug_info["selection_method"] = "fallback_no_valid"
        result = "negative" if task_type == "classification" else ""
        debug_info["selected_response_preview"] = result[:100] if result else "(empty)"
        return result, debug_info

    embs = np.stack([embeddings[i].astype(np.float32) for i in valid_idx], axis=0)
    embs = _l2_normalize(embs)

    # DBSCAN over clients for this sample
    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric=dbscan_metric)
    cl = clustering.fit_predict(embs)
    debug_info["cluster_labels"] = cl.tolist()

    # majority cluster among labels != -1
    non_outlier = cl[cl != -1]
    unique_clusters = np.unique(non_outlier) if non_outlier.size > 0 else np.array([])
    debug_info["num_clusters"] = len(unique_clusters)
    
    if non_outlier.size > 0:
        # pick largest cluster id
        uniq, cnt = np.unique(non_outlier, return_counts=True)
        maj_cluster = int(uniq[int(np.argmax(cnt))])
        members = np.where(cl == maj_cluster)[0]
        debug_info["majority_cluster_size"] = len(members)
        debug_info["majority_cluster_members"] = [valid_idx[m] for m in members]
        
        centroid = _l2_normalize(embs[members].mean(axis=0, keepdims=True))[0]
        # choose prediction closest to centroid
        sims = (embs[members] @ centroid)
        best_local = int(members[int(np.argmax(sims))])
        best_global = valid_idx[best_local]
        
        debug_info["selection_method"] = "dbscan_centroid"
        debug_info["selected_client"] = best_global
        result = str(labels[best_global])
        debug_info["selected_response_preview"] = result[:100] if result else "(empty)"
        
        if verbose:
            print(f"  [Sample {sample_idx}] DBSCAN: {len(unique_clusters)} clusters, "
                  f"majority has {len(members)} clients {[valid_idx[m] for m in members]}, "
                  f"selected client {best_global}")
        
        return result, debug_info

    # fallback A: medoid by highest average cosine similarity
    debug_info["selection_method"] = "medoid_fallback"
    med_local = _medoid_by_avg_cosine(embs)
    med_global = valid_idx[med_local]
    med_label = labels[med_global]
    debug_info["selected_client"] = med_global
    
    if verbose:
        print(f"  [Sample {sample_idx}] DBSCAN: no clusters (all outliers), "
              f"using medoid from client {med_global}")
    
    if task_type == "instruction":
        result = str(med_label) if med_label else ""
        debug_info["selected_response_preview"] = result[:100] if result else "(empty)"
        return result, debug_info
    
    # For classification: check if it's a valid label
    if med_label in {"positive", "negative"}:
        result = str(med_label)
        debug_info["selected_response_preview"] = result
        return result, debug_info

    # fallback B: majority vote over normalized label strings (classification only)
    debug_info["selection_method"] = "majority_vote_fallback"
    votes: Dict[str, int] = {"positive": 0, "negative": 0}
    for i in valid_idx:
        if labels[i] in votes:
            votes[str(labels[i])] += 1
    if votes["positive"] == votes["negative"]:
        result = "negative"
    else:
        result = "positive" if votes["positive"] > votes["negative"] else "negative"
    debug_info["selected_response_preview"] = result
    return result, debug_info


def semantic_consensus_batch(
    *,
    client_embeddings: Dict[int, np.ndarray],
    client_labels: Dict[int, List[Optional[str]]],
    num_clients: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    dbscan_metric: str,
    task_type: str = "classification",
    verbose: bool = False,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Compute semantic consensus for a batch of samples.
    
    For classification: returns list of "positive"/"negative"
    For instruction: returns list of full response texts (consensus responses)
    
    Returns:
        (consensus_labels, batch_debug_info)
    """
    # client_embeddings[client_id] shape [n_samples, emb_dim] float16
    any_client = next(iter(client_embeddings.keys()))
    n_samples = int(client_embeddings[any_client].shape[0])
    consensus: List[str] = []
    all_debug: List[Dict[str, Any]] = []
    
    # Summary stats
    selection_method_counts: Dict[str, int] = {}
    cluster_counts: List[int] = []
    
    if verbose:
        print(f"\n[Server] Processing {n_samples} samples from {num_clients} clients...")
    
    for s in range(n_samples):
        embs: List[Optional[np.ndarray]] = [None] * num_clients
        labs: List[Optional[str]] = [None] * num_clients
        for cid in range(num_clients):
            if cid in client_embeddings:
                vec = client_embeddings[cid][s]
                embs[cid] = vec
                labs[cid] = client_labels[cid][s] if cid in client_labels else None
        
        result, debug_info = semantic_consensus_one_sample(
            embeddings=embs,
            labels=labs,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            dbscan_metric=dbscan_metric,
            task_type=task_type,
            sample_idx=s,
            verbose=verbose,
        )
        consensus.append(result)
        all_debug.append(debug_info)
        
        # Track stats
        method = debug_info["selection_method"]
        selection_method_counts[method] = selection_method_counts.get(method, 0) + 1
        cluster_counts.append(debug_info["num_clusters"])
    
    batch_debug = {
        "num_samples": n_samples,
        "num_clients": num_clients,
        "selection_method_counts": selection_method_counts,
        "avg_clusters_per_sample": sum(cluster_counts) / len(cluster_counts) if cluster_counts else 0,
        "samples_with_clusters": sum(1 for c in cluster_counts if c > 0),
        "samples_all_outliers": sum(1 for c in cluster_counts if c == 0),
        "per_sample_debug": all_debug,
    }
    
    if verbose:
        print(f"\n[Server] Consensus Summary:")
        print(f"  Total samples: {n_samples}")
        print(f"  Selection methods: {selection_method_counts}")
        print(f"  Samples with DBSCAN clusters: {batch_debug['samples_with_clusters']}")
        print(f"  Samples all outliers (medoid fallback): {batch_debug['samples_all_outliers']}")
        print(f"  Avg clusters per sample: {batch_debug['avg_clusters_per_sample']:.2f}")
    
    return consensus, batch_debug


def server_loop(
    *,
    cfg: Dict[str, Any],
    num_clients: int,
    rounds: int,
    in_queue,
    out_queues: Dict[int, Any],
    event_queue,
) -> None:
    db = cfg["server"]["dbscan"]
    eps = float(db["eps"])
    min_samples = int(db["min_samples"])
    metric = str(db["metric"])
    send_mode = str(cfg["server"].get("send_mode", "embeddings"))
    task_type = str(cfg.get("task", {}).get("type", "classification"))

    # If server is responsible for embeddings (send_mode=text), initialize embedding model
    sbert_model: Optional[SentenceTransformer] = None
    emb_pos: Optional[np.ndarray] = None
    emb_neg: Optional[np.ndarray] = None
    emb_dim: Optional[int] = None
    if send_mode == "text":
        t_init = time.perf_counter()
        st_name = cfg["server"]["embedding_model_name"]
        st_dev = cfg["server"].get("embedding_device", "cpu")
        sbert_model = SentenceTransformer(st_name, device=str(st_dev))
        emb_dim = sbert_model.get_sentence_embedding_dimension()
        
        if task_type == "classification":
            # Pre-compute embeddings for classification labels
            embs = sbert_model.encode(["positive", "negative"], convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            emb_pos = embs[0].astype(np.float16)
            emb_neg = embs[1].astype(np.float16)
        # For instruction tasks, we'll embed dynamically per round
        
        event_queue.put(
            {"type": "timing", "data": timing_record("server_embedding_model_init", time.perf_counter() - t_init, None, None)}
        )

    for r in range(rounds):
        # gather uploads
        client_embeddings: Dict[int, np.ndarray] = {}
        client_labels: Dict[int, List[Optional[str]]] = {}
        got = 0
        while got < num_clients:
            msg = in_queue.get()
            if msg.get("type") != "upload" or int(msg["round"]) != r:
                continue
            cid = int(msg["client_id"])
            client_labels[cid] = msg["labels"]
            if send_mode == "embeddings":
                client_embeddings[cid] = msg["embeddings"]
            else:
                # send_mode=text: compute embeddings on server from client's text
                t_emb = time.perf_counter()
                labs: List[Optional[str]] = msg["labels"]
                if emb_dim is None:
                    emb_dim = 384
                
                if task_type == "instruction":
                    # Instruction tasks: embed full response text using SBERT
                    valid_texts = [lab if lab else "" for lab in labs]
                    arr = sbert_model.encode(valid_texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
                else:
                    # Classification: lookup pre-computed label embeddings
                    arr = np.zeros((len(labs), int(emb_dim)), dtype=np.float16)
                    for i, lab in enumerate(labs):
                        if lab == "positive" and emb_pos is not None:
                            arr[i] = emb_pos
                        elif lab == "negative" and emb_neg is not None:
                            arr[i] = emb_neg
                
                client_embeddings[cid] = arr
                event_queue.put(
                    {"type": "timing", "data": timing_record("server_embedding_computation", time.perf_counter() - t_emb, cid, r)}
                )
            got += 1

        t0 = time.perf_counter()
        verbose = cfg.get("server", {}).get("verbose_consensus", False)
        consensus, consensus_debug = semantic_consensus_batch(
            client_embeddings=client_embeddings,
            client_labels=client_labels,
            num_clients=num_clients,
            dbscan_eps=eps,
            dbscan_min_samples=min_samples,
            dbscan_metric=metric,
            task_type=task_type,
            verbose=verbose,
        )
        event_queue.put({"type": "timing", "data": timing_record("server_aggregation", time.perf_counter() - t0, None, r)})
        
        # Log consensus debug info (without per-sample details which could be large)
        consensus_summary = {
            "round": r,
            "num_samples": consensus_debug["num_samples"],
            "selection_method_counts": consensus_debug["selection_method_counts"],
            "samples_with_clusters": consensus_debug["samples_with_clusters"],
            "samples_all_outliers": consensus_debug["samples_all_outliers"],
            "avg_clusters_per_sample": consensus_debug["avg_clusters_per_sample"],
            "task_type": task_type,
        }
        event_queue.put({"type": "consensus_labels", "data": {"round": r, "labels": consensus, "task_type": task_type}})
        event_queue.put({"type": "consensus_debug", "data": consensus_summary})

        # broadcast packaging + send
        t1 = time.perf_counter()
        for cid in range(num_clients):
            out_queues[cid].put({"type": "consensus", "round": r, "labels": consensus})
        event_queue.put({"type": "timing", "data": timing_record("broadcast_packaging", time.perf_counter() - t1, None, r)})

    # shutdown
    for cid in range(num_clients):
        out_queues[cid].put({"type": "done"})


