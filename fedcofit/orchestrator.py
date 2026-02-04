from __future__ import annotations

import json
import os
import queue
import time
from dataclasses import asdict
from multiprocessing import Process, Queue
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fedcofit.client import client_loop
from fedcofit.comm_cost import CommTotals, bytes_for_embeddings_upload, bytes_for_labels_download, comm_record
from fedcofit.data import FederatedSplits, prepare_federated_splits
from fedcofit.metrics import summarize_client_accuracies
from fedcofit.mmlu_eval import run_mmlu_evaluation, write_mmlu_results_to_summary
from fedcofit.final_eval import write_final_eval_to_summary
from fedcofit.server import server_loop
from fedcofit.timing import timing_record
from fedcofit.utils import ensure_dir, get_env_info, jsonl_append, now_iso, write_text

# Optional: Energy tracking via Lamarr Energy Tracker (LET)
try:
    from lamarr_energy_tracker import EnergyTracker
    HAS_ENERGY_TRACKER = True
except ImportError:
    HAS_ENERGY_TRACKER = False


def _init_results_dir(cfg: Dict[str, Any]) -> str:
    """
    Create a unique results directory for each experiment run.
    Format: results/<method>_<model>_<dataset>_<num_clients>clients_<timestamp>/
    Example: results/FedCoFiT_TinyLlama_1_1B_glue_sst2_10clients_20251213_120000/
    """
    root = cfg["experiment"]["output_dir"]
    
    # Extract components for folder name
    method_name = cfg["experiment"].get("method", "FedCoFiT")  # Default to FedCoFiT
    
    model_name = cfg["model"]["base_model_name_or_path"].split("/")[-1]  # e.g., "TinyLlama-1.1B-Chat-v1.0"
    model_short = model_name.replace("-", "_").replace(".", "_")[:25]  # Shorten and sanitize
    
    task_name = cfg["task"]["name"].replace("/", "_")  # e.g., "glue_sst2"
    
    num_clients = cfg["hardware"]["num_clients"]
    
    # Timestamp for uniqueness
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Build folder name: method_model_dataset_clients_timestamp
    folder_name = f"{method_name}_{model_short}_{task_name}_{num_clients}clients_{timestamp}"
    
    out_dir = os.path.join(root, folder_name)
    ensure_dir(out_dir)
    
    # Initialize empty log files
    for fname in ["metrics.jsonl", "timing.jsonl", "comm.jsonl"]:
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            f.write("")
    
    return out_dir


def _write_resolved_config(cfg: Dict[str, Any], out_dir: str) -> None:
    import yaml

    with open(os.path.join(out_dir, "resolved_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _write_summary(
    *,
    out_dir: str,
    cfg: Dict[str, Any],
    metrics_by_round: Dict[int, Dict[int, float]],
    timings: List[Dict[str, Any]],
    comm_events: List[Dict[str, Any]],
    env_info: Dict[str, Any],
    total_seconds: float,
    energy_info: Optional[Dict[str, Any]] = None,
    mmlu_results: Optional[Dict[str, Any]] = None,
    final_eval_results: Optional[Dict[int, Dict[str, Any]]] = None,
) -> None:
    rounds = int(cfg["federated"]["rounds"])
    num_clients = int(cfg["hardware"]["num_clients"])

    # Header with method name
    method_name = cfg["experiment"].get("method", "FedCoFiT")
    lines: List[str] = []
    lines.append(f"## {method_name} Summary: `{cfg['experiment']['name']}`")
    lines.append("")
    lines.append(f"- **Method**: {method_name}")
    lines.append(f"- **Timestamp**: {now_iso()}")
    lines.append(f"- **Total runtime (s)**: {total_seconds:.2f}")
    lines.append(f"- **Clients**: {num_clients}")
    lines.append(f"- **Rounds**: {rounds} (round 0 = post-expert)")
    lines.append("")

    # Determine task type for appropriate metrics display
    task_type = cfg.get("task", {}).get("type", "classification")
    task_name = cfg.get("task", {}).get("name", "unknown")
    
    if task_type == "instruction":
        lines.append(f"### Perplexity vs rounds ({task_name})")
        lines.append("")
        lines.append("*Note: For instruction-following tasks, lower perplexity = better model.*")
        lines.append("")
    else:
        lines.append(f"### Accuracy vs rounds ({task_name})")
        lines.append("")
    
    lines.append("| Round | Mean | Std | Min | Max |")
    lines.append("|---:|---:|---:|---:|---:|")
    for r in range(0, rounds + 1):
        accs = [metrics_by_round.get(r, {}).get(cid, float("nan")) for cid in range(num_clients)]
        accs_clean = [a for a in accs if np.isfinite(a)]
        summ = summarize_client_accuracies(accs_clean)
        lines.append(f"| {r} | {summ['mean']:.4f} | {summ['std']:.4f} | {summ['min']:.4f} | {summ['max']:.4f} |")
    lines.append("")

    # Timing breakdown
    lines.append("### Wall-clock time breakdown (sum over events)")
    lines.append("")
    by_stage: Dict[str, float] = {}
    for t in timings:
        by_stage[str(t["stage"])] = by_stage.get(str(t["stage"]), 0.0) + float(t["seconds"])
    lines.append("| Stage | Seconds |")
    lines.append("|---|---:|")
    for k in sorted(by_stage.keys()):
        lines.append(f"| {k} | {by_stage[k]:.2f} |")
    lines.append("")

    # Communication
    lines.append("### Communication cost")
    lines.append("")
    if comm_events:
        last = comm_events[-1]
        lines.append(f"- **Total bytes up**: {last['cumulative_bytes_up']}")
        lines.append(f"- **Total bytes down**: {last['cumulative_bytes_down']}")
        lines.append(f"- **Total bytes**: {last['cumulative_bytes_total']}")
        per_round = [e for e in comm_events if e.get("round") is not None]
        if per_round:
            avg = float(np.mean([e["bytes_total"] for e in per_round]))
            lines.append(f"- **Avg bytes/round**: {avg:.2f}")
    else:
        lines.append("- No comm events recorded.")
    lines.append("")

    # Energy consumption (if tracked)
    if energy_info and energy_info.get("energy_kwh", 0.0) > 0:
        lines.append("### Energy consumption (via Lamarr Energy Tracker)")
        lines.append("")
        lines.append(f"- **Total energy consumed**: {energy_info.get('energy_kwh', 0.0):.6f} kWh")
        lines.append(f"- **CO2 emissions**: {energy_info.get('co2_kg', 0.0):.6f} kg CO2-eq")
        lines.append(f"- **GPU energy**: {energy_info.get('gpu_energy_kwh', 0.0):.6f} kWh")
        lines.append(f"- **CPU energy**: {energy_info.get('cpu_energy_kwh', 0.0):.6f} kWh")
        lines.append(f"- **RAM energy**: {energy_info.get('ram_energy_kwh', 0.0):.6f} kWh")
        if energy_info.get("gpu_power_watts"):
            lines.append(f"- **Avg GPU power**: {energy_info.get('gpu_power_watts', 0.0):.1f} W")
        if energy_info.get("gpu_model"):
            lines.append(f"- **Hardware**: {energy_info.get('cpu_model', 'N/A')} + {energy_info.get('gpu_model', 'N/A')}")
        lines.append("")
        lines.append("> Tracked using [Lamarr Energy Tracker](https://github.com/lamarr-institute/lamarr-energy-tracker)")
        lines.append("")

    # MMLU Benchmark Results (for classification tasks)
    if mmlu_results:
        write_mmlu_results_to_summary(mmlu_results, lines)
    
    # Final Evaluation Results (for instruction-following tasks)
    if final_eval_results:
        write_final_eval_to_summary(final_eval_results, lines, is_multi_client=True)

    # Environment
    lines.append("### Environment")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(env_info, indent=2))
    lines.append("```")
    lines.append("")

    write_text(os.path.join(out_dir, "summary.md"), "\n".join(lines))


def run_experiment(cfg: Dict[str, Any]) -> str:
    out_dir = _init_results_dir(cfg)
    
    # Store output directory in config so clients can save evaluation files
    cfg["_internal_output_dir"] = out_dir
    
    if bool(cfg.get("logging", {}).get("write_resolved_config", True)):
        _write_resolved_config(cfg, out_dir)

    env_info = get_env_info()

    # data splits (indices only)
    splits: FederatedSplits = prepare_federated_splits(cfg)
    splits_dict = asdict(splits)

    num_clients = int(cfg["hardware"]["num_clients"])
    rounds = int(cfg["federated"]["rounds"])

    # Energy tracking (optional)
    energy_tracker = None
    track_energy = bool(cfg.get("logging", {}).get("track_energy", True)) and HAS_ENERGY_TRACKER
    if track_energy:
        project_name = cfg["experiment"]["name"]
        energy_tracker = EnergyTracker(project_name=project_name, output_dir=out_dir)
        energy_tracker.start()

    # IPC
    server_in: Queue = Queue()
    client_out: Dict[int, Queue] = {i: Queue() for i in range(num_clients)}
    events: Queue = Queue()

    # server
    server_p = Process(
        target=server_loop,
        kwargs=dict(
            cfg=cfg,
            num_clients=num_clients,
            rounds=rounds,
            in_queue=server_in,
            out_queues=client_out,
            event_queue=events,
        ),
        daemon=True,
    )
    server_p.start()

    # clients
    procs: List[Process] = []
    for cid in range(num_clients):
        gpu_id = int(cid // int(cfg["hardware"].get("clients_per_gpu", 2)))
        p = Process(
            target=client_loop,
            kwargs=dict(
                cfg=cfg,
                client_id=cid,
                gpu_id=gpu_id,
                splits=splits_dict,
                server_in_queue=server_in,
                client_out_queue=client_out[cid],
                event_queue=events,
            ),
            daemon=True,
        )
        p.start()
        procs.append(p)

    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    timing_path = os.path.join(out_dir, "timing.jsonl")
    comm_path = os.path.join(out_dir, "comm.jsonl")

    start = time.perf_counter()
    done_clients = 0
    metrics_by_round: Dict[int, Dict[int, float]] = {}
    have_metrics_counts: Dict[int, int] = {}
    emb_dim: Optional[int] = None
    comm_totals = CommTotals()
    comm_events: List[Dict[str, Any]] = []
    timings: List[Dict[str, Any]] = []
    send_mode = str(cfg["server"].get("send_mode", "embeddings"))
    mmlu_results_by_client: Dict[int, Dict[str, Any]] = {}
    final_eval_results_by_client: Dict[int, Dict[str, Any]] = {}

    # event loop: stop when all clients report done
    while done_clients < num_clients:
        try:
            ev = events.get(timeout=5.0)
        except queue.Empty:
            # check for crashed processes
            for p in procs:
                if not p.is_alive() and p.exitcode not in (0, None):
                    raise RuntimeError(f"Client crashed: pid={p.pid} exitcode={p.exitcode}")
            continue

        et = ev.get("type")
        data = ev.get("data", {})

        if et == "timing":
            jsonl_append(timing_path, data)
            timings.append(dict(data))
        elif et == "client_info":
            # record per-client device mapping / info
            jsonl_append(metrics_path, {"type": "client_info", **data})
            if "emb_dim" in data and emb_dim is None:
                emb_dim = int(data["emb_dim"])
        elif et == "metrics":
            # per-client metrics record
            jsonl_append(metrics_path, {"type": "client", **data})
            # also treat eval time as timing event
            if "eval_seconds" in data:
                tr = timing_record("evaluation", float(data["eval_seconds"]), int(data["client_id"]), int(data["round"]))
                jsonl_append(timing_path, tr)
                timings.append(dict(tr))
            r = int(data["round"])
            cid = int(data["client_id"])
            # For instruction tasks: use perplexity; for classification: use accuracy
            if "perplexity" in data and "accuracy" not in data:
                # Instruction task - store perplexity (will be displayed as perplexity in summary)
                metrics_by_round.setdefault(r, {})[cid] = float(data["perplexity"])
            else:
                # Classification task - store accuracy
                metrics_by_round.setdefault(r, {})[cid] = float(data["accuracy"])
            have_metrics_counts[r] = len(metrics_by_round[r])
            if have_metrics_counts[r] == num_clients:
                accs = [metrics_by_round[r][i] for i in range(num_clients)]
                summ = summarize_client_accuracies(accs)
                jsonl_append(metrics_path, {"type": "summary", "round": r, "num_clients": num_clients, **summ})
        elif et == "checkpoint":
            jsonl_append(metrics_path, {"type": "checkpoint", **data})
        elif et == "mmlu_results":
            # MMLU evaluation results from a client (classification tasks)
            cid = int(data.get("client_id", 0))
            mmlu_results_by_client[cid] = dict(data)
            jsonl_append(metrics_path, {"type": "mmlu", **data})
            print(f"[FedCoFiT] Client {cid} MMLU Accuracy: {data.get('accuracy', 0.0):.4f}")
        elif et == "final_eval_results":
            # Final evaluation results from a client (instruction-following tasks)
            cid = int(data.get("client_id", 0))
            final_eval_results_by_client[cid] = dict(data)
            jsonl_append(metrics_path, {"type": "final_eval", **data})
            # Print summary of results
            if "held_out" in data:
                print(f"[FedCoFiT] Client {cid} Held-Out NLL: {data['held_out'].get('nll', 0.0):.4f}, "
                      f"Perplexity: {data['held_out'].get('perplexity', 0.0):.4f}")
            if "ifeval" in data:
                print(f"[FedCoFiT] Client {cid} IFEval Prompt-Strict: {data['ifeval'].get('prompt_strict_accuracy', 0.0):.4f}")
            if "mmlu" in data and data["mmlu"]:
                print(f"[FedCoFiT] Client {cid} MMLU Accuracy: {data['mmlu'].get('accuracy', 0.0):.4f}")
        elif et == "done":
            done_clients += 1
        elif et == "consensus_labels":
            # compute per-round comm bytes when server reports consensus labels
            r = int(data["round"])
            labels = list(map(str, data["labels"]))
            if emb_dim is None:
                # MiniLM-L6-v2 default; still, we prefer the reported dim
                emb_dim = 384
            # Calculate bytes_up based on what clients actually send
            text_bytes = bytes_for_labels_download(num_clients, labels)
            if send_mode == "embeddings":
                # Clients send embeddings + text (server needs text for consensus selection)
                emb_bytes = bytes_for_embeddings_upload(num_clients, len(labels), int(emb_dim), bytes_per_elem=2)
                bytes_up = emb_bytes + text_bytes
            else:
                # send_mode=text: clients send only text, server computes embeddings
                bytes_up = text_bytes
            
            # bytes_down: server sends consensus text to all clients
            bytes_down = text_bytes
            rec = comm_record(round_idx=r, stage="round_comm", bytes_up=bytes_up, bytes_down=bytes_down, cumulative=comm_totals)
            jsonl_append(comm_path, rec)
            comm_events.append(dict(rec))
        elif et == "consensus_debug":
            # Log consensus debug information
            r = int(data["round"])
            jsonl_append(metrics_path, {"type": "consensus_debug", **data})
            print(f"[FedCoFiT] Round {r} consensus: "
                  f"{data.get('samples_with_clusters', 0)}/{data.get('num_samples', 0)} samples clustered, "
                  f"{data.get('samples_all_outliers', 0)} medoid fallback, "
                  f"methods={data.get('selection_method_counts', {})}")
        else:
            # ignore unknown events
            pass

    total = time.perf_counter() - start
    tr_total = timing_record("total_runtime", total, None, None)
    jsonl_append(timing_path, tr_total)
    timings.append(dict(tr_total))

    # Stop energy tracking and collect results
    energy_info: Optional[Dict[str, Any]] = None
    if energy_tracker is not None:
        try:
            energy_tracker.stop()
            # Read energy data from the emissions.csv file that CodeCarbon/LET creates
            emissions_csv = os.path.join(out_dir, "emissions.csv")
            if os.path.exists(emissions_csv):
                import csv
                with open(emissions_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        last_row = rows[-1]
                        energy_info = {
                            "energy_kwh": float(last_row.get("energy_consumed", 0.0)),
                            "co2_kg": float(last_row.get("emissions", 0.0)),
                            "gpu_energy_kwh": float(last_row.get("gpu_energy", 0.0)),
                            "cpu_energy_kwh": float(last_row.get("cpu_energy", 0.0)),
                            "ram_energy_kwh": float(last_row.get("ram_energy", 0.0)),
                            "duration_seconds": float(last_row.get("duration", 0.0)),
                            "gpu_power_watts": float(last_row.get("gpu_power", 0.0)),
                            "cpu_model": str(last_row.get("cpu_model", "")),
                            "gpu_model": str(last_row.get("gpu_model", "")),
                        }
                        # Also log to metrics
                        jsonl_append(metrics_path, {"type": "energy", **energy_info})
        except Exception as e:
            print(f"[fedcofit] Warning: Energy tracking failed: {e}")

    # finalize
    for p in procs:
        p.join(timeout=5.0)
    server_p.join(timeout=5.0)

    _write_summary(
        out_dir=out_dir,
        cfg=cfg,
        metrics_by_round=metrics_by_round,
        timings=timings,
        comm_events=comm_events,
        env_info=env_info,
        total_seconds=total,
        energy_info=energy_info,
        mmlu_results=mmlu_results_by_client,
        final_eval_results=final_eval_results_by_client,
    )
    return out_dir


