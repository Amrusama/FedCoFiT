from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset


@dataclass
class FederatedSplits:
    client_train_indices: List[List[int]]
    client_val_indices: List[List[int]]
    dp_source_train_indices: List[int]
    # Task metadata
    task_type: str = "classification"  # "classification" or "instruction"
    labels: List[str] = field(default_factory=lambda: ["negative", "positive"])
    text_column: str = "sentence"
    label_column: str = "label"
    # For instruction-following tasks
    instruction_column: Optional[str] = None
    context_column: Optional[str] = None
    response_column: Optional[str] = None


# ============================================================================
# Dataset Registry - Add new datasets here
# ============================================================================

def load_sst2() -> Tuple[DatasetDict, Dict[str, Any]]:
    """Load SST-2 sentiment classification dataset."""
    dd = load_dataset("glue", "sst2")
    metadata = {
        "task_type": "classification",
        "labels": ["negative", "positive"],
        "text_column": "sentence",
        "label_column": "label",
        "train_split": "train",
        "val_split": "validation",
    }
    return dd, metadata


def load_dolly15k() -> Tuple[DatasetDict, Dict[str, Any]]:
    """Load Databricks Dolly-15k instruction-following dataset."""
    # Load the dataset
    ds = load_dataset("databricks/databricks-dolly-15k")
    
    # The dataset only has a "train" split, so we'll create train/val manually
    train_ds = ds["train"]
    
    metadata = {
        "task_type": "instruction",
        "labels": [],  # No fixed labels for instruction tasks
        "instruction_column": "instruction",
        "context_column": "context",
        "response_column": "response",
        "category_column": "category",  # For filtering by task type if needed
        "train_split": "train",
        "val_split": None,  # Will be split from train
    }
    return ds, metadata


def load_alpaca() -> Tuple[DatasetDict, Dict[str, Any]]:
    """Load Stanford Alpaca instruction-following dataset."""
    ds = load_dataset("tatsu-lab/alpaca")
    
    metadata = {
        "task_type": "instruction",
        "labels": [],
        "instruction_column": "instruction",
        "context_column": "input",
        "response_column": "output",
        "train_split": "train",
        "val_split": None,
    }
    return ds, metadata


def load_wizardlm() -> Tuple[DatasetDict, Dict[str, Any]]:
    """Load WizardLM evol-instruct 70k instruction-following dataset.
    
    The WizardLM dataset contains ~70k pairs of instructions and outputs,
    used for training the WizardLM model. It uses evolved instructions
    for more complex and diverse training data.
    
    Dataset: WizardLMTeam/WizardLM_evol_instruct_70k
    """
    ds = load_dataset("WizardLMTeam/WizardLM_evol_instruct_70k")
    
    metadata = {
        "task_type": "instruction",
        "labels": [],
        "instruction_column": "instruction",
        "context_column": None,  # WizardLM doesn't have a separate context field
        "response_column": "output",
        "train_split": "train",
        "val_split": None,
    }
    return ds, metadata


def load_openorca() -> Tuple[DatasetDict, Dict[str, Any]]:
    """Load OpenOrca instruction-following dataset.
    
    OpenOrca is a large-scale instruction dataset with ~4M examples.
    It combines multiple sources including FLAN, CoT, and more.
    We use a subset for practical federated learning experiments.
    
    Dataset: Open-Orca/OpenOrca
    Columns: id, system_prompt, question, response
    """
    # Load a subset of OpenOrca (full dataset is very large)
    ds = load_dataset("Open-Orca/OpenOrca", split="train[:100000]")  # First 100k examples
    
    # Wrap in DatasetDict format for consistency
    ds = {"train": ds}
    
    metadata = {
        "task_type": "instruction",
        "labels": [],
        "instruction_column": "question",      # OpenOrca uses "question" for instructions
        "context_column": "system_prompt",     # System prompt as context
        "response_column": "response",         # Response column
        "train_split": "train",
        "val_split": None,
    }
    return ds, metadata


def load_sharegpt() -> Tuple[DatasetDict, Dict[str, Any]]:
    """Load ShareGPT-style conversation dataset.
    
    ShareGPT contains multi-turn conversations between users and AI assistants.
    We convert multi-turn conversations to single-turn instruction-response pairs.
    
    We use Magpie-Align/Magpie-Pro-300K-Filtered which is a high-quality 
    conversation dataset similar to ShareGPT format and reliably available.
    
    Fallback options tried in order:
    1. Magpie-Align/Magpie-Pro-300K-Filtered (reliable, high quality)
    2. HuggingFaceH4/ultrachat_200k (alternative conversation data)
    """
    # Try primary source: Magpie (similar quality to ShareGPT, more reliable)
    print("[ShareGPT] Loading Magpie-Align/Magpie-Pro-300K-Filtered...")
    try:
        ds = load_dataset("Magpie-Align/Magpie-Pro-300K-Filtered", split="train[:52000]")
        print(f"[ShareGPT] Loaded Magpie dataset with {len(ds)} examples")
        
        # Magpie has 'instruction' and 'response' columns directly
        if "instruction" in ds.column_names and "response" in ds.column_names:
            # Already in correct format
            ds = ds.select_columns(["instruction", "response"])
        else:
            # Process conversations - Magpie uses "from"/"value" format
            def extract_magpie(example):
                conversations = example.get("conversations", [])
                instruction = ""
                response = ""
                for turn in conversations:
                    # Magpie uses "from" (human/gpt) and "value" instead of "role"/"content"
                    role = turn.get("from", "") or turn.get("role", "")
                    content = turn.get("value", "") or turn.get("content", "")
                    if role in ("human", "user") and not instruction:
                        instruction = content
                    elif role in ("gpt", "assistant") and instruction and not response:
                        response = content
                        break
                return {"instruction": instruction, "response": response}
            
            ds = ds.map(extract_magpie, remove_columns=ds.column_names)
        
        ds = ds.filter(lambda x: len(str(x.get("instruction", ""))) > 0 and len(str(x.get("response", ""))) > 0)
        ds = {"train": ds}
        
        metadata = {
            "task_type": "instruction",
            "labels": [],
            "instruction_column": "instruction",
            "context_column": None,
            "response_column": "response",
            "train_split": "train",
            "val_split": None,
        }
        return ds, metadata
        
    except Exception as e:
        print(f"[ShareGPT] Magpie failed: {e}")
    
    # Fallback: HuggingFaceH4/ultrachat_200k
    print("[ShareGPT] Falling back to HuggingFaceH4/ultrachat_200k...")
    try:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:52000]")
        print(f"[ShareGPT] Loaded UltraChat with {len(ds)} examples")
        
        def extract_ultrachat(example):
            messages = example.get("messages", [])
            instruction = ""
            response = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user" and not instruction:
                    instruction = content
                elif role == "assistant" and instruction and not response:
                    response = content
                    break
            return {"instruction": instruction, "response": response}
        
        ds = ds.map(extract_ultrachat, remove_columns=ds.column_names)
        ds = ds.filter(lambda x: len(str(x.get("instruction", ""))) > 0 and len(str(x.get("response", ""))) > 0)
        ds = {"train": ds}
        
        metadata = {
            "task_type": "instruction",
            "labels": [],
            "instruction_column": "instruction",
            "context_column": None,
            "response_column": "response",
            "train_split": "train",
            "val_split": None,
        }
        return ds, metadata
        
    except Exception as e:
        print(f"[ShareGPT] UltraChat failed: {e}")
        raise RuntimeError(
            "Failed to load ShareGPT-style dataset. Please check your internet connection "
            "or try installing: pip install datasets huggingface_hub"
        )


def load_ultrachat() -> Tuple[DatasetDict, Dict[str, Any]]:
    """Load UltraChat dataset directly.
    
    UltraChat is a large-scale multi-turn conversation dataset with ~200k examples.
    We use the SFT (supervised fine-tuning) subset which has high-quality conversations.
    """
    print("[UltraChat] Loading HuggingFaceH4/ultrachat_200k...")
    try:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:52000]")
        print(f"[UltraChat] Loaded {len(ds)} examples")
        
        def extract_ultrachat(example):
            messages = example.get("messages", [])
            instruction = ""
            response = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user" and not instruction:
                    instruction = content
                elif role == "assistant" and instruction and not response:
                    response = content
                    break
            return {"instruction": instruction, "response": response}
        
        ds = ds.map(extract_ultrachat, remove_columns=ds.column_names)
        ds = ds.filter(lambda x: len(str(x.get("instruction", ""))) > 0 and len(str(x.get("response", ""))) > 0)
        ds = {"train": ds}
        
        metadata = {
            "task_type": "instruction",
            "labels": [],
            "instruction_column": "instruction",
            "context_column": None,
            "response_column": "response",
            "train_split": "train",
            "val_split": None,
        }
        return ds, metadata
        
    except Exception as e:
        print(f"[UltraChat] Failed to load: {e}")
        raise RuntimeError(
            "Failed to load UltraChat dataset. Please check your internet connection "
            "or try installing: pip install datasets huggingface_hub"
        )


# Dataset registry maps task name to loader function
DATASET_REGISTRY = {
    "glue/sst2": load_sst2,
    "sst2": load_sst2,
    "databricks/dolly-15k": load_dolly15k,
    "dolly-15k": load_dolly15k,
    "dolly15k": load_dolly15k,
    "tatsu-lab/alpaca": load_alpaca,
    "alpaca": load_alpaca,
    "WizardLMTeam/WizardLM_evol_instruct_70k": load_wizardlm,
    "wizardlm": load_wizardlm,
    "wizard": load_wizardlm,
    "Open-Orca/OpenOrca": load_openorca,
    "openorca": load_openorca,
    "orca": load_openorca,
    "anon8231489123/ShareGPT_Vicuna_unfiltered": load_sharegpt,
    "sharegpt": load_sharegpt,
    "share_gpt": load_sharegpt,
    "vicuna": load_sharegpt,  # ShareGPT was used to train Vicuna
    "HuggingFaceH4/ultrachat_200k": load_ultrachat,
    "ultrachat": load_ultrachat,
    "ultra_chat": load_ultrachat,
}


def load_task_dataset(task_name: str) -> Tuple[DatasetDict, Dict[str, Any]]:
    """Load a dataset by task name from the registry."""
    if task_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[task_name]()


def _dirichlet_split_indices(labels: np.ndarray, num_clients: int, alpha: float, rng: np.random.Generator) -> List[List[int]]:
    # Standard label-wise Dirichlet partitioning
    classes = np.unique(labels)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))
        # turn proportions into counts
        counts = (proportions * len(idx_c)).astype(int)
        # fix rounding: distribute remainder
        diff = len(idx_c) - int(counts.sum())
        for k in rng.choice(num_clients, size=abs(diff), replace=True):
            counts[k] += 1 if diff > 0 else -1
        counts = np.clip(counts, 0, None)
        # assign
        start = 0
        for i in range(num_clients):
            take = int(counts[i])
            if take > 0:
                client_indices[i].extend(idx_c[start : start + take].tolist())
            start += take
    for i in range(num_clients):
        rng.shuffle(client_indices[i])
    return client_indices


def _iid_split_indices(n: int, num_clients: int, rng: np.random.Generator) -> List[List[int]]:
    idx = np.arange(n)
    rng.shuffle(idx)
    splits = np.array_split(idx, num_clients)
    return [s.tolist() for s in splits]


def split_clients_indices(
    train_ds: Dataset,
    *,
    num_clients: int,
    split_type: str,
    dirichlet_alpha: float,
    seed: int,
    label_column: str = "label",
) -> List[List[int]]:
    """Split dataset indices across clients using IID or Dirichlet distribution.
    
    Args:
        train_ds: Training dataset
        num_clients: Number of federated clients
        split_type: "iid" or "dirichlet"
        dirichlet_alpha: Concentration parameter for Dirichlet (lower = more heterogeneous)
        seed: Random seed for reproducibility
        label_column: Column name containing labels (for dirichlet split)
        
    Returns:
        List of index lists, one per client
    """
    rng = np.random.default_rng(seed)
    if split_type == "dirichlet":
        # Check for label column - use pseudo-label if needed
        if label_column in train_ds.column_names:
            labels = np.array(train_ds[label_column], dtype=np.int64)
        elif "_pseudo_label" in train_ds.column_names:
            labels = np.array(train_ds["_pseudo_label"], dtype=np.int64)
        else:
            raise ValueError(f"Dirichlet split requires label column '{label_column}' in dataset")
        idxs = _dirichlet_split_indices(labels, num_clients, dirichlet_alpha, rng)
    elif split_type == "iid":
        idxs = _iid_split_indices(len(train_ds), num_clients, rng)
    else:
        raise ValueError(f"Unknown split type: {split_type}")
    return idxs


def split_train_val_indices(idxs: List[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    if not idxs:
        return idxs, idxs
    rng = np.random.default_rng(seed)
    idxs = list(idxs)
    rng.shuffle(idxs)
    n_val = int(round(len(idxs) * val_ratio))
    n_val = max(1, n_val) if len(idxs) >= 2 else 0
    val = idxs[:n_val]
    tr = idxs[n_val:]
    return tr, val


def build_public_unlabeled_pool(
    train_ds: Dataset,
    *,
    dp_total_size: int,
    seed: int,
    label_column: str = "label",
) -> Tuple[Dataset, List[int]]:
    """Build a public unlabeled pool from the training set.
    
    For classification tasks, removes the label column.
    For instruction tasks, removes the response column.
    """
    rng = np.random.default_rng(seed)
    n = len(train_ds)
    take = min(dp_total_size, n)
    chosen = rng.choice(n, size=take, replace=False)
    chosen = chosen.astype(int).tolist()
    
    # Remove the appropriate column based on what exists
    dp = train_ds.select(chosen)
    if label_column and label_column in dp.column_names:
        dp = dp.remove_columns([label_column])
    
    return dp, chosen


def dp_round_indices(
    dp_size: int,
    *,
    round_idx: int,
    dp_samples_per_round: int,
    seed: int,
) -> List[int]:
    rng = np.random.default_rng(seed + 10_000 * (round_idx + 1))
    take = min(dp_samples_per_round, dp_size)
    idx = rng.choice(dp_size, size=take, replace=False)
    idx = idx.astype(int).tolist()
    idx.sort()
    return idx


def prepare_federated_splits(cfg: Dict[str, Any]) -> FederatedSplits:
    """Prepare federated data splits for any supported dataset."""
    # Load dataset from registry
    task_name = cfg["task"]["name"]
    dd, metadata = load_task_dataset(task_name)
    
    train_split = metadata.get("train_split", "train")
    train_ds = dd[train_split]

    num_clients = int(cfg["hardware"]["num_clients"])
    split_cfg = cfg["data"]["split"]
    split_type = str(split_cfg["type"])
    alpha = float(split_cfg.get("dirichlet_alpha", 0.5))
    val_ratio = float(split_cfg.get("client_val_ratio", 0.05))
    seed = int(cfg["experiment"]["seed"])

    # For instruction tasks, dirichlet split requires a label column
    # Use IID split if dirichlet is requested but no labels exist
    task_type = metadata.get("task_type", "classification")
    if task_type == "instruction" and split_type == "dirichlet":
        # Use category column for dirichlet if available, else fallback to IID
        category_col = metadata.get("category_column")
        if category_col and category_col in train_ds.column_names:
            # Create pseudo-labels from categories for dirichlet splitting
            categories = train_ds[category_col]
            unique_cats = list(set(categories))
            cat_to_int = {c: i for i, c in enumerate(unique_cats)}
            pseudo_labels = np.array([cat_to_int[c] for c in categories])
            # Temporarily add pseudo-label column
            train_ds = train_ds.add_column("_pseudo_label", pseudo_labels.tolist())
            client_splits = split_clients_indices(
                train_ds,
                num_clients=num_clients,
                split_type="dirichlet",
                dirichlet_alpha=alpha,
                seed=seed,
            )
            train_ds = train_ds.remove_columns(["_pseudo_label"])
        else:
            print(f"Warning: Dirichlet split not supported for {task_name} without category column. Using IID.")
            split_type = "iid"
            client_splits = split_clients_indices(
                train_ds,
                num_clients=num_clients,
                split_type=split_type,
                dirichlet_alpha=alpha,
                seed=seed,
            )
    else:
        client_splits = split_clients_indices(
            train_ds,
            num_clients=num_clients,
            split_type=split_type,
            dirichlet_alpha=alpha,
            seed=seed,
        )

    client_train_indices: List[List[int]] = []
    client_val_indices: List[List[int]] = []
    for i, idxs in enumerate(client_splits):
        tr, va = split_train_val_indices(idxs, val_ratio, seed=seed + i + 1)
        client_train_indices.append(tr)
        client_val_indices.append(va)

    # Public unlabeled pool is optional (used by FedCoFiT, not by FLoRA)
    dp_cfg = cfg["data"].get("public_unlabeled", None)
    if dp_cfg is not None:
        dp_total_size = int(dp_cfg["dp_total_size"])
        label_col = metadata.get("label_column") or metadata.get("response_column", "response")
        _, chosen_train_indices = build_public_unlabeled_pool(
            train_ds, 
            dp_total_size=dp_total_size, 
            seed=seed + 999,
            label_column=label_col,
        )
    else:
        chosen_train_indices = []  # Empty for methods that don't use public unlabeled data

    # Build return object with metadata
    splits = FederatedSplits(
        client_train_indices=client_train_indices,
        client_val_indices=client_val_indices,
        dp_source_train_indices=chosen_train_indices,
        task_type=task_type,
        labels=metadata.get("labels", []),
        text_column=metadata.get("text_column", "sentence"),
        label_column=metadata.get("label_column", "label"),
        instruction_column=metadata.get("instruction_column"),
        context_column=metadata.get("context_column"),
        response_column=metadata.get("response_column"),
    )
    
    return splits


