from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from fedcofit.prompts import build_prompt, build_instruction_prompt, normalize_label
from fedcofit.data import load_task_dataset
from fedcofit.mmlu_eval import run_mmlu_evaluation
from fedcofit.final_eval import run_final_evaluation
from fedcofit.timing import timing_record
from fedcofit.utils import ensure_dir, resolve_dtype, set_global_seed


def _torch_device(gpu_id: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def _collate(tokenizer, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # examples: dict with input_ids, attention_mask, labels (as 1-D tensors, unpadded)
    # We must pad manually because tokenizer.pad doesn't handle labels properly
    max_len = max(e["input_ids"].shape[0] for e in examples)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    input_ids_list: List[torch.Tensor] = []
    attention_mask_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    for e in examples:
        seq_len = e["input_ids"].shape[0]
        pad_len = max_len - seq_len
        # pad on the right
        input_ids_list.append(torch.cat([e["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        attention_mask_list.append(torch.cat([e["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        labels_list.append(torch.cat([e["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(input_ids_list, dim=0),
        "attention_mask": torch.stack(attention_mask_list, dim=0),
        "labels": torch.stack(labels_list, dim=0),
    }


def _make_sft_example(
    tokenizer, 
    sentence: str, 
    label: str, 
    max_seq_len: int,
    task_type: str = "classification",
    instruction: Optional[str] = None,
    context: Optional[str] = None,
    model_name: str = "default",
) -> Dict[str, torch.Tensor]:
    """Create a supervised fine-tuning example.
    
    For classification: Uses sentence + label (e.g., "positive")
    For instruction: Uses instruction + context + response
    """
    if task_type == "instruction":
        # For instruction tasks, 'sentence' is not used, 'label' is the response
        prompt = build_instruction_prompt(
            instruction=instruction or sentence,  # fallback to sentence if instruction not provided
            context=context,
            model_name=model_name,
        )
        full = prompt + label  # label is the response for instruction tasks
    else:
        # Classification task
        prompt = build_prompt(sentence, strict=False)
        full = prompt + label
    
    tok_full = tokenizer(full, truncation=True, max_length=max_seq_len, add_special_tokens=True)
    tok_prompt = tokenizer(prompt, truncation=True, max_length=max_seq_len, add_special_tokens=True)

    input_ids = torch.tensor(tok_full["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(tok_full["attention_mask"], dtype=torch.long)
    labels = input_ids.clone()
    # mask prompt tokens
    p_len = min(len(tok_prompt["input_ids"]), labels.shape[0])
    labels[:p_len] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _make_prompt_only(tokenizer, sentence: str, max_seq_len: int, strict: bool) -> Dict[str, torch.Tensor]:
    prompt = build_prompt(sentence, strict=strict)
    tok = tokenizer(prompt, truncation=True, max_length=max_seq_len, add_special_tokens=True, return_tensors="pt")
    return {"prompt": prompt, "input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]}


def _decode_new_tokens(tokenizer, output_ids: torch.Tensor, prompt_input_len: int) -> List[str]:
    """
    Extract generated tokens from model.generate output.
    
    With left-padding, the input is: [PAD, PAD, ..., TOKEN1, TOKEN2, ...]
    And output is:                   [PAD, PAD, ..., TOKEN1, TOKEN2, ..., GEN1, GEN2, ...]
    
    So we slice from the PADDED input length (not unpadded).
    """
    # output_ids: [B, prompt_input_len + max_new_tokens]
    # Slice from the padded input length to get only generated tokens
    outs: List[str] = []
    for i in range(output_ids.shape[0]):
        gen_ids = output_ids[i, prompt_input_len:]
        outs.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return outs


def _generate_batch(
    model,
    tokenizer,
    *,
    sentences: List[str],
    max_seq_len: int,
    gen_cfg: Dict[str, Any],
    strict: bool,
) -> List[Optional[str]]:
    prompts = [build_prompt(s, strict=strict) for s in sentences]
    # Use left-padding for decoder-only generation (best practice)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    tok = tokenizer(
        prompts,
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    tokenizer.padding_side = original_padding_side  # restore for training
    input_ids = tok["input_ids"].to(model.device)
    attention_mask = tok["attention_mask"].to(model.device)
    
    # Build generation kwargs - only include temperature if do_sample=True
    do_sample = bool(gen_cfg["do_sample"])
    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(gen_cfg["max_new_tokens"]),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(gen_cfg["temperature"])
    
    out = model.generate(**gen_kwargs)
    # With left-padding, new tokens are always at the end after the padded input length
    prompt_input_len = input_ids.shape[1]
    raws = _decode_new_tokens(tokenizer, out, prompt_input_len)
    return [normalize_label(r) for r in raws]


def _generate_one(
    model,
    tokenizer,
    *,
    sentence: str,
    max_seq_len: int,
    gen_cfg: Dict[str, Any],
    retry_once: bool,
) -> Optional[str]:
    # Build generation kwargs - only include temperature if do_sample=True
    do_sample = bool(gen_cfg["do_sample"])
    base_gen_kwargs = {
        "max_new_tokens": int(gen_cfg["max_new_tokens"]),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        base_gen_kwargs["temperature"] = float(gen_cfg["temperature"])

    # Single-sample generation (no padding needed, so no warning)
    obj = _make_prompt_only(tokenizer, sentence, max_seq_len, strict=False)
    input_ids = obj["input_ids"].to(model.device)
    attention_mask = obj["attention_mask"].to(model.device)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **base_gen_kwargs,
    )
    gen_ids = out[0, input_ids.shape[1] :]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
    lab = normalize_label(raw)
    if lab is not None:
        return lab

    if not retry_once:
        return None

    # Retry with stricter prompt
    obj2 = _make_prompt_only(tokenizer, sentence, max_seq_len, strict=True)
    input_ids2 = obj2["input_ids"].to(model.device)
    attention_mask2 = obj2["attention_mask"].to(model.device)
    out2 = model.generate(
        input_ids=input_ids2,
        attention_mask=attention_mask2,
        **base_gen_kwargs,
    )
    gen_ids2 = out2[0, input_ids2.shape[1] :]
    raw2 = tokenizer.decode(gen_ids2, skip_special_tokens=True)
    return normalize_label(raw2)


def generate_responses_for_instructions(
    model,
    tokenizer,
    *,
    instructions: List[str],
    contexts: Optional[List[str]],
    max_seq_len: int,
    gen_cfg: Dict[str, Any],
    batch_size: int,
    model_name: str = "default",
) -> List[str]:
    """
    Generate full text responses for instruction-following tasks.
    Returns list of generated response strings.
    """
    if not instructions:
        return []
    
    bs = max(1, int(batch_size))
    responses: List[str] = []
    
    # Use left-padding for generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    try:
        for start in range(0, len(instructions), bs):
            chunk_instructions = instructions[start:start + bs]
            chunk_contexts = contexts[start:start + bs] if contexts else [None] * len(chunk_instructions)
            
            # Build prompts
            prompts = []
            for inst, ctx in zip(chunk_instructions, chunk_contexts):
                prompt = build_instruction_prompt(
                    instruction=inst,
                    context=ctx if ctx and str(ctx).strip() else None,
                    model_name=model_name,
                )
                prompts.append(prompt)
            
            # Tokenize
            tok = tokenizer(
                prompts,
                truncation=True,
                max_length=max_seq_len,
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            input_ids = tok["input_ids"].to(model.device)
            attention_mask = tok["attention_mask"].to(model.device)
            
            # Generate
            do_sample = bool(gen_cfg.get("do_sample", False))
            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": int(gen_cfg.get("max_new_tokens", 128)),
                "do_sample": do_sample,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = float(gen_cfg.get("temperature", 0.7))
            # Add repetition penalty to prevent degenerate repetition loops
            rep_penalty = gen_cfg.get("repetition_penalty")
            if rep_penalty is not None:
                gen_kwargs["repetition_penalty"] = float(rep_penalty)
            
            with torch.no_grad():
                output = model.generate(**gen_kwargs)
            
            # Decode only the generated part
            prompt_len = input_ids.shape[1]
            for i in range(output.shape[0]):
                gen_ids = output[i, prompt_len:]
                response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                responses.append(response if response else "")
    finally:
        tokenizer.padding_side = original_padding_side
    
    return responses


def generate_labels_constrained(
    model,
    tokenizer,
    *,
    sentences: List[str],
    max_seq_len: int,
    gen_cfg: Dict[str, Any],
    retry_once: bool,
    batch_size: int,
) -> List[Optional[str]]:
    """
    Constrained outputs via normalization + one strict retry.
    Returns list of normalized labels or None (invalid).
    For CLASSIFICATION tasks only.
    """
    if not sentences:
        return []
    bs = max(1, int(batch_size))
    preds: List[Optional[str]] = [None] * len(sentences)

    # First pass (non-strict)
    for start in range(0, len(sentences), bs):
        chunk = sentences[start : start + bs]
        chunk_preds = _generate_batch(model, tokenizer, sentences=chunk, max_seq_len=max_seq_len, gen_cfg=gen_cfg, strict=False)
        preds[start : start + bs] = chunk_preds

    if not retry_once:
        return preds

    # Retry strict only for invalids
    invalid_idx = [i for i, p in enumerate(preds) if p is None]
    if not invalid_idx:
        return preds
    retry_sents = [sentences[i] for i in invalid_idx]
    retry_preds: List[Optional[str]] = []
    for start in range(0, len(retry_sents), bs):
        chunk = retry_sents[start : start + bs]
        retry_preds.extend(
            _generate_batch(model, tokenizer, sentences=chunk, max_seq_len=max_seq_len, gen_cfg=gen_cfg, strict=True)
        )
    for ii, p in zip(invalid_idx, retry_preds):
        preds[ii] = p
    return preds


def _train_mixed_epochs(
    *,
    model,
    tokenizer,
    optimizer,
    scheduler,
    max_seq_len: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    epochs: int,
    private_sentences: List[str],
    private_labels: List[str],
    pseudo_buffer: List[Tuple[str, str]],
    mix_private_ratio: float,
    rng: np.random.Generator,
    task_type: str = "classification",
    model_name: str = "default",
) -> int:
    """
    Sequential training on combined private + pseudo data (like FLoRA).
    
    Creates a combined dataset with the appropriate mix ratio, then iterates
    through it sequentially with gradient accumulation. This ensures:
    - Each private sample is seen exactly once per epoch
    - Pseudo samples are included proportionally to mix_private_ratio
    - Consistent with FLoRA's training behavior
    """
    model.train()
    
    # Build combined dataset based on mix_private_ratio
    # If mix_private_ratio=0.5, we want 50% private, 50% pseudo
    # So for N private samples, we add N pseudo samples (sampled with replacement if needed)
    combined_data: List[Tuple[str, str]] = []
    
    # Add all private samples
    for sent, lab in zip(private_sentences, private_labels):
        combined_data.append((sent, lab))
    
    # Add pseudo samples to match the ratio
    # If mix_private_ratio=0.5: add same number of pseudo as private
    # If mix_private_ratio=0.7: add fewer pseudo (private=70%, pseudo=30%)
    if len(pseudo_buffer) > 0 and mix_private_ratio < 1.0:
        n_private = len(private_sentences)
        # Calculate how many pseudo samples needed for desired ratio
        # private / (private + pseudo) = mix_private_ratio
        # pseudo = private * (1 - mix_private_ratio) / mix_private_ratio
        n_pseudo_needed = int(n_private * (1.0 - mix_private_ratio) / mix_private_ratio)
        n_pseudo_needed = max(0, n_pseudo_needed)
        
        # Sample pseudo samples (with replacement if needed)
        for i in range(n_pseudo_needed):
            idx = int(rng.integers(0, len(pseudo_buffer)))
            combined_data.append(pseudo_buffer[idx])
    
    n = len(combined_data)
    if n == 0:
        return 0
    
    # Pre-create training examples
    examples = []
    for sent, lab in combined_data:
        ex = _make_sft_example(
            tokenizer, sent, lab, max_seq_len,
            task_type=task_type,
            instruction=sent if task_type == "instruction" else None,
            model_name=model_name,
        )
        examples.append(ex)
    
    steps_done = 0
    
    for epoch in range(epochs):
        # Shuffle combined data each epoch
        indices = rng.permutation(n).tolist()
        optimizer.zero_grad(set_to_none=True)
        accum_count = 0
        
        for start in range(0, n, micro_batch_size):
            batch_idxs = indices[start:start + micro_batch_size]
            batch_ex = [examples[i] for i in batch_idxs]
            batch = _collate(tokenizer, batch_ex)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / float(grad_accum_steps)
            loss.backward()
            accum_count += 1
            
            if accum_count >= grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                steps_done += 1
                accum_count = 0
        
        # Handle remaining gradients at end of epoch
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            steps_done += 1
    
    return steps_done


def _train_mixed_steps(
    *,
    model,
    tokenizer,
    optimizer,
    scheduler,
    max_seq_len: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    steps: int,
    private_sentences: List[str],
    private_labels: List[str],
    pseudo_buffer: List[Tuple[str, str]],
    mix_private_ratio: float,
    rng: np.random.Generator,
    task_type: str = "classification",
    model_name: str = "default",
) -> int:
    """
    Fixed-step training on combined private + pseudo data.
    
    Trains for exactly `steps` optimizer updates, sampling from combined data
    with the appropriate mix ratio. This allows fair comparison with FLoRA
    when both use the same number of steps.
    
    With mix_private_ratio=0.5:
    - FLoRA sees: steps × batch_size private samples
    - FedCoFiT sees: steps × batch_size × 0.5 private + steps × batch_size × 0.5 pseudo
    """
    model.train()
    
    # Build combined dataset
    combined_data: List[Tuple[str, str]] = []
    
    # Add all private samples
    for sent, lab in zip(private_sentences, private_labels):
        combined_data.append((sent, lab))
    
    # Add pseudo samples to match the ratio
    if len(pseudo_buffer) > 0 and mix_private_ratio < 1.0:
        n_private = len(private_sentences)
        n_pseudo_needed = int(n_private * (1.0 - mix_private_ratio) / mix_private_ratio)
        n_pseudo_needed = max(0, n_pseudo_needed)
        
        for i in range(n_pseudo_needed):
            idx = int(rng.integers(0, len(pseudo_buffer)))
            combined_data.append(pseudo_buffer[idx])
    
    n = len(combined_data)
    if n == 0:
        return 0
    
    # Pre-create training examples
    examples = []
    for sent, lab in combined_data:
        ex = _make_sft_example(
            tokenizer, sent, lab, max_seq_len,
            task_type=task_type,
            instruction=sent if task_type == "instruction" else None,
            model_name=model_name,
        )
        examples.append(ex)
    
    # Train for fixed number of steps
    optimizer.zero_grad(set_to_none=True)
    steps_done = 0
    
    while steps_done < steps:
        # Shuffle indices for this pass
        indices = rng.permutation(n).tolist()
        idx_ptr = 0
        
        while steps_done < steps and idx_ptr < n:
            # Accumulate gradients over grad_accum_steps micro-batches
            for ga in range(grad_accum_steps):
                if idx_ptr >= n:
                    # Reshuffle if we run out of data
                    indices = rng.permutation(n).tolist()
                    idx_ptr = 0
                
                # Get batch indices
                batch_end = min(idx_ptr + micro_batch_size, n)
                batch_idxs = indices[idx_ptr:batch_end]
                idx_ptr = batch_end
                
                batch_ex = [examples[i] for i in batch_idxs]
                batch = _collate(tokenizer, batch_ex)
                batch = {k: v.to(model.device) for k, v in batch.items()}
                out = model(**batch)
                loss = out.loss / float(grad_accum_steps)
                loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            steps_done += 1
    
    return steps_done


def _train_private_epochs(
    *,
    model,
    tokenizer,
    optimizer,
    scheduler,
    max_seq_len: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    epochs: int,
    private_sentences: List[str],
    private_labels: List[str],
    rng: np.random.Generator,
    task_type: str = "classification",
    model_name: str = "default",
) -> None:
    """
    Spec-faithful expert phase: iterate over the private dataset for `epochs`, shuffling each epoch.
    """
    model.train()
    n = len(private_sentences)
    for ep in range(epochs):
        order = np.arange(n)
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        # iterate mini-batches; grad accumulation happens across consecutive micro-batches
        step_in_accum = 0
        for start in range(0, n, micro_batch_size):
            idxs = order[start : start + micro_batch_size].tolist()
            batch_ex = []
            for j in idxs:
                sent, lab = private_sentences[j], private_labels[j]
                batch_ex.append(_make_sft_example(
                    tokenizer, sent, lab, max_seq_len,
                    task_type=task_type,
                    instruction=sent if task_type == "instruction" else None,
                    model_name=model_name,
                ))
            batch = _collate(tokenizer, batch_ex)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / float(grad_accum_steps)
            loss.backward()
            step_in_accum += 1
            if step_in_accum >= grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step_in_accum = 0
        # flush partial accumulation at end of epoch
        if step_in_accum > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)


def _build_model_and_tokenizer(cfg: Dict[str, Any], device: torch.device):
    dtype = resolve_dtype(cfg["model"]["dtype"])
    prefer_flash = bool(cfg["model"].get("use_flash_attention_2", True))
    attn_impl = "flash_attention_2" if prefer_flash else "sdpa"

    tok = AutoTokenizer.from_pretrained(cfg["model"]["base_model_name_or_path"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model"]["base_model_name_or_path"],
            torch_dtype=dtype,
            device_map=None,
            attn_implementation=attn_impl,
        )
    except Exception as e:
        # FlashAttention2 may not be available in some environments; fallback to SDPA.
        if attn_impl == "flash_attention_2":
            model = AutoModelForCausalLM.from_pretrained(
                cfg["model"]["base_model_name_or_path"],
                torch_dtype=dtype,
                device_map=None,
                attn_implementation="sdpa",
            )
        else:
            raise
    model.to(device)

    lcfg = cfg["lora"]
    peft_cfg = LoraConfig(
        r=int(lcfg["r"]),
        lora_alpha=int(lcfg["alpha"]),
        lora_dropout=float(lcfg["dropout"]),
        target_modules=list(lcfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model, tok


def _load_private_data_indices(
    cfg: Dict[str, Any], 
    client_id: int, 
    splits: Dict[str, Any]
) -> Tuple[List[str], List[str], Optional[List[str]], Optional[List[str]]]:
    """Load client's private data.
    
    Returns:
        For classification: (sentences, labels, None, None)
        For instruction: (instructions, responses, contexts, None)
    """
    task_name = cfg["task"]["name"]
    dd, metadata = load_task_dataset(task_name)
    train_split = metadata.get("train_split", "train")
    train_ds = dd[train_split]
    
    idxs = splits["client_train_indices"][client_id]
    ds = train_ds.select(list(map(int, idxs)))
    
    task_type = metadata.get("task_type", "classification")
    
    if task_type == "instruction":
        # Instruction-following task
        instruction_col = metadata.get("instruction_column", "instruction")
        context_col = metadata.get("context_column", "context")
        response_col = metadata.get("response_column", "response")
        
        instructions = list(ds[instruction_col])
        responses = list(ds[response_col])
        contexts = list(ds[context_col]) if context_col in ds.column_names else [None] * len(instructions)
        
        return instructions, responses, contexts, None
    else:
        # Classification task (SST-2, etc.)
        text_col = metadata.get("text_column", "sentence")
        label_col = metadata.get("label_column", "label")
        labels_list = metadata.get("labels", ["negative", "positive"])
        
        sentences = list(ds[text_col])
        labels_int = list(ds[label_col])
        labels = [labels_list[int(x)] for x in labels_int]
        
        return sentences, labels, None, None


def _load_public_unlabeled(cfg: Dict[str, Any], splits: Dict[str, Any]) -> Tuple[List[str], Optional[List[str]]]:
    """Load public unlabeled data pool.
    
    Returns:
        For classification: (sentences, None)
        For instruction: (instructions, contexts)
    """
    task_name = cfg["task"]["name"]
    dd, metadata = load_task_dataset(task_name)
    train_split = metadata.get("train_split", "train")
    train_ds = dd[train_split]
    
    chosen = list(map(int, splits["dp_source_train_indices"]))
    if not chosen:
        return [], None
        
    dp = train_ds.select(chosen)
    
    task_type = metadata.get("task_type", "classification")
    
    if task_type == "instruction":
        instruction_col = metadata.get("instruction_column", "instruction")
        context_col = metadata.get("context_column", "context")
        instructions = list(dp[instruction_col])
        contexts = list(dp[context_col]) if context_col in dp.column_names else None
        return instructions, contexts
    else:
        text_col = metadata.get("text_column", "sentence")
        return list(dp[text_col]), None


def _dp_round_indices(dp_size: int, round_idx: int, dp_samples_per_round: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed + 10_000 * (round_idx + 1))
    take = min(dp_samples_per_round, dp_size)
    idx = rng.choice(dp_size, size=take, replace=False).astype(int).tolist()
    idx.sort()
    return idx


def client_loop(
    *,
    cfg: Dict[str, Any],
    client_id: int,
    gpu_id: int,
    splits: Dict[str, Any],
    server_in_queue,
    client_out_queue,
    event_queue,
) -> None:
    seed = int(cfg["experiment"]["seed"]) + 1000 + client_id
    set_global_seed(seed)

    device = _torch_device(gpu_id)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    event_queue.put(
        {
            "type": "client_info",
            "data": {
                "client_id": client_id,
                "gpu_id": gpu_id,
                "device": str(device),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "cuda_device_name": torch.cuda.get_device_name(gpu_id) if torch.cuda.is_available() else None,
            },
        }
    )

    model, tokenizer = _build_model_and_tokenizer(cfg, device=device)

    max_seq_len = int(cfg["model"]["max_seq_len"])
    micro_batch_size = int(cfg["training"]["micro_batch_size"])
    grad_accum_steps = int(cfg["training"]["grad_accum_steps"])
    lr = float(cfg["training"]["learning_rate"])
    warmup_ratio = float(cfg["training"]["warmup_ratio"])
    
    # Get task type from config
    task_type = cfg["task"].get("type", "classification")
    model_name = cfg["model"].get("base_model_name_or_path", "default")

    # data - now returns 4 values for flexibility
    priv_data = _load_private_data_indices(cfg, client_id, splits)
    priv_sentences, priv_labels, priv_contexts, _ = priv_data
    
    dp_data = _load_public_unlabeled(cfg, splits)
    dp_sentences, dp_contexts = dp_data
    dp_size = len(dp_sentences) if dp_sentences else 0

    # pseudo buffer FIFO
    pseudo_max = int(cfg["federated"]["pseudo_buffer_max"])
    pseudo_buffer: List[Tuple[str, str]] = []
    
    # Network simulation config
    network_cfg = cfg.get("network", {})
    simulate_latency = bool(network_cfg.get("simulate_latency", False))
    bandwidth_mbps = float(network_cfg.get("bandwidth_mbps", 100.0))  # Default 100 Mbps
    
    def simulate_network_delay(num_bytes: int) -> float:
        """Simulate network transfer delay based on bandwidth."""
        if not simulate_latency or num_bytes <= 0:
            return 0.0
        # Convert: bytes / (Mbps * 1e6 / 8) = seconds
        delay_seconds = num_bytes / (bandwidth_mbps * 1e6 / 8)
        time.sleep(delay_seconds)
        return delay_seconds

    # Clients send text responses to server, server computes embeddings
    # This reduces client resource requirements and communication overhead
    emb_dim = 384  # MiniLM-L6-v2 dimension (for reporting only)
    
    event_queue.put({"type": "client_info", "data": {"client_id": client_id, "emb_dim": emb_dim}})

    # optimizer/scheduler helper for expert: total steps = ceil(n/micro)*epochs
    expert_epochs = int(cfg["expert"]["expert_epochs"])
    steps_per_epoch = max(1, math.ceil(len(priv_sentences) / float(micro_batch_size)))
    total_steps = int(steps_per_epoch * expert_epochs)
    warmup_steps = int(round(total_steps * warmup_ratio))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # expert training
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed + 1)
    _train_private_epochs(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        max_seq_len=max_seq_len,
        micro_batch_size=micro_batch_size,
        grad_accum_steps=grad_accum_steps,
        epochs=expert_epochs,
        private_sentences=priv_sentences,
        private_labels=priv_labels,
        rng=rng,
        task_type=task_type,
        model_name=model_name,
    )
    event_queue.put({"type": "timing", "data": timing_record("expert_training", time.perf_counter() - t0, client_id, None)})

    # save expert adapter
    ckpt_root = cfg["expert"]["save_dir"]
    save_dir = os.path.join(ckpt_root, f"client_{client_id}", "expert_adapter")
    ensure_dir(save_dir)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    event_queue.put({"type": "checkpoint", "data": {"client_id": client_id, "path": save_dir}})

    # evaluation after expert phase
    acc0 = evaluate_client(cfg, model, tokenizer, device, client_id, round_idx=0, splits=splits)
    event_queue.put({"type": "metrics", "data": acc0})

    rounds = int(cfg["federated"]["rounds"])
    dp_samples_per_round = int(cfg["data"]["public_unlabeled"]["dp_samples_per_round"])
    gen_cfg = cfg["generation"]
    retry_once = bool(gen_cfg.get("retry_once_with_strict_prompt", True))
    gen_batch_size = int(gen_cfg.get("batch_size", 32))

    mix_private_ratio = float(cfg["federated"]["mix_private_ratio"])
    local_epochs = cfg["federated"].get("local_epochs", None)
    local_steps = cfg["federated"].get("local_steps", None)
    
    # Determine training mode: local_steps takes priority over local_epochs
    use_steps_mode = local_steps is not None

    for r in range(rounds):
        round_indices = _dp_round_indices(dp_size, r, dp_samples_per_round, seed=int(cfg["experiment"]["seed"]))
        round_sents = [dp_sentences[j] for j in round_indices]
        round_contexts = [dp_contexts[j] for j in round_indices] if dp_contexts else None

        # prediction generation time
        t_pred = time.perf_counter()
        
        if task_type == "instruction":
            # For instruction tasks: generate full response text
            preds = generate_responses_for_instructions(
                model,
                tokenizer,
                instructions=round_sents,
                contexts=round_contexts,
                max_seq_len=max_seq_len,
                gen_cfg=gen_cfg,
                batch_size=gen_batch_size,
                model_name=model_name,
            )
        else:
            # For classification: generate and normalize to "positive"/"negative"
            preds = generate_labels_constrained(
                model,
                tokenizer,
                sentences=round_sents,
                max_seq_len=max_seq_len,
                gen_cfg=gen_cfg,
                retry_once=retry_once,
                batch_size=gen_batch_size,
            )
        event_queue.put({"type": "timing", "data": timing_record("prediction_generation", time.perf_counter() - t_pred, client_id, r)})

        # Prepare text responses to send to server (server computes embeddings)
        if task_type == "instruction":
            # For instruction tasks: send full response text
            labels_for_server: List[Optional[str]] = [p if p else "" for p in preds]
        else:
            # For classification: send "positive"/"negative" labels
            labels_for_server: List[Optional[str]] = [p if p in {"positive", "negative"} else None for p in preds]

        # Upload text responses to server (server computes embeddings)
        t_up = time.perf_counter()
        
        # Calculate bytes_up: only text is sent
        bytes_up = sum(len(lab.encode('utf-8')) if lab else 0 for lab in labels_for_server)
        
        server_in_queue.put(
            {
                "type": "upload",
                "round": r,
                "client_id": client_id,
                "labels": labels_for_server,  # Text responses only
            }
        )
        event_queue.put({"type": "timing", "data": timing_record("upload_packaging", time.perf_counter() - t_up, client_id, r)})
        
        # Simulate network upload delay
        upload_delay = simulate_network_delay(bytes_up)
        if upload_delay > 0:
            event_queue.put({"type": "timing", "data": timing_record("network_upload", upload_delay, client_id, r)})

        # receive consensus
        msg = client_out_queue.get()
        assert msg["type"] == "consensus" and int(msg["round"]) == r
        consensus: List[str] = list(msg["labels"])
        
        # Simulate network download delay (consensus labels are tiny)
        bytes_down = len(consensus) * 10  # Approximate bytes for label strings
        download_delay = simulate_network_delay(bytes_down)
        if download_delay > 0:
            event_queue.put({"type": "timing", "data": timing_record("network_download", download_delay, client_id, r)})

        # add pseudo-labeled samples to FIFO buffer
        # per spec: if THIS client had invalid output after retry, skip that sample for client-side training
        added_count = 0
        if task_type == "instruction":
            # For instruction tasks: consensus is the full response text
            for i, (sent, lab) in enumerate(zip(round_sents, consensus)):
                my_pred = labels_for_server[i] if i < len(labels_for_server) else None
                if not my_pred:  # Skip if we didn't generate a valid response
                    continue
                if not lab:  # Skip if consensus is empty
                    continue
                # Add instruction + consensus response to pseudo buffer
                pseudo_buffer.append((sent, lab))
                added_count += 1
            
            # Log sample consensus responses for debugging (first 2)
            if client_id == 0 and len(consensus) > 0:
                print(f"\n[Client {client_id}] Round {r} - Added {added_count}/{len(consensus)} pseudo-labels to buffer")
                print(f"[Client {client_id}] Pseudo buffer size: {len(pseudo_buffer)}")
                for i, (sent, lab) in enumerate(zip(round_sents[:2], consensus[:2])):
                    print(f"  Sample {i}:")
                    print(f"    Instruction: {sent[:80]}...")
                    print(f"    Consensus Response: {lab[:120]}..." if len(lab) > 120 else f"    Consensus Response: {lab}")
        else:
            # For classification: consensus is "positive" or "negative"
            for sent, lab, my_lab in zip(round_sents, consensus, labels_for_server):
                if my_lab is None:
                    continue
                if lab not in {"positive", "negative"}:
                    continue
                pseudo_buffer.append((sent, lab))
        
        if len(pseudo_buffer) > pseudo_max:
            pseudo_buffer[:] = pseudo_buffer[-pseudo_max:]

        # Round training: sequential iteration through combined private+pseudo dataset
        # Supports two modes:
        # 1. local_steps: fixed number of optimizer steps (for fair comparison with FLoRA)
        # 2. local_epochs: N epochs through combined data (default)
        
        rng_r = np.random.default_rng(seed + 100 + r)
        t_tr = time.perf_counter()
        
        if use_steps_mode:
            # Fixed steps mode: train for exactly local_steps optimizer updates
            steps_this_round = int(local_steps)
            
            optimizer_r = torch.optim.AdamW(model.parameters(), lr=lr)
            warmup_steps_r = int(round(steps_this_round * warmup_ratio))
            scheduler_r = get_linear_schedule_with_warmup(
                optimizer_r, num_warmup_steps=warmup_steps_r, num_training_steps=steps_this_round
            )
            
            _train_mixed_steps(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer_r,
                scheduler=scheduler_r,
                max_seq_len=max_seq_len,
                micro_batch_size=micro_batch_size,
                grad_accum_steps=grad_accum_steps,
                steps=steps_this_round,
                private_sentences=priv_sentences,
                private_labels=priv_labels,
                pseudo_buffer=pseudo_buffer,
                mix_private_ratio=mix_private_ratio,
                rng=rng_r,
                task_type=task_type,
                model_name=model_name,
            )
        else:
            # Epochs mode: train for N epochs through combined data
            epochs = int(local_epochs) if local_epochs else 1
            
            # Calculate combined dataset size for scheduler
            n_private = len(priv_sentences)
            n_pseudo_needed = int(n_private * (1.0 - mix_private_ratio) / mix_private_ratio) if mix_private_ratio < 1.0 and len(pseudo_buffer) > 0 else 0
            combined_size = n_private + n_pseudo_needed
            
            # Calculate steps: one pass through combined data with grad accumulation
            effective_batch = micro_batch_size * grad_accum_steps
            steps_this_round = max(1, math.ceil(combined_size / effective_batch)) * epochs
            
            optimizer_r = torch.optim.AdamW(model.parameters(), lr=lr)
            warmup_steps_r = int(round(steps_this_round * warmup_ratio))
            scheduler_r = get_linear_schedule_with_warmup(
                optimizer_r, num_warmup_steps=warmup_steps_r, num_training_steps=steps_this_round
            )
            
            _train_mixed_epochs(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer_r,
                scheduler=scheduler_r,
                max_seq_len=max_seq_len,
                micro_batch_size=micro_batch_size,
                grad_accum_steps=grad_accum_steps,
                epochs=epochs,
                private_sentences=priv_sentences,
                private_labels=priv_labels,
                pseudo_buffer=pseudo_buffer,
                mix_private_ratio=mix_private_ratio,
                rng=rng_r,
                task_type=task_type,
                model_name=model_name,
            )
        
        event_queue.put({"type": "timing", "data": timing_record("round_training", time.perf_counter() - t_tr, client_id, r)})

        # evaluation after round r => report as round r+1 in logs
        acc = evaluate_client(cfg, model, tokenizer, device, client_id, round_idx=r + 1, splits=splits)
        event_queue.put({"type": "metrics", "data": acc})

    # ========================================
    # Final Evaluation (End-of-Training)
    # ========================================
    # For instruction-following tasks: Held-out NLL, IFEval, MMLU
    # For classification tasks: MMLU only (accuracy is computed per-round)
    # In FedCoFiT, each client has a different model (personalized),
    # so we run evaluations on ALL clients to capture the variance.
    
    eval_cfg = cfg.get("evaluation", {})
    task_type = cfg.get("task", {}).get("type", "classification")
    
    # Determine if this client should run final evaluation
    eval_clients = eval_cfg.get("final_eval_clients", "all")  # "all", "first", or list like [0, 5, 9]
    should_eval = False
    if eval_clients == "all":
        should_eval = True
    elif eval_clients == "first":
        should_eval = (client_id == 0)
    elif isinstance(eval_clients, list):
        should_eval = (client_id in eval_clients)
    
    if should_eval:
        if task_type == "instruction":
            # Full evaluation suite for instruction-following tasks
            print(f"\n[Client {client_id}] Running final evaluation suite...")
            out_dir = cfg.get("_internal_output_dir")
            final_results = run_final_evaluation(model, tokenizer, cfg, client_id=client_id, output_dir=out_dir)
            event_queue.put({
                "type": "final_eval_results",
                "data": {
                    "client_id": client_id,
                    **final_results,
                }
            })
        else:
            # Classification tasks: just MMLU for regression check
            mmlu_cfg = eval_cfg.get("mmlu", {})
            if mmlu_cfg.get("enabled", False):
                print(f"\n[Client {client_id}] Running MMLU benchmark evaluation...")
                mmlu_results = run_mmlu_evaluation(model, tokenizer, cfg)
                event_queue.put({
                    "type": "mmlu_results",
                    "data": {
                        "client_id": client_id,
                        **mmlu_results,
                    }
                })

    event_queue.put({"type": "done", "data": {"client_id": client_id}})


def evaluate_client(
    cfg: Dict[str, Any], 
    model, 
    tokenizer, 
    device: torch.device, 
    client_id: int, 
    round_idx: int,
    splits: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate client model on validation/test set.
    
    For classification tasks (SST-2): Computes accuracy
    For instruction-following tasks (Alpaca, Dolly): Computes perplexity
    
    For instruction tasks, evaluates on the CLIENT'S OWN TRAINING DATA
    (not random samples from full dataset). This gives a meaningful metric
    that should DECREASE during training.
    """
    import math
    
    task_name = cfg["task"]["name"]
    dd, metadata = load_task_dataset(task_name)
    task_type = metadata.get("task_type", "classification")
    
    gen_cfg = cfg["generation"]
    max_seq_len = int(cfg["model"]["max_seq_len"])
    retry_once = bool(gen_cfg.get("retry_once_with_strict_prompt", True))
    gen_batch_size = int(gen_cfg.get("batch_size", 32))
    model_name = cfg["model"].get("base_model_name_or_path", "default")

    model.eval()
    t0 = time.perf_counter()
    
    if task_type == "instruction":
        # For instruction tasks, compute perplexity on CLIENT'S OWN DATA
        train_split = metadata.get("train_split", "train")
        train_ds = dd[train_split]
        
        # Get evaluation config
        eval_cfg = cfg.get("evaluation", {})
        per_round_cfg = eval_cfg.get("per_round", {})
        max_eval_samples = int(per_round_cfg.get("max_samples", 300))
        
        # Use client's own training indices if available
        if splits is not None and "client_train_indices" in splits:
            client_train_idxs = splits["client_train_indices"][client_id]
            # Use a fixed subset of client's data for evaluation
            rng = np.random.default_rng(cfg["experiment"]["seed"] + 999)  # Fixed seed
            eval_size = min(max_eval_samples, len(client_train_idxs))
            if len(client_train_idxs) > eval_size:
                selected = rng.choice(len(client_train_idxs), size=eval_size, replace=False)
                eval_idxs = [client_train_idxs[i] for i in selected]
            else:
                eval_idxs = client_train_idxs
        else:
            # Fallback: use random samples from full dataset (legacy behavior)
            eval_size = min(max_eval_samples, len(train_ds))
            rng = np.random.default_rng(cfg["experiment"]["seed"] + 999)
            eval_idxs = rng.choice(len(train_ds), size=eval_size, replace=False).tolist()
        
        eval_ds = train_ds.select(eval_idxs)
        
        instruction_col = metadata.get("instruction_column", "instruction")
        context_col = metadata.get("context_column", "context")
        response_col = metadata.get("response_column", "response")
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i in range(len(eval_ds)):
                instruction = eval_ds[instruction_col][i]
                context = eval_ds[context_col][i] if context_col in eval_ds.column_names else None
                response = eval_ds[response_col][i]
                
                # Create example and compute loss
                prompt = build_instruction_prompt(
                    instruction=instruction,
                    context=context if context and str(context).strip() else None,
                    model_name=model_name,
                )
                full = prompt + response
                
                tok_full = tokenizer(full, truncation=True, max_length=max_seq_len, add_special_tokens=True)
                tok_prompt = tokenizer(prompt, truncation=True, max_length=max_seq_len, add_special_tokens=True)
                
                input_ids = torch.tensor(tok_full["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
                attention_mask = torch.tensor(tok_full["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)
                labels = input_ids.clone()
                
                # Mask prompt tokens
                p_len = min(len(tok_prompt["input_ids"]), labels.shape[1])
                labels[0, :p_len] = -100
                
                # Count non-masked tokens first
                num_tokens = (labels != -100).sum().item()
                
                # Skip if no response tokens (all masked)
                if num_tokens == 0:
                    continue
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss.item()
                
                # Skip if loss is NaN or infinite (numerical issues)
                if not math.isfinite(loss):
                    continue
                
                total_loss += loss * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / max(1, total_tokens)
        perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow
        
        seconds = time.perf_counter() - t0
        return {
            "client_id": client_id,
            "round": round_idx,
            "perplexity": float(perplexity),
            "avg_loss": float(avg_loss),
            "total_tokens": int(total_tokens),
            "eval_seconds": float(seconds),
        }
    else:
        # Classification evaluation (SST-2, etc.)
        val_split = metadata.get("val_split", "validation")
        if val_split and val_split in dd:
            val_ds = dd[val_split]
        else:
            # Use a subset of training data for validation
            train_split = metadata.get("train_split", "train")
            train_ds = dd[train_split]
            eval_size = min(500, len(train_ds))
            rng = np.random.default_rng(cfg["experiment"]["seed"] + 999)
            eval_idxs = rng.choice(len(train_ds), size=eval_size, replace=False).tolist()
            val_ds = train_ds.select(eval_idxs)
        
        text_col = metadata.get("text_column", "sentence")
        label_col = metadata.get("label_column", "label")
        labels_list = metadata.get("labels", ["negative", "positive"])
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            sents = [str(s) for s in val_ds[text_col]]
            golds = [labels_list[int(x)] for x in val_ds[label_col]]
            preds = generate_labels_constrained(
                model,
                tokenizer,
                sentences=sents,
                max_seq_len=max_seq_len,
                gen_cfg=gen_cfg,
                retry_once=retry_once,
                batch_size=gen_batch_size,
            )
            for p, g in zip(preds, golds):
                if p is not None and p == g:
                    correct += 1
                total += 1
        
        seconds = time.perf_counter() - t0
        return {
            "client_id": client_id,
            "round": round_idx,
            "accuracy": float(correct) / float(max(1, total)),
            "correct": int(correct),
            "total": int(total),
            "eval_seconds": float(seconds),
        }


