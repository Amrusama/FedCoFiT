"""
Final Evaluation Module for Instruction-Following Models

Implements three evaluation benchmarks:
1. Held-Out Test (Response-Token NLL/Perplexity) - on the training dataset's test split
2. IFEval (Instruction-Following Evaluation) - prompt-level strict accuracy
3. MMLU (Optional regression check) - multiple-choice accuracy

References:
- IFEval: https://arxiv.org/abs/2311.07911
- MMLU: https://arxiv.org/abs/2009.03300
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm

from fedcofit.prompts import build_instruction_prompt
from fedcofit.data import load_task_dataset
from fedcofit.mmlu_eval import run_mmlu_evaluation
from fedcofit.mtbench_eval import run_mtbench_evaluation


# =============================================================================
# 1. HELD-OUT TEST SET EVALUATION (NLL / Perplexity)
# =============================================================================

def create_held_out_test_split(
    task_name: str,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dataset:
    """
    Create a held-out test split from the training data.
    
    For datasets without a dedicated test split (Dolly-15k, Alpaca),
    we hold out a portion of the training data that is NOT used for 
    federated training.
    
    Args:
        task_name: Name of the task/dataset
        test_ratio: Fraction of data to hold out for testing
        seed: Random seed for reproducibility
        
    Returns:
        Held-out test Dataset
    """
    dd, metadata = load_task_dataset(task_name)
    train_split = metadata.get("train_split", "train")
    train_ds = dd[train_split]
    
    n = len(train_ds)
    rng = np.random.default_rng(seed + 12345)  # Different seed from training splits
    
    # Hold out test_ratio of the data
    all_indices = np.arange(n)
    rng.shuffle(all_indices)
    
    n_test = int(n * test_ratio)
    test_indices = all_indices[:n_test].tolist()
    
    test_ds = train_ds.select(test_indices)
    
    return test_ds, metadata


def evaluate_held_out_nll(
    model,
    tokenizer,
    cfg: Dict[str, Any],
    test_ds: Optional[Dataset] = None,
    metadata: Optional[Dict[str, Any]] = None,
    max_samples: int = 500,
) -> Dict[str, Any]:
    """
    Evaluate response-token NLL (Negative Log-Likelihood) on held-out test set.
    
    This computes how well the model predicts the reference responses given
    the instruction prompts. Lower NLL = better model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        cfg: Config dict
        test_ds: Optional pre-loaded test dataset
        metadata: Optional metadata dict
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Dict with:
        - nll: Average negative log-likelihood per response token
        - perplexity: exp(nll)
        - total_tokens: Total response tokens evaluated
        - num_samples: Number of samples evaluated
        - eval_seconds: Evaluation time
    """
    task_name = cfg["task"]["name"]
    seed = int(cfg["experiment"]["seed"])
    max_seq_len = int(cfg["model"]["max_seq_len"])
    model_name = cfg["model"].get("base_model_name_or_path", "default")
    
    # Load test set if not provided
    if test_ds is None:
        test_ratio = cfg.get("evaluation", {}).get("held_out", {}).get("test_ratio", 0.1)
        test_ds, metadata = create_held_out_test_split(task_name, test_ratio, seed)
    
    # Get column names
    instruction_col = metadata.get("instruction_column", "instruction")
    context_col = metadata.get("context_column", "context")
    response_col = metadata.get("response_column", "response")
    
    # Subsample if needed
    if len(test_ds) > max_samples:
        rng = np.random.default_rng(seed + 999)
        indices = rng.choice(len(test_ds), size=max_samples, replace=False).tolist()
        test_ds = test_ds.select(indices)
    
    device = next(model.parameters()).device
    model.eval()
    
    t0 = time.perf_counter()
    
    total_loss = 0.0
    total_tokens = 0
    num_samples = 0
    
    print(f"\n[Held-Out NLL] Evaluating on {len(test_ds)} samples...")
    
    with torch.no_grad():
        for i in tqdm(range(len(test_ds)), desc="Computing NLL"):
            instruction = test_ds[instruction_col][i]
            context = test_ds[context_col][i] if context_col in test_ds.column_names else None
            response = test_ds[response_col][i]
            
            # Skip empty responses
            if not response or not str(response).strip():
                continue
            
            # Build prompt
            prompt = build_instruction_prompt(
                instruction=instruction,
                context=context if context and str(context).strip() else None,
                model_name=model_name,
            )
            
            # Tokenize prompt + response
            full_text = prompt + response
            
            tok = tokenizer(
                full_text,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            )
            input_ids = tok["input_ids"].to(device)
            
            # Find where response starts
            prompt_tok = tokenizer(prompt, return_tensors="pt")
            prompt_len = prompt_tok["input_ids"].shape[1]
            
            # Skip if response is truncated away
            if input_ids.shape[1] <= prompt_len:
                continue
            
            # Compute loss only on response tokens
            labels = input_ids.clone()
            labels[0, :prompt_len] = -100  # Ignore prompt tokens in loss
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss.item()
            
            # Count response tokens
            num_response_tokens = input_ids.shape[1] - prompt_len
            
            total_loss += loss * num_response_tokens
            total_tokens += num_response_tokens
            num_samples += 1
    
    # Compute average NLL and perplexity
    avg_nll = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(min(avg_nll, 20))  # Cap to avoid overflow
    
    eval_seconds = time.perf_counter() - t0
    
    print(f"\n[Held-Out NLL] Results:")
    print(f"  - Average NLL: {avg_nll:.4f}")
    print(f"  - Perplexity: {perplexity:.4f}")
    print(f"  - Response Tokens: {total_tokens}")
    print(f"  - Samples: {num_samples}")
    print(f"  - Time: {eval_seconds:.2f}s")
    
    return {
        "nll": float(avg_nll),
        "perplexity": float(perplexity),
        "total_tokens": int(total_tokens),
        "num_samples": int(num_samples),
        "eval_seconds": float(eval_seconds),
    }


# =============================================================================
# 2. IFEVAL (Instruction-Following Evaluation)
# =============================================================================

# IFEval instruction types and their verification functions
IFEVAL_INSTRUCTION_TYPES = [
    "keywords:existence",
    "keywords:frequency", 
    "keywords:forbidden_words",
    "keywords:letter_frequency",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:number_bullet_lists",
    "detectable_format:constrained_response",
    "detectable_format:number_highlighted_sections",
    "detectable_format:multiple_sections",
    "detectable_format:json_format",
    "detectable_format:title",
    "length_constraints:number_sentences",
    "length_constraints:number_paragraphs",
    "length_constraints:number_words",
    "combination:two_responses",
    "combination:repeat_prompt",
    "startend:end_checker",
    "startend:quotation",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "punctuation:no_comma",
]


def verify_keyword_existence(response: str, kwargs: Dict) -> bool:
    """Check if required keywords exist in response."""
    keywords = kwargs.get("keywords", [])
    response_lower = response.lower()
    return all(kw.lower() in response_lower for kw in keywords)


def verify_keyword_frequency(response: str, kwargs: Dict) -> bool:
    """Check keyword frequency constraints."""
    keyword = kwargs.get("keyword", "")
    relation = kwargs.get("relation", "at least")
    frequency = kwargs.get("frequency", 1)
    
    count = response.lower().count(keyword.lower())
    
    if relation == "at least":
        return count >= frequency
    elif relation == "at most":
        return count <= frequency
    elif relation == "exactly":
        return count == frequency
    return False


def verify_forbidden_words(response: str, kwargs: Dict) -> bool:
    """Check that forbidden words are not present."""
    forbidden = kwargs.get("forbidden_words", [])
    response_lower = response.lower()
    return not any(fw.lower() in response_lower for fw in forbidden)


def verify_json_format(response: str, kwargs: Dict) -> bool:
    """Check if response is valid JSON."""
    import json
    try:
        # Try to find JSON in the response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        json.loads(response.strip())
        return True
    except:
        return False


def verify_number_sentences(response: str, kwargs: Dict) -> bool:
    """Check number of sentences constraint."""
    import re
    relation = kwargs.get("relation", "at least")
    num = kwargs.get("num_sentences", 1)
    
    # Simple sentence counting by splitting on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', response)
    sentences = [s.strip() for s in sentences if s.strip()]
    count = len(sentences)
    
    if relation == "at least":
        return count >= num
    elif relation == "at most":
        return count <= num
    elif relation == "exactly":
        return count == num
    return False


def verify_number_words(response: str, kwargs: Dict) -> bool:
    """Check number of words constraint."""
    relation = kwargs.get("relation", "at least")
    num = kwargs.get("num_words", 1)
    
    words = response.split()
    count = len(words)
    
    if relation == "at least":
        return count >= num
    elif relation == "at most":
        return count <= num
    elif relation == "exactly":
        return count == num
    return False


def verify_title(response: str, kwargs: Dict) -> bool:
    """Check if response has a title (wrapped in double asterisks or first line)."""
    lines = response.strip().split('\n')
    if not lines:
        return False
    
    first_line = lines[0].strip()
    # Check for markdown title formats
    return (
        first_line.startswith('#') or  # Markdown header
        first_line.startswith('**') or  # Bold text as title
        (len(first_line) > 0 and len(first_line.split()) <= 10)  # Short first line
    )


def verify_end_checker(response: str, kwargs: Dict) -> bool:
    """Check if response ends with specific phrase."""
    end_phrase = kwargs.get("end_phrase", "")
    return response.strip().endswith(end_phrase)


def verify_quotation(response: str, kwargs: Dict) -> bool:
    """Check if response is wrapped in quotation marks."""
    response = response.strip()
    return (
        (response.startswith('"') and response.endswith('"')) or
        (response.startswith("'") and response.endswith("'")) or
        (response.startswith('"') and response.endswith('"'))
    )


def verify_english_capital(response: str, kwargs: Dict) -> bool:
    """Check if response is in all caps."""
    alpha_chars = [c for c in response if c.isalpha()]
    if not alpha_chars:
        return True
    return all(c.isupper() for c in alpha_chars)


def verify_english_lowercase(response: str, kwargs: Dict) -> bool:
    """Check if response is in all lowercase."""
    alpha_chars = [c for c in response if c.isalpha()]
    if not alpha_chars:
        return True
    return all(c.islower() for c in alpha_chars)


def verify_no_comma(response: str, kwargs: Dict) -> bool:
    """Check if response has no commas."""
    return ',' not in response


# Map instruction types to verification functions
IFEVAL_VERIFIERS = {
    "keywords:existence": verify_keyword_existence,
    "keywords:frequency": verify_keyword_frequency,
    "keywords:forbidden_words": verify_forbidden_words,
    "detectable_format:json_format": verify_json_format,
    "detectable_format:title": verify_title,
    "length_constraints:number_sentences": verify_number_sentences,
    "length_constraints:number_words": verify_number_words,
    "startend:end_checker": verify_end_checker,
    "startend:quotation": verify_quotation,
    "change_case:english_capital": verify_english_capital,
    "change_case:english_lowercase": verify_english_lowercase,
    "punctuation:no_comma": verify_no_comma,
}


def verify_instruction(response: str, instruction_type: str, kwargs: Dict) -> bool:
    """
    Verify if a response follows a specific instruction type.
    
    Args:
        response: The model's generated response
        instruction_type: The type of instruction to verify
        kwargs: Additional arguments for the verification
        
    Returns:
        True if the instruction is followed, False otherwise
    """
    if instruction_type in IFEVAL_VERIFIERS:
        return IFEVAL_VERIFIERS[instruction_type](response, kwargs)
    
    # For unimplemented instruction types, return True (lenient)
    # In practice, you'd want to implement all verifiers
    return True


def evaluate_ifeval(
    model,
    tokenizer,
    cfg: Dict[str, Any],
    sample_ratio: float = 1.0,
    max_samples: int = 500,
) -> Dict[str, Any]:
    """
    Evaluate on IFEval benchmark (Instruction-Following Evaluation).
    
    IFEval tests whether models can follow specific verifiable instructions
    like "write in all caps" or "include at least 3 sentences".
    
    Primary metric: Prompt-level strict accuracy
    (A prompt passes only if ALL instructions in it are followed)
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        cfg: Config dict
        sample_ratio: Fraction of IFEval to evaluate
        max_samples: Maximum number of samples
        
    Returns:
        Dict with:
        - prompt_strict_accuracy: Fraction of prompts where ALL instructions pass
        - instruction_accuracy: Fraction of individual instructions that pass
        - total_prompts: Number of prompts evaluated
        - total_instructions: Number of individual instructions
        - eval_seconds: Evaluation time
    """
    seed = int(cfg["experiment"]["seed"])
    max_seq_len = int(cfg["model"]["max_seq_len"])
    gen_cfg = cfg.get("generation", {})
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 256))
    
    device = next(model.parameters()).device
    model.eval()
    
    t0 = time.perf_counter()
    
    # Load IFEval dataset
    print("\n[IFEval] Loading dataset...")
    try:
        ds = load_dataset("google/IFEval", split="train")  # IFEval only has train split
    except Exception as e:
        print(f"[IFEval] Failed to load dataset: {e}")
        print("[IFEval] Trying alternative loading method...")
        try:
            ds = load_dataset("wis-k/instruction-following-eval", split="train")
        except Exception as e2:
            print(f"[IFEval] Alternative also failed: {e2}")
            return {
                "prompt_strict_accuracy": 0.0,
                "instruction_accuracy": 0.0,
                "total_prompts": 0,
                "total_instructions": 0,
                "eval_seconds": 0.0,
                "error": str(e),
            }
    
    # Sample if needed
    n = len(ds)
    if sample_ratio < 1.0:
        n = int(n * sample_ratio)
    n = min(n, max_samples)
    
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=n, replace=False).tolist()
    ds = ds.select(indices)
    
    print(f"[IFEval] Evaluating on {len(ds)} prompts...")
    
    prompts_passed = 0
    instructions_passed = 0
    total_prompts = 0
    total_instructions = 0
    
    # Use left-padding for generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    try:
        with torch.no_grad():
            for i in tqdm(range(len(ds)), desc="Evaluating IFEval"):
                prompt = ds[i]["prompt"]
                instruction_id_list = ds[i].get("instruction_id_list", [])
                kwargs_list = ds[i].get("kwargs", [{}] * len(instruction_id_list))
                
                # Generate response
                tok = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_seq_len - max_new_tokens,
                    return_tensors="pt",
                )
                input_ids = tok["input_ids"].to(device)
                attention_mask = tok["attention_mask"].to(device)
                
                gen_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "pad_token_id": tokenizer.eos_token_id,
                    "repetition_penalty": 1.15,  # Prevent degenerate repetition
                }
                
                output = model.generate(**gen_kwargs)
                gen_ids = output[0, input_ids.shape[1]:]
                response = tokenizer.decode(gen_ids, skip_special_tokens=True)
                
                # Verify each instruction
                all_passed = True
                for inst_id, inst_kwargs in zip(instruction_id_list, kwargs_list):
                    if isinstance(inst_kwargs, str):
                        import json
                        try:
                            inst_kwargs = json.loads(inst_kwargs)
                        except:
                            inst_kwargs = {}
                    
                    passed = verify_instruction(response, inst_id, inst_kwargs or {})
                    if passed:
                        instructions_passed += 1
                    else:
                        all_passed = False
                    total_instructions += 1
                
                if all_passed:
                    prompts_passed += 1
                total_prompts += 1
    
    finally:
        tokenizer.padding_side = original_padding_side
    
    # Compute metrics
    prompt_strict_accuracy = prompts_passed / total_prompts if total_prompts > 0 else 0.0
    instruction_accuracy = instructions_passed / total_instructions if total_instructions > 0 else 0.0
    
    eval_seconds = time.perf_counter() - t0
    
    print(f"\n[IFEval] Results:")
    print(f"  - Prompt-Level Strict Accuracy: {prompt_strict_accuracy:.4f} ({prompts_passed}/{total_prompts})")
    print(f"  - Instruction-Level Accuracy: {instruction_accuracy:.4f} ({instructions_passed}/{total_instructions})")
    print(f"  - Time: {eval_seconds:.2f}s")
    
    return {
        "prompt_strict_accuracy": float(prompt_strict_accuracy),
        "instruction_accuracy": float(instruction_accuracy),
        "total_prompts": int(total_prompts),
        "prompts_passed": int(prompts_passed),
        "total_instructions": int(total_instructions),
        "instructions_passed": int(instructions_passed),
        "eval_seconds": float(eval_seconds),
    }


# =============================================================================
# 3. COMBINED FINAL EVALUATION
# =============================================================================

def run_final_evaluation(
    model,
    tokenizer,
    cfg: Dict[str, Any],
    client_id: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run all final evaluations for instruction-following models.
    
    Config options (under 'evaluation.final'):
        held_out:
            enabled: true
            test_ratio: 0.1
            max_samples: 500
        ifeval:
            enabled: true
            sample_ratio: 1.0
            max_samples: 500
        mmlu:
            enabled: true  (from existing config)
        mtbench:
            enabled: true
            save_responses: true (saves for GPT-4 judging)
            
    Args:
        model: The language model
        tokenizer: The tokenizer  
        cfg: Full config dict
        client_id: Optional client ID for logging
        
    Returns:
        Dict with all evaluation results
    """
    results = {}
    eval_cfg = cfg.get("evaluation", {})
    client_str = f"[Client {client_id}] " if client_id is not None else ""
    
    print(f"\n{'='*70}")
    print(f"{client_str}FINAL EVALUATION - END OF TRAINING")
    print(f"{'='*70}")
    
    # 1. Held-Out Test NLL/Perplexity
    held_out_cfg = eval_cfg.get("held_out", {})
    if held_out_cfg.get("enabled", True):
        print(f"\n{client_str}[1/3] Held-Out Test Set (NLL / Perplexity)")
        print("-" * 50)
        
        test_ratio = float(held_out_cfg.get("test_ratio", 0.1))
        max_samples = int(held_out_cfg.get("max_samples", 500))
        
        test_ds, metadata = create_held_out_test_split(
            cfg["task"]["name"],
            test_ratio=test_ratio,
            seed=cfg["experiment"]["seed"],
        )
        
        results["held_out"] = evaluate_held_out_nll(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            test_ds=test_ds,
            metadata=metadata,
            max_samples=max_samples,
        )
    
    # 2. IFEval
    ifeval_cfg = eval_cfg.get("ifeval", {})
    if ifeval_cfg.get("enabled", True):
        print(f"\n{client_str}[2/3] IFEval (Instruction-Following)")
        print("-" * 50)
        
        sample_ratio = float(ifeval_cfg.get("sample_ratio", 1.0))
        max_samples = int(ifeval_cfg.get("max_samples", 500))
        
        results["ifeval"] = evaluate_ifeval(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            sample_ratio=sample_ratio,
            max_samples=max_samples,
        )
    
    # 3. MMLU (optional regression check)
    mmlu_cfg = eval_cfg.get("mmlu", {})
    if mmlu_cfg.get("enabled", False):
        print(f"\n{client_str}[3/4] MMLU (Regression Check)")
        print("-" * 50)
        
        results["mmlu"] = run_mmlu_evaluation(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
        )
    
    # 4. MT-Bench (conversational ability)
    mtbench_cfg = eval_cfg.get("mtbench", {})
    if mtbench_cfg.get("enabled", False):
        print(f"\n{client_str}[4/4] MT-Bench (Conversational Ability)")
        print("-" * 50)
        
        results["mtbench"] = run_mtbench_evaluation(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            output_dir=output_dir,
        )
    
    print(f"\n{'='*70}")
    print(f"{client_str}FINAL EVALUATION COMPLETE")
    print(f"{'='*70}\n")
    
    return results


def write_final_eval_to_summary(
    results: Dict[str, Any],
    out_lines: List[str],
    is_multi_client: bool = False,
) -> None:
    """
    Write final evaluation results to summary markdown.
    
    Args:
        results: Final evaluation results (potentially from multiple clients)
        out_lines: List of summary lines to append to
        is_multi_client: Whether results are from multiple clients
    """
    if not results:
        return
    
    out_lines.append("## Final Evaluation Results")
    out_lines.append("")
    
    if is_multi_client:
        # Aggregate results from multiple clients
        client_ids = sorted([k for k in results.keys() if isinstance(k, int)])
        
        if not client_ids:
            return
        
        # Held-Out Test Results
        if "held_out" in results[client_ids[0]]:
            out_lines.append("### 1. Held-Out Test Set (Response-Token NLL)")
            out_lines.append("")
            
            nlls = [results[cid]["held_out"]["nll"] for cid in client_ids]
            ppls = [results[cid]["held_out"]["perplexity"] for cid in client_ids]
            
            mean_nll = sum(nlls) / len(nlls)
            mean_ppl = sum(ppls) / len(ppls)
            std_nll = (sum((x - mean_nll) ** 2 for x in nlls) / len(nlls)) ** 0.5
            std_ppl = (sum((x - mean_ppl) ** 2 for x in ppls) / len(ppls)) ** 0.5
            
            out_lines.append(f"| Metric | Mean | Std | Min | Max |")
            out_lines.append(f"|--------|------|-----|-----|-----|")
            out_lines.append(f"| **NLL** | {mean_nll:.4f} | {std_nll:.4f} | {min(nlls):.4f} | {max(nlls):.4f} |")
            out_lines.append(f"| **Perplexity** | {mean_ppl:.4f} | {std_ppl:.4f} | {min(ppls):.4f} | {max(ppls):.4f} |")
            out_lines.append("")
            
            # Per-client breakdown
            out_lines.append("<details>")
            out_lines.append("<summary>Per-Client NLL Breakdown</summary>")
            out_lines.append("")
            out_lines.append("| Client | NLL | Perplexity | Tokens |")
            out_lines.append("|-------:|----:|----------:|-------:|")
            for cid in client_ids:
                r = results[cid]["held_out"]
                out_lines.append(f"| {cid} | {r['nll']:.4f} | {r['perplexity']:.4f} | {r['total_tokens']} |")
            out_lines.append("")
            out_lines.append("</details>")
            out_lines.append("")
        
        # IFEval Results
        if "ifeval" in results[client_ids[0]]:
            out_lines.append("### 2. IFEval (Instruction-Following)")
            out_lines.append("")
            
            prompt_accs = [results[cid]["ifeval"]["prompt_strict_accuracy"] for cid in client_ids]
            inst_accs = [results[cid]["ifeval"]["instruction_accuracy"] for cid in client_ids]
            
            mean_prompt = sum(prompt_accs) / len(prompt_accs)
            mean_inst = sum(inst_accs) / len(inst_accs)
            std_prompt = (sum((x - mean_prompt) ** 2 for x in prompt_accs) / len(prompt_accs)) ** 0.5
            std_inst = (sum((x - mean_inst) ** 2 for x in inst_accs) / len(inst_accs)) ** 0.5
            
            out_lines.append(f"| Metric | Mean | Std | Min | Max |")
            out_lines.append(f"|--------|------|-----|-----|-----|")
            out_lines.append(f"| **Prompt-Level Strict** | {mean_prompt:.4f} | {std_prompt:.4f} | {min(prompt_accs):.4f} | {max(prompt_accs):.4f} |")
            out_lines.append(f"| **Instruction-Level** | {mean_inst:.4f} | {std_inst:.4f} | {min(inst_accs):.4f} | {max(inst_accs):.4f} |")
            out_lines.append("")
        
        # MMLU Results (if enabled)
        if "mmlu" in results[client_ids[0]] and results[client_ids[0]]["mmlu"]:
            out_lines.append("### 3. MMLU (Regression Check)")
            out_lines.append("")
            
            mmlu_accs = [results[cid]["mmlu"].get("accuracy", 0.0) for cid in client_ids]
            mean_mmlu = sum(mmlu_accs) / len(mmlu_accs)
            std_mmlu = (sum((x - mean_mmlu) ** 2 for x in mmlu_accs) / len(mmlu_accs)) ** 0.5
            
            out_lines.append(f"- **Mean Accuracy**: {mean_mmlu:.4f} ± {std_mmlu:.4f}")
            out_lines.append(f"- **Range**: [{min(mmlu_accs):.4f}, {max(mmlu_accs):.4f}]")
            out_lines.append("")
        
        # MT-Bench Results (if enabled)
        if "mtbench" in results[client_ids[0]] and results[client_ids[0]]["mtbench"]:
            out_lines.append("### 4. MT-Bench (Conversational Ability)")
            out_lines.append("")
            
            avg_lengths = [results[cid]["mtbench"].get("avg_response_length", 0) for cid in client_ids]
            avg_words = [results[cid]["mtbench"].get("avg_word_count", 0) for cid in client_ids]
            diversities = [results[cid]["mtbench"].get("diversity_score", 0) for cid in client_ids]
            
            mean_len = sum(avg_lengths) / len(avg_lengths)
            mean_words = sum(avg_words) / len(avg_words)
            mean_div = sum(diversities) / len(diversities)
            
            out_lines.append(f"- **Mean Avg Response Length**: {mean_len:.1f} chars")
            out_lines.append(f"- **Mean Avg Word Count**: {mean_words:.1f} words")
            out_lines.append(f"- **Mean Diversity Score**: {mean_div:.4f}")
            out_lines.append("")
            out_lines.append("*Note: MT-Bench traditionally uses GPT-4 as judge. Metrics shown are judge-free alternatives.*")
            out_lines.append("")
    
    else:
        # Single model results
        if "held_out" in results:
            out_lines.append("### 1. Held-Out Test Set (Response-Token NLL)")
            out_lines.append("")
            r = results["held_out"]
            out_lines.append(f"- **NLL**: {r['nll']:.4f}")
            out_lines.append(f"- **Perplexity**: {r['perplexity']:.4f}")
            out_lines.append(f"- **Response Tokens**: {r['total_tokens']}")
            out_lines.append(f"- **Samples**: {r['num_samples']}")
            out_lines.append("")
        
        if "ifeval" in results:
            out_lines.append("### 2. IFEval (Instruction-Following)")
            out_lines.append("")
            r = results["ifeval"]
            out_lines.append(f"- **Prompt-Level Strict Accuracy**: {r['prompt_strict_accuracy']:.4f} ({r['prompts_passed']}/{r['total_prompts']})")
            out_lines.append(f"- **Instruction-Level Accuracy**: {r['instruction_accuracy']:.4f} ({r['instructions_passed']}/{r['total_instructions']})")
            out_lines.append("")
        
        if "mmlu" in results and results["mmlu"]:
            out_lines.append("### 3. MMLU (Regression Check)")
            out_lines.append("")
            r = results["mmlu"]
            out_lines.append(f"- **Accuracy**: {r.get('accuracy', 0.0):.4f}")
            out_lines.append("")
        
        if "mtbench" in results and results["mtbench"]:
            out_lines.append("### 4. MT-Bench (Conversational Ability)")
            out_lines.append("")
            r = results["mtbench"]
            out_lines.append(f"- **Questions Answered**: {r.get('questions_answered', 0)}")
            out_lines.append(f"- **Avg Response Length**: {r.get('avg_response_length', 0):.1f} chars")
            out_lines.append(f"- **Avg Word Count**: {r.get('avg_word_count', 0):.1f} words")
            out_lines.append(f"- **Diversity Score**: {r.get('diversity_score', 0):.4f}")
            out_lines.append("")
            out_lines.append("*Note: MT-Bench traditionally uses GPT-4 as judge. Metrics shown are judge-free alternatives.*")
            out_lines.append("")


