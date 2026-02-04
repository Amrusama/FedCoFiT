"""
MMLU (Massive Multitask Language Understanding) Evaluation Module

Implements 5-shot prompting evaluation on the MMLU benchmark following the
FLoRA paper methodology.

MMLU consists of 57 subjects across STEM, humanities, social sciences, and other domains.
Evaluation uses accuracy on multiple-choice questions (A/B/C/D).

Reference: https://arxiv.org/abs/2009.03300
"""
from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


# MMLU subject categories - All 57 subjects
MMLU_SUBJECTS = [
    # STEM
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management",
    "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations", "security_studies",
    "sociology", "us_foreign_policy", "virology", "world_religions"
]

# =============================================================================
# SMART SUBJECT PRESETS - Optimized for different model sizes/capabilities
# =============================================================================

# Subjects where small models (1-3B params) tend to perform better:
# - High school level (more accessible knowledge)
# - General knowledge and reasoning
# - Common sense and logic
# - Avoids highly specialized professional domains
MMLU_SUBJECTS_SMALL_MODEL_FRIENDLY = [
    # General Knowledge & Reasoning (small models do reasonably well)
    "miscellaneous",           # General trivia - broad but accessible
    "global_facts",            # World facts - common knowledge
    "logical_fallacies",       # Logic/reasoning - less memorization
    "world_religions",         # Cultural knowledge - broad exposure in pretraining
    
    # High School Level (simpler than college)
    "high_school_geography",   # Basic geography
    "high_school_psychology",  # Intro psychology
    "high_school_world_history",  # General history
    "high_school_us_history",  # US history
    "high_school_government_and_politics",  # Civics
    "high_school_macroeconomics",  # Basic econ
    "high_school_microeconomics",  # Basic econ
    "elementary_mathematics",  # Simple math
    "high_school_statistics",  # Basic stats
    
    # Social Sciences (less technical than STEM)
    "sociology",               # Social patterns
    "public_relations",        # Communication
    "marketing",               # Business basics
    "management",              # Business basics
    "human_sexuality",         # Health education
    "human_aging",             # Health education
    "nutrition",               # Common health knowledge
    
    # Philosophy & Ethics (reasoning-based)
    "philosophy",              # Conceptual thinking
    "moral_disputes",          # Ethical reasoning
    "business_ethics",         # Applied ethics
    
    # Conceptual Science (reasoning > memorization)
    "conceptual_physics",      # Physics concepts without heavy math
    "astronomy",               # Space facts - interesting to models
    "prehistory",              # Ancient history
]

# Subjects that require specialized professional knowledge - avoid for small models
MMLU_SUBJECTS_PROFESSIONAL = [
    "professional_law",
    "professional_medicine", 
    "professional_accounting",
    "professional_psychology",
    "clinical_knowledge",
    "medical_genetics",
    "virology",
    "college_medicine",
    "anatomy",
]

# STEM subjects that require strong technical knowledge
MMLU_SUBJECTS_HARD_STEM = [
    "abstract_algebra",
    "college_mathematics",
    "college_physics", 
    "college_chemistry",
    "college_biology",
    "college_computer_science",
    "electrical_engineering",
    "econometrics",
    "formal_logic",
    "machine_learning",
]

# Subject presets mapping
MMLU_SUBJECT_PRESETS = {
    "all": MMLU_SUBJECTS,
    "small_model_friendly": MMLU_SUBJECTS_SMALL_MODEL_FRIENDLY,
    "easy": MMLU_SUBJECTS_SMALL_MODEL_FRIENDLY,  # Alias
}

# Choice letters
CHOICES = ["A", "B", "C", "D"]


def format_subject(subject: str) -> str:
    """Convert subject code to human-readable format."""
    return subject.replace("_", " ").title()


def build_mmlu_prompt(
    question: str,
    choices: List[str],
    subject: str,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    model_name: str = "default",
) -> str:
    """
    Build a 5-shot MMLU prompt.
    
    Args:
        question: The question text
        choices: List of 4 answer choices
        subject: The MMLU subject (e.g., "abstract_algebra")
        few_shot_examples: List of few-shot examples (question, choices, answer)
        model_name: Model name for format selection
        
    Returns:
        Formatted prompt string
    """
    subject_formatted = format_subject(subject)
    
    # Check if this is a chat model that needs special format
    is_tinyllama_chat = "chat" in model_name.lower() or "Chat" in model_name
    is_llama_chat = "llama" in model_name.lower() and "chat" in model_name.lower()
    
    if is_tinyllama_chat:
        # TinyLlama-Chat: Use a very direct prompt that forces single-letter output
        # The key is to make it absolutely clear we want ONLY a letter
        
        # Build few-shot examples in a compact format
        examples = []
        if few_shot_examples:
            # Only use 3 examples to keep prompt short
            for ex in few_shot_examples[:3]:
                ex_q = ex["question"][:100]  # Truncate long questions
                ex_choices = ex["choices"]
                ex_answer = CHOICES[ex["answer"]]
                choice_str = " ".join([f"({CHOICES[i]}) {c[:30]}" for i, c in enumerate(ex_choices)])
                examples.append(f"{ex_q}\n{choice_str}\nAnswer: {ex_answer}")
        
        # Target question
        choice_str = " ".join([f"({CHOICES[i]}) {c}" for i, c in enumerate(choices)])
        
        # Very direct prompt format
        prompt = f"""<|system|>
You are taking a multiple choice test. For each question, respond with ONLY the letter of the correct answer (A, B, C, or D). Do not explain.</s>
<|user|>
Subject: {subject_formatted}

"""
        if examples:
            prompt += "\n\n".join(examples) + "\n\n"
        
        prompt += f"""{question}
{choice_str}
Answer:</s>
<|assistant|>
"""
        return prompt
    
    # Default format (for base models or other chat models)
    prompt_parts = []
    
    # Header
    prompt_parts.append(f"The following are multiple choice questions (with answers) about {subject_formatted}.\n")
    
    # Few-shot examples
    if few_shot_examples:
        for ex in few_shot_examples:
            ex_question = ex["question"]
            ex_choices = ex["choices"]
            ex_answer_idx = ex["answer"]
            ex_answer = CHOICES[ex_answer_idx]
            
            prompt_parts.append(f"Question: {ex_question}")
            for i, choice in enumerate(ex_choices):
                prompt_parts.append(f"{CHOICES[i]}. {choice}")
            prompt_parts.append(f"Answer: {ex_answer}\n")
    
    # Target question
    prompt_parts.append(f"Question: {question}")
    for i, choice in enumerate(choices):
        prompt_parts.append(f"{CHOICES[i]}. {choice}")
    prompt_parts.append("Answer:")
    
    return "\n".join(prompt_parts)


def extract_answer(generated_text: str) -> Optional[str]:
    """
    Extract the answer letter (A/B/C/D) from generated text.
    
    Args:
        generated_text: The model's generated output
        
    Returns:
        The answer letter or None if not found
    """
    import re
    
    if not generated_text:
        return None
    
    # Clean up
    text = generated_text.strip()
    text_upper = text.upper()
    
    # Check first character (most common case for MMLU)
    if text and text[0].upper() in CHOICES:
        # Make sure it's not just a word starting with A/B/C/D (like "And", "But", "Can", "Do")
        if len(text) == 1 or not text[1].isalpha():
            return text[0].upper()
    
    # Patterns to match - ordered by likelihood
    patterns = [
        # Direct answer patterns
        r'^([A-D])[.\):\s]',           # A. or A) or A: or A (space)
        r'^\(([A-D])\)',               # (A)
        r'^([A-D])\s*$',               # Just A/B/C/D
        
        # "The answer is X" patterns
        r'(?:the\s+)?(?:correct\s+)?answer\s+is\s+([A-D])\b',  # "The correct answer is A"
        r'(?:the\s+)?(?:correct\s+)?answer[:\s]+([A-D])\b',    # "Answer: A" or "answer A"
        r'\bis\s+([A-D])\b',           # "is A" (for "The answer is A")
        r'\bwould\s+be\s+([A-D])\b',   # "would be A"
        
        # Option reference patterns  
        r'option\s+([A-D])\b',         # "option A"
        r'choice\s+([A-D])\b',         # "choice A"
        r'\(([A-D])\)\s+is\s+correct', # "(A) is correct"
        
        # Parenthetical answer
        r'\(([A-D])\)',                # (A) anywhere
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_upper, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Last resort: find ANY isolated A-D in the text
    # Look for A, B, C, D that are surrounded by non-letters
    for match in re.finditer(r'(?<![A-Za-z])([A-D])(?![A-Za-z])', text_upper):
        return match.group(1)
    
    return None


def load_mmlu_dataset(
    split: str = "test",
    subjects: Optional[List[str]] = None,
    sample_ratio: float = 1.0,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    Load MMLU dataset from HuggingFace.
    
    Args:
        split: "test", "validation", or "dev"
        subjects: List of subjects to include (None = all)
        sample_ratio: Fraction of samples to use (0.0-1.0)
        seed: Random seed for sampling
        
    Returns:
        Tuple of (test_samples, dev_samples_by_subject)
        - test_samples: List of test examples
        - dev_samples_by_subject: Dict mapping subject to list of dev examples (for few-shot)
    """
    subjects_to_load = subjects or MMLU_SUBJECTS
    
    test_samples = []
    dev_samples_by_subject: Dict[str, List[Dict[str, Any]]] = {}
    
    rng = random.Random(seed)
    
    for subject in tqdm(subjects_to_load, desc="Loading MMLU subjects"):
        try:
            # Load test split
            ds = load_dataset("cais/mmlu", subject, split=split, trust_remote_code=True)
            
            # Convert to list of dicts
            subject_samples = []
            for i in range(len(ds)):
                sample = {
                    "question": ds[i]["question"],
                    "choices": ds[i]["choices"],
                    "answer": int(ds[i]["answer"]),
                    "subject": subject,
                }
                subject_samples.append(sample)
            
            # Apply sampling ratio
            if sample_ratio < 1.0:
                n_samples = max(1, int(len(subject_samples) * sample_ratio))
                subject_samples = rng.sample(subject_samples, n_samples)
            
            test_samples.extend(subject_samples)
            
            # Load dev split for few-shot examples
            try:
                dev_ds = load_dataset("cais/mmlu", subject, split="dev", trust_remote_code=True)
                dev_samples_by_subject[subject] = [
                    {
                        "question": dev_ds[i]["question"],
                        "choices": dev_ds[i]["choices"],
                        "answer": int(dev_ds[i]["answer"]),
                        "subject": subject,
                    }
                    for i in range(len(dev_ds))
                ]
            except Exception:
                # Some subjects might not have dev split
                dev_samples_by_subject[subject] = []
                
        except Exception as e:
            print(f"Warning: Failed to load subject '{subject}': {e}")
            continue
    
    return test_samples, dev_samples_by_subject


def get_few_shot_examples(
    subject: str,
    dev_samples_by_subject: Dict[str, List[Dict[str, Any]]],
    num_shots: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Get few-shot examples for a subject.
    
    Args:
        subject: The MMLU subject
        dev_samples_by_subject: Dict mapping subject to dev examples
        num_shots: Number of few-shot examples (default: 5)
        seed: Random seed for selection
        
    Returns:
        List of few-shot examples
    """
    dev_samples = dev_samples_by_subject.get(subject, [])
    
    if not dev_samples:
        return []
    
    rng = random.Random(seed)
    n_examples = min(num_shots, len(dev_samples))
    return rng.sample(dev_samples, n_examples)


def evaluate_mmlu(
    model,
    tokenizer,
    cfg: Dict[str, Any],
    sample_ratio: float = 0.1,
    num_shots: int = 5,
    batch_size: int = 1,
    subjects: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a model on MMLU benchmark.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        cfg: Config dict (for generation settings)
        sample_ratio: Fraction of MMLU to evaluate (0.0-1.0)
        num_shots: Number of few-shot examples (default: 5)
        batch_size: Batch size for generation
        subjects: List of subjects to evaluate (None = all)
        
    Returns:
        Dict with evaluation results:
        - accuracy: Overall accuracy
        - per_subject_accuracy: Dict mapping subject to accuracy
        - total_samples: Total number of samples
        - correct: Number of correct predictions
        - eval_seconds: Evaluation time in seconds
    """
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    max_seq_len = int(cfg.get("model", {}).get("max_seq_len", 512))
    model_name = cfg.get("model", {}).get("base_model_name_or_path", "default")
    
    device = next(model.parameters()).device
    model.eval()
    
    t0 = time.perf_counter()
    
    # Load dataset
    print(f"\n[MMLU] Loading dataset with sample_ratio={sample_ratio}...")
    test_samples, dev_samples_by_subject = load_mmlu_dataset(
        split="test",
        subjects=subjects,
        sample_ratio=sample_ratio,
        seed=seed,
    )
    
    if not test_samples:
        print("[MMLU] No samples loaded, returning empty results.")
        return {
            "accuracy": 0.0,
            "per_subject_accuracy": {},
            "total_samples": 0,
            "correct": 0,
            "eval_seconds": time.perf_counter() - t0,
        }
    
    print(f"[MMLU] Loaded {len(test_samples)} test samples from {len(dev_samples_by_subject)} subjects")
    
    # Evaluate
    correct = 0
    total = 0
    per_subject_correct: Dict[str, int] = {}
    per_subject_total: Dict[str, int] = {}
    
    # Set up generation config
    gen_cfg = cfg.get("generation", {})
    do_sample = bool(gen_cfg.get("do_sample", False))
    
    # Use left-padding for generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    # Debug: track generation patterns
    debug_samples = []
    none_count = 0
    
    try:
        with torch.no_grad():
            for sample in tqdm(test_samples, desc="Evaluating MMLU"):
                subject = sample["subject"]
                
                # Get few-shot examples
                few_shot = get_few_shot_examples(
                    subject, 
                    dev_samples_by_subject,
                    num_shots=num_shots,
                    seed=seed,
                )
                
                # Build prompt
                prompt = build_mmlu_prompt(
                    question=sample["question"],
                    choices=sample["choices"],
                    subject=subject,
                    few_shot_examples=few_shot,
                    model_name=model_name,
                )
                
                # Tokenize
                tok = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_seq_len - 10,  # Leave room for generation
                    return_tensors="pt",
                )
                input_ids = tok["input_ids"].to(device)
                attention_mask = tok["attention_mask"].to(device)
                
                # Generate - need enough tokens for "The correct answer is A" style responses
                gen_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": 15,  # Increased from 3 to handle verbose responses
                    "do_sample": do_sample,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                if do_sample:
                    gen_kwargs["temperature"] = float(gen_cfg.get("temperature", 0.0))
                
                output = model.generate(**gen_kwargs)
                
                # Extract generated tokens
                gen_ids = output[0, input_ids.shape[1]:]
                generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                
                # Extract answer
                pred = extract_answer(generated_text)
                gold = CHOICES[sample["answer"]]
                
                # Debug: collect first 20 samples for analysis
                if len(debug_samples) < 20:
                    debug_samples.append({
                        "generated": generated_text[:50],  # Show more characters
                        "pred": pred,
                        "gold": gold,
                        "correct": pred == gold,
                    })
                
                if pred is None:
                    none_count += 1
                
                # Update counts
                if pred == gold:
                    correct += 1
                    per_subject_correct[subject] = per_subject_correct.get(subject, 0) + 1
                
                total += 1
                per_subject_total[subject] = per_subject_total.get(subject, 0) + 1
        
        # Print debug info
        print(f"\n[MMLU DEBUG] Answer extraction failed (None) for {none_count}/{total} samples ({100*none_count/total:.1f}%)")
        print(f"[MMLU DEBUG] First 20 generations:")
        for i, d in enumerate(debug_samples):
            status = "✓" if d["correct"] else "✗"
            gen_display = d['generated'].replace('\n', ' ')[:50]  # Show 50 chars, remove newlines
            print(f"  {i+1}. [{status}] gen='{gen_display}' → pred={d['pred']}, gold={d['gold']}")
    
    finally:
        # Restore padding side
        tokenizer.padding_side = original_padding_side
    
    # Compute per-subject accuracy
    per_subject_accuracy = {}
    for subject in per_subject_total:
        subj_correct = per_subject_correct.get(subject, 0)
        subj_total = per_subject_total[subject]
        per_subject_accuracy[subject] = float(subj_correct) / float(subj_total) if subj_total > 0 else 0.0
    
    # Overall accuracy
    accuracy = float(correct) / float(total) if total > 0 else 0.0
    
    eval_seconds = time.perf_counter() - t0
    
    print(f"\n[MMLU] Results:")
    print(f"  - Overall Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"  - Evaluation Time: {eval_seconds:.2f}s")
    
    return {
        "accuracy": accuracy,
        "per_subject_accuracy": per_subject_accuracy,
        "total_samples": total,
        "correct": correct,
        "eval_seconds": eval_seconds,
    }


def run_mmlu_evaluation(
    model,
    tokenizer,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run MMLU evaluation with settings from config.
    
    Config options (under 'evaluation.mmlu'):
        enabled: true/false
        sample_ratio: 0.0-1.0 (default: 0.1 = 10%)
        num_shots: 5 (default)
        subjects: null (all), preset name ("small_model_friendly", "easy"), or list of subjects
        
    Subject presets:
        - "all": All 57 MMLU subjects (hardest, requires broad knowledge)
        - "small_model_friendly" / "easy": 25 subjects optimized for small models
          Focuses on: general knowledge, high school level, reasoning-based questions
          Avoids: professional domains, college-level STEM, medical/legal expertise
        
    Args:
        model: The language model
        tokenizer: The tokenizer
        cfg: Full config dict
        
    Returns:
        MMLU evaluation results or empty dict if disabled
    """
    eval_cfg = cfg.get("evaluation", {}).get("mmlu", {})
    
    if not eval_cfg.get("enabled", False):
        return {}
    
    sample_ratio = float(eval_cfg.get("sample_ratio", 0.1))
    num_shots = int(eval_cfg.get("num_shots", 5))
    subjects_cfg = eval_cfg.get("subjects", None)
    
    # Resolve subjects - support presets or explicit list
    if subjects_cfg is None:
        subjects = None  # Use all subjects
        subjects_desc = "all (57 subjects)"
    elif isinstance(subjects_cfg, str):
        # Check if it's a preset name
        if subjects_cfg in MMLU_SUBJECT_PRESETS:
            subjects = MMLU_SUBJECT_PRESETS[subjects_cfg]
            subjects_desc = f"preset '{subjects_cfg}' ({len(subjects)} subjects)"
        else:
            raise ValueError(f"Unknown MMLU subject preset: {subjects_cfg}. "
                           f"Available presets: {list(MMLU_SUBJECT_PRESETS.keys())}")
    elif isinstance(subjects_cfg, list):
        subjects = subjects_cfg
        subjects_desc = f"custom list ({len(subjects)} subjects)"
    else:
        raise ValueError(f"Invalid subjects config type: {type(subjects_cfg)}")
    
    print(f"\n{'='*60}")
    print(f"Running MMLU Evaluation ({num_shots}-shot)")
    print(f"  Sample ratio: {sample_ratio*100:.1f}%")
    print(f"  Subjects: {subjects_desc}")
    print(f"{'='*60}")
    
    results = evaluate_mmlu(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        sample_ratio=sample_ratio,
        num_shots=num_shots,
        subjects=subjects,
    )
    
    return results


def write_mmlu_results_to_summary(
    results: Dict[str, Any],
    out_lines: List[str],
) -> None:
    """
    Add MMLU results to summary markdown lines.
    
    Args:
        results: MMLU evaluation results - can be:
            - Dict with client_id keys (FedCoFiT: multiple clients)
            - Dict with 'accuracy' key (single result)
        out_lines: List of summary lines to append to
    """
    if not results:
        return
    
    out_lines.append("### MMLU Benchmark Results")
    out_lines.append("")
    
    # Check if results is from multiple clients (FedCoFiT) or single (FLoRA)
    if isinstance(results, dict) and all(isinstance(k, int) for k in results.keys()):
        # Multiple client results (FedCoFiT)
        client_ids = sorted(results.keys())
        accuracies = [results[cid].get("accuracy", 0.0) for cid in client_ids]
        
        mean_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
        min_acc = min(accuracies) if accuracies else 0.0
        max_acc = max(accuracies) if accuracies else 0.0
        
        total_correct = sum(results[cid].get("correct", 0) for cid in client_ids)
        total_samples = sum(results[cid].get("total_samples", 0) for cid in client_ids)
        total_time = sum(results[cid].get("eval_seconds", 0.0) for cid in client_ids)
        
        out_lines.append(f"- **Mean Accuracy**: {mean_acc:.4f}")
        out_lines.append(f"- **Min/Max Accuracy**: {min_acc:.4f} / {max_acc:.4f}")
        out_lines.append(f"- **Clients Evaluated**: {len(client_ids)}")
        out_lines.append(f"- **Total Evaluation Time**: {total_time:.2f}s")
        out_lines.append("")
        
        # Per-client breakdown
        out_lines.append("#### Per-Client MMLU Accuracy")
        out_lines.append("| Client | Accuracy | Correct/Total |")
        out_lines.append("|---:|---:|---:|")
        for cid in client_ids:
            r = results[cid]
            out_lines.append(f"| {cid} | {r.get('accuracy', 0.0):.4f} | {r.get('correct', 0)}/{r.get('total_samples', 0)} |")
        out_lines.append("")
        
        # Use first client's per-subject breakdown for detail
        first_result = results[client_ids[0]]
        per_subj = first_result.get("per_subject_accuracy", {})
    else:
        # Single result (FLoRA or legacy format)
        out_lines.append(f"- **Overall Accuracy**: {results.get('accuracy', 0.0):.4f} ({results.get('correct', 0)}/{results.get('total_samples', 0)})")
        out_lines.append(f"- **Evaluation Time**: {results.get('eval_seconds', 0.0):.2f}s")
        out_lines.append("")
        per_subj = results.get("per_subject_accuracy", {})
    
    # Per-subject breakdown (top 5 best and worst)
    if per_subj:
        sorted_subj = sorted(per_subj.items(), key=lambda x: x[1], reverse=True)
        
        out_lines.append("#### Top 5 Subjects")
        out_lines.append("| Subject | Accuracy |")
        out_lines.append("|---|---:|")
        for subj, acc in sorted_subj[:5]:
            out_lines.append(f"| {format_subject(subj)} | {acc:.4f} |")
        out_lines.append("")
        
        out_lines.append("#### Bottom 5 Subjects")
        out_lines.append("| Subject | Accuracy |")
        out_lines.append("|---|---:|")
        for subj, acc in sorted_subj[-5:]:
            out_lines.append(f"| {format_subject(subj)} | {acc:.4f} |")
        out_lines.append("")

