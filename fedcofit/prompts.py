from __future__ import annotations

import re
from typing import Optional, Tuple, Dict, Any


# ============================================================================
# Classification Task Prompts (SST-2, etc.)
# ============================================================================

CLASSIFICATION_PROMPT_TEMPLATE = """Task: Sentiment classification.
Return ONLY one word: positive or negative.

Text:
{sentence}

Answer:
"""

CLASSIFICATION_STRICT_PROMPT_TEMPLATE = """Task: Sentiment classification.
You MUST output exactly positive or negative.
Return ONLY one word: positive or negative.

Text:
{sentence}

Answer:
"""


# ============================================================================
# Instruction-Following Task Prompts (Dolly-15k, Alpaca, etc.)
# ============================================================================

# Prompt template for instruction-following (used for training)
INSTRUCTION_PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

# Prompt template with context (for instructions that have additional context)
INSTRUCTION_WITH_CONTEXT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{context}

### Response:
"""

# TinyLlama Chat format (matches the model's expected format)
TINYLLAMA_CHAT_TEMPLATE = """<|system|>
You are a helpful assistant.</s>
<|user|>
{instruction}</s>
<|assistant|>
"""

TINYLLAMA_CHAT_WITH_CONTEXT_TEMPLATE = """<|system|>
You are a helpful assistant.</s>
<|user|>
{instruction}

Context: {context}</s>
<|assistant|>
"""


# ============================================================================
# Prompt Builder Functions
# ============================================================================

def build_prompt(sentence: str, strict: bool = False) -> str:
    """Build a classification prompt (backwards compatible)."""
    tpl = CLASSIFICATION_STRICT_PROMPT_TEMPLATE if strict else CLASSIFICATION_PROMPT_TEMPLATE
    return tpl.format(sentence=sentence)


def build_classification_prompt(sentence: str, strict: bool = False) -> str:
    """Build a prompt for classification tasks."""
    tpl = CLASSIFICATION_STRICT_PROMPT_TEMPLATE if strict else CLASSIFICATION_PROMPT_TEMPLATE
    return tpl.format(sentence=sentence)


def build_instruction_prompt(
    instruction: str, 
    context: Optional[str] = None,
    model_name: str = "default",
) -> str:
    """Build a prompt for instruction-following tasks.
    
    Args:
        instruction: The task instruction
        context: Optional additional context/input
        model_name: Model name to select appropriate chat template
        
    Returns:
        Formatted prompt string
    """
    # Use TinyLlama chat format for TinyLlama models
    if "tinyllama" in model_name.lower():
        if context and context.strip():
            return TINYLLAMA_CHAT_WITH_CONTEXT_TEMPLATE.format(
                instruction=instruction, 
                context=context
            )
        else:
            return TINYLLAMA_CHAT_TEMPLATE.format(instruction=instruction)
    else:
        # Default Alpaca-style format
        if context and context.strip():
            return INSTRUCTION_WITH_CONTEXT_TEMPLATE.format(
                instruction=instruction, 
                context=context
            )
        else:
            return INSTRUCTION_PROMPT_TEMPLATE.format(instruction=instruction)


def build_task_prompt(
    task_type: str,
    text: Optional[str] = None,
    instruction: Optional[str] = None,
    context: Optional[str] = None,
    model_name: str = "default",
    strict: bool = False,
) -> str:
    """Universal prompt builder for any task type.
    
    Args:
        task_type: "classification" or "instruction"
        text: Text for classification tasks
        instruction: Instruction for instruction-following tasks
        context: Optional context for instruction-following tasks
        model_name: Model name for template selection
        strict: Use strict prompting for classification
        
    Returns:
        Formatted prompt string
    """
    if task_type == "classification":
        if text is None:
            raise ValueError("Classification task requires 'text' argument")
        return build_classification_prompt(text, strict=strict)
    elif task_type == "instruction":
        if instruction is None:
            raise ValueError("Instruction task requires 'instruction' argument")
        return build_instruction_prompt(instruction, context=context, model_name=model_name)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


# ============================================================================
# Label Normalization (for classification tasks)
# ============================================================================

def normalize_label(text: str) -> Optional[str]:
    """Normalize a classification label to 'positive' or 'negative'."""
    if text is None:
        return None
    # lowercase, strip whitespace/punctuation
    s = text.strip().lower()
    # replace any non-letter chars with spaces, then strip again
    s = re.sub(r"[^a-z\s]+", " ", s).strip()
    if not s:
        return None
    # if output contains more than one word, take the first token
    first = s.split()[0]
    if first in {"positive", "negative"}:
        return first
    return None


def normalize_with_retry(raw: str, retry_raw: Optional[str] = None) -> Tuple[Optional[str], bool]:
    """Normalize with optional retry.
    
    Returns (label_or_none, used_retry)
    """
    lab = normalize_label(raw)
    if lab is not None:
        return lab, False
    if retry_raw is None:
        return None, False
    lab2 = normalize_label(retry_raw)
    if lab2 is not None:
        return lab2, True
    return None, True


# ============================================================================
# Response Processing (for instruction-following tasks)
# ============================================================================

def clean_response(text: str, max_length: int = 512) -> str:
    """Clean and truncate a generated response.
    
    Args:
        text: Raw generated text
        max_length: Maximum character length
        
    Returns:
        Cleaned response string
    """
    if text is None:
        return ""
    
    # Strip whitespace
    text = text.strip()
    
    # Remove any trailing incomplete sentences (optional)
    # For now, just truncate at max_length
    if len(text) > max_length:
        text = text[:max_length]
        # Try to end at a sentence boundary
        last_period = text.rfind(".")
        if last_period > max_length * 0.5:
            text = text[:last_period + 1]
    
    return text


def compute_response_embedding_similarity(
    response1: str, 
    response2: str, 
    embedding_model: Any = None,
) -> float:
    """Compute semantic similarity between two responses.
    
    This is used for instruction-following consensus in FedCoFiT.
    
    Args:
        response1: First response
        response2: Second response
        embedding_model: Sentence embedding model
        
    Returns:
        Cosine similarity score (0-1)
    """
    if embedding_model is None:
        # Fallback to simple string matching
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)
    
    # Use embedding model for semantic similarity
    emb1 = embedding_model.encode(response1)
    emb2 = embedding_model.encode(response2)
    
    # Cosine similarity
    import numpy as np
    dot = np.dot(emb1, emb2)
    norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    if norm == 0:
        return 0.0
    return float(dot / norm)


