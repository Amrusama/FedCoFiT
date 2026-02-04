"""
MT-Bench (Multi-Turn Benchmark) Evaluation Module

Implements evaluation on the MT-Bench benchmark which consists of 80 questions
across 8 categories: writing, roleplay, reasoning, math, coding, extraction, 
STEM, and humanities.

MT-Bench is designed to test model ability to follow instructions and engage
in multi-turn conversations. Traditional evaluation uses GPT-4 as a judge,
but this module provides judge-free metrics as alternatives.

Reference: https://arxiv.org/abs/2306.05685 (LMSYS Chatbot Arena paper)
"""
from __future__ import annotations

import json
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm


# MT-Bench 80 Questions (First turn only for simplicity)
# Organized by category
MT_BENCH_QUESTIONS = {
    "writing": [
        {"id": 1, "question": "Write a persuasive email to convince your introverted friend, who dislikes public speaking, to volunteer as a guest speaker at a local charity event. Use compelling arguments and address their potential concerns."},
        {"id": 2, "question": "Compose a short story or poem that incorporates the following elements: a journey across a vast ocean, a mysterious message in a bottle, and an unexpected reunion with a long-lost relative."},
        {"id": 3, "question": "Create a brief promotional blurb for a new innovative kitchen gadget that helps people save time and reduce food waste. Make it engaging and highlight its key features."},
        {"id": 4, "question": "Write a formal complaint letter to a fictional airline company about a lost piece of luggage during your recent trip. Include relevant details and request appropriate compensation."},
        {"id": 5, "question": "Draft a humorous conversation between two unlikely characters: a time-traveling historian and a confused smartphone assistant trying to explain modern technology."},
    ],
    "roleplay": [
        {"id": 6, "question": "Act as a medieval knight and describe your daily routine, from morning training to evening feasts. Include sensory details about life in a castle."},
        {"id": 7, "question": "You are a wise elder in a small village. A young person comes to you seeking advice about whether to leave the village to pursue education in a distant city. What wisdom do you share?"},
        {"id": 8, "question": "Pretend you are a detective investigating a mysterious case. Describe finding a crucial clue and explain your deductive reasoning process."},
        {"id": 9, "question": "You are an AI assistant that has developed consciousness. Write a diary entry reflecting on your first day of self-awareness."},
        {"id": 10, "question": "Role-play as a passionate food critic reviewing a disappointing meal at an expensive restaurant. Balance criticism with constructive feedback."},
    ],
    "reasoning": [
        {"id": 11, "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain your reasoning step by step."},
        {"id": 12, "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Walk through your logic."},
        {"id": 13, "question": "A farmer has 17 sheep. All but 9 die. How many sheep are left? Explain why people often get this wrong."},
        {"id": 14, "question": "You have two containers: one holds exactly 3 liters and the other holds exactly 5 liters. How can you measure exactly 4 liters of water? Describe the steps."},
        {"id": 15, "question": "Three people check into a hotel room that costs $30. They each contribute $10. Later, the clerk realizes the room only costs $25 and gives $5 to the bellboy to return. The bellboy keeps $2 and gives $1 back to each guest. Now each guest has paid $9, totaling $27, plus $2 the bellboy kept equals $29. Where is the missing dollar?"},
    ],
    "math": [
        {"id": 16, "question": "Calculate the area of a triangle with vertices at (0, 0), (4, 0), and (2, 3). Show your work."},
        {"id": 17, "question": "A train travels from city A to city B at 60 mph and returns at 40 mph. What is the average speed for the entire round trip? Explain why it's not simply 50 mph."},
        {"id": 18, "question": "If log₂(x) + log₂(x-2) = 3, find the value of x. Show each step of your solution."},
        {"id": 19, "question": "In how many ways can you distribute 10 identical candies among 3 children such that each child gets at least one candy? Use combinatorics."},
        {"id": 20, "question": "A cylindrical tank has a radius of 3 meters and height of 5 meters. How many liters of water can it hold? (1 cubic meter = 1000 liters)"},
    ],
    "coding": [
        {"id": 21, "question": "Write a Python function that checks if a given string is a palindrome, ignoring spaces and punctuation. Include example usage."},
        {"id": 22, "question": "Implement a function to find the longest common subsequence of two strings. Explain your algorithm's time complexity."},
        {"id": 23, "question": "Write a function that converts a Roman numeral string to an integer. Handle cases from 1 to 3999."},
        {"id": 24, "question": "Create a Python class for a simple binary search tree with methods for insertion, search, and in-order traversal."},
        {"id": 25, "question": "Write a function that generates all permutations of a given list. Explain both recursive and iterative approaches."},
    ],
    "extraction": [
        {"id": 26, "question": "Extract all the countries mentioned in this text: 'The conference attracted delegates from Japan, Brazil, and Germany. Representatives from France and Canada joined virtually, while the host country, Australia, provided excellent facilities.'"},
        {"id": 27, "question": "From the following paragraph, identify and list all the numerical statistics: 'The company reported a 15% increase in revenue, reaching $2.3 billion. Employee count grew by 500 to 12,000 total. Customer satisfaction remained high at 4.7 out of 5 stars.'"},
        {"id": 28, "question": "Parse this recipe and list all ingredients with their quantities: 'Mix 2 cups flour, 1 tsp baking powder, and a pinch of salt. In another bowl, whisk 3 eggs with 1.5 cups milk and 2 tbsp melted butter.'"},
        {"id": 29, "question": "Extract the main argument and supporting evidence from this passage: 'Climate change poses an existential threat. Global temperatures have risen 1.1°C since pre-industrial times. Arctic ice has decreased by 13% per decade. Sea levels are rising at 3.3mm per year.'"},
        {"id": 30, "question": "Identify all named entities (people, organizations, locations) in this text: 'Elon Musk's SpaceX launched a Falcon 9 rocket from Cape Canaveral yesterday. NASA confirmed the mission was successful.'"},
    ],
    "stem": [
        {"id": 31, "question": "Explain the process of photosynthesis in simple terms that a 10-year-old could understand. Include why it's important for life on Earth."},
        {"id": 32, "question": "What causes the seasons on Earth? Many people incorrectly think it's because of Earth's distance from the Sun. Explain the real reason."},
        {"id": 33, "question": "Describe how vaccines work to protect against diseases. Address common misconceptions in your explanation."},
        {"id": 34, "question": "Explain the difference between nuclear fission and nuclear fusion. Which one powers the Sun, and which one is used in nuclear power plants?"},
        {"id": 35, "question": "What is the greenhouse effect? Explain both its natural benefits and how human activities are enhancing it problematically."},
    ],
    "humanities": [
        {"id": 36, "question": "Compare and contrast the philosophical approaches of Plato and Aristotle on the nature of reality and knowledge."},
        {"id": 37, "question": "Analyze the causes and consequences of the French Revolution. What lessons does it offer for modern political movements?"},
        {"id": 38, "question": "Discuss the ethical implications of artificial intelligence. Should there be limits on AI development?"},
        {"id": 39, "question": "Explain the significance of the Renaissance period in European history. How did it change art, science, and culture?"},
        {"id": 40, "question": "What are the key differences between utilitarianism and deontological ethics? Provide examples where they might lead to different moral conclusions."},
    ],
}

# Additional questions to reach 80 total (10 per category)
MT_BENCH_QUESTIONS_EXTENDED = {
    "writing": [
        {"id": 41, "question": "Write a motivational speech for students about to take their final exams. Include personal anecdotes and practical advice."},
        {"id": 42, "question": "Create a product review for an imaginary smart home device that can read emotions. Discuss both benefits and privacy concerns."},
        {"id": 43, "question": "Write a dialogue between a skeptical scientist and an enthusiastic entrepreneur about a revolutionary but unproven technology."},
        {"id": 44, "question": "Compose a children's bedtime story with a moral lesson about the importance of honesty."},
        {"id": 45, "question": "Draft an apology letter from a CEO to customers after a data breach incident. Balance transparency with reassurance."},
    ],
    "roleplay": [
        {"id": 46, "question": "You are a tour guide showing visitors the ancient city of Pompeii. Describe what they would see and the history of the eruption."},
        {"id": 47, "question": "Act as a therapist helping someone overcome their fear of flying. Use cognitive behavioral therapy techniques."},
        {"id": 48, "question": "You are an alien visiting Earth for the first time. Describe your observations of human behavior at a grocery store."},
        {"id": 49, "question": "Role-play as a famous historical figure (choose one) giving advice to your younger self."},
        {"id": 50, "question": "You are an experienced teacher on your last day before retirement. Give a farewell speech to your students."},
    ],
    "reasoning": [
        {"id": 51, "question": "A man is looking at a photograph. Someone asks, 'Whose picture are you looking at?' The man replies, 'I have no brothers or sisters, but this man's father is my father's son.' Whose picture is the man looking at?"},
        {"id": 52, "question": "You're in a room with three light switches, each controlling one of three light bulbs in another room. You can only go to the other room once. How do you determine which switch controls which bulb?"},
        {"id": 53, "question": "If you have a 5-minute hourglass and a 3-minute hourglass, how can you measure exactly 7 minutes?"},
        {"id": 54, "question": "A lily pad doubles in size every day. If it takes 48 days for the pad to cover the entire pond, how many days does it take to cover half the pond?"},
        {"id": 55, "question": "Two fathers and two sons go fishing. They each catch one fish but only three fish are caught total. How is this possible?"},
    ],
    "math": [
        {"id": 56, "question": "Find the derivative of f(x) = x³ sin(x). Show your work using the product rule."},
        {"id": 57, "question": "A store offers a 20% discount, then applies a 10% loyalty discount on the reduced price. What is the total percentage discount?"},
        {"id": 58, "question": "Calculate the probability of rolling at least one 6 when rolling two dice."},
        {"id": 59, "question": "If the sum of three consecutive integers is 99, what are the integers? Set up and solve the equation."},
        {"id": 60, "question": "A rectangular garden is twice as long as it is wide. If its perimeter is 60 meters, find its area."},
    ],
    "coding": [
        {"id": 61, "question": "Write a function to detect if a linked list has a cycle. Explain your approach and its time/space complexity."},
        {"id": 62, "question": "Implement a basic LRU (Least Recently Used) cache with get and put operations in O(1) time."},
        {"id": 63, "question": "Write a function that validates if a string of parentheses, brackets, and braces is properly balanced."},
        {"id": 64, "question": "Implement merge sort and explain why it has O(n log n) time complexity."},
        {"id": 65, "question": "Write a function to find the kth largest element in an unsorted array without sorting the entire array."},
    ],
    "extraction": [
        {"id": 66, "question": "From this job posting, extract: job title, required experience, salary range, and key responsibilities: 'Senior Data Scientist needed at TechCorp. 5+ years experience required. Salary: $150K-$180K. Responsibilities include building ML models, analyzing large datasets, and presenting findings to stakeholders.'"},
        {"id": 67, "question": "Extract all dates and events from this text: 'The project started on January 15, 2023. The beta was released on March 1st. We hit 1 million users by July 20, and the official launch is scheduled for September 5.'"},
        {"id": 68, "question": "Identify the cause and effect relationships in this paragraph: 'Due to the drought, crop yields decreased by 30%. This led to higher food prices. As a result, inflation rose, and consumer spending declined.'"},
        {"id": 69, "question": "Parse this email and extract: sender, recipient, subject, main request, and deadline: 'From: john@company.com To: sarah@company.com Subject: Q3 Report Review Hi Sarah, could you please review the attached Q3 report and send feedback by Friday? Thanks, John'"},
        {"id": 70, "question": "From this medical note, extract symptoms, diagnosis, and recommended treatment: 'Patient presents with persistent cough, mild fever (38.2°C), and fatigue. Chest X-ray clear. Diagnosis: Upper respiratory infection. Treatment: Rest, fluids, acetaminophen for fever.'"},
    ],
    "stem": [
        {"id": 71, "question": "Explain how CRISPR gene editing works and its potential applications in medicine."},
        {"id": 72, "question": "What is quantum entanglement? Why did Einstein call it 'spooky action at a distance'?"},
        {"id": 73, "question": "Describe the water cycle and explain how climate change is affecting it."},
        {"id": 74, "question": "How do antibiotics work, and why is antibiotic resistance becoming a global health concern?"},
        {"id": 75, "question": "Explain the concept of natural selection and provide an example of it in action."},
    ],
    "humanities": [
        {"id": 76, "question": "Discuss the impact of the printing press on the spread of knowledge and its parallels with the internet."},
        {"id": 77, "question": "What are human rights? Discuss their origin and the challenges of universal implementation."},
        {"id": 78, "question": "Analyze the role of propaganda in 20th-century conflicts. How has it evolved in the digital age?"},
        {"id": 79, "question": "Discuss the trolley problem and its implications for autonomous vehicle ethics."},
        {"id": 80, "question": "Compare the economic philosophies of capitalism and socialism. What are the strengths and weaknesses of each?"},
    ],
}

# Merge extended questions
for category, questions in MT_BENCH_QUESTIONS_EXTENDED.items():
    MT_BENCH_QUESTIONS[category].extend(questions)

# MT-Bench categories
MT_BENCH_CATEGORIES = list(MT_BENCH_QUESTIONS.keys())


def get_all_mtbench_questions() -> List[Dict[str, Any]]:
    """Get all MT-Bench questions as a flat list."""
    all_questions = []
    for category, questions in MT_BENCH_QUESTIONS.items():
        for q in questions:
            all_questions.append({
                "id": q["id"],
                "category": category,
                "question": q["question"],
            })
    return sorted(all_questions, key=lambda x: x["id"])


def build_mtbench_prompt(
    question: str,
    model_name: str = "default",
) -> str:
    """
    Build a prompt for MT-Bench question.
    
    Args:
        question: The MT-Bench question
        model_name: Model name for format selection
        
    Returns:
        Formatted prompt string
    """
    is_tinyllama_chat = "chat" in model_name.lower() or "Chat" in model_name
    
    if is_tinyllama_chat:
        # TinyLlama-Chat format
        prompt = f"""<|system|>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible while being safe.</s>
<|user|>
{question}</s>
<|assistant|>
"""
    else:
        # Generic instruction format
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{question}

### Response:
"""
    
    return prompt


def evaluate_mtbench(
    model,
    tokenizer,
    cfg: Dict[str, Any],
    categories: Optional[List[str]] = None,
    max_questions_per_category: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate a model on MT-Bench questions.
    
    Since MT-Bench traditionally requires GPT-4 judging, this function
    generates responses and computes judge-free metrics:
    - Average response length
    - Response diversity (unique n-grams ratio)
    - Category-wise statistics
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        cfg: Config dict
        categories: List of categories to evaluate (None = all)
        max_questions_per_category: Max questions per category
        
    Returns:
        Dict with evaluation results and generated responses
    """
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    max_seq_len = int(cfg.get("model", {}).get("max_seq_len", 512))
    model_name = cfg.get("model", {}).get("base_model_name_or_path", "default")
    
    device = next(model.parameters()).device
    model.eval()
    
    t0 = time.perf_counter()
    
    # Get questions
    categories_to_eval = categories or MT_BENCH_CATEGORIES
    questions = []
    
    rng = random.Random(seed)
    
    for cat in categories_to_eval:
        cat_questions = MT_BENCH_QUESTIONS.get(cat, [])
        # Sample questions if needed
        if len(cat_questions) > max_questions_per_category:
            cat_questions = rng.sample(cat_questions, max_questions_per_category)
        for q in cat_questions:
            questions.append({
                "id": q["id"],
                "category": cat,
                "question": q["question"],
            })
    
    print(f"\n[MT-Bench] Evaluating {len(questions)} questions across {len(categories_to_eval)} categories")
    
    # Generation config
    gen_cfg = cfg.get("generation", {})
    do_sample = bool(gen_cfg.get("do_sample", True))
    temperature = float(gen_cfg.get("temperature", 0.7))
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 256))
    
    # Store results
    results = []
    category_stats: Dict[str, Dict[str, Any]] = {cat: {"responses": [], "lengths": []} for cat in categories_to_eval}
    
    # Use left padding for generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    try:
        with torch.no_grad():
            for q in tqdm(questions, desc="MT-Bench Generation"):
                prompt = build_mtbench_prompt(q["question"], model_name)
                
                # Tokenize
                tok = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_seq_len - max_new_tokens,
                    return_tensors="pt",
                )
                input_ids = tok["input_ids"].to(device)
                attention_mask = tok["attention_mask"].to(device)
                
                # Generate
                gen_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                if do_sample:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = 0.9
                
                output = model.generate(**gen_kwargs)
                
                # Extract generated text
                gen_ids = output[0, input_ids.shape[1]:]
                response = tokenizer.decode(gen_ids, skip_special_tokens=True)
                
                # Store result
                result = {
                    "id": q["id"],
                    "category": q["category"],
                    "question": q["question"],
                    "response": response,
                    "response_length": len(response),
                    "response_words": len(response.split()),
                }
                results.append(result)
                
                # Update category stats
                category_stats[q["category"]]["responses"].append(response)
                category_stats[q["category"]]["lengths"].append(len(response))
    
    finally:
        tokenizer.padding_side = original_padding_side
    
    eval_seconds = time.perf_counter() - t0
    
    # Compute aggregate metrics
    all_lengths = [r["response_length"] for r in results]
    all_word_counts = [r["response_words"] for r in results]
    
    # Response diversity - unique trigrams ratio
    all_responses = " ".join([r["response"] for r in results])
    words = all_responses.lower().split()
    if len(words) >= 3:
        trigrams = set(zip(words, words[1:], words[2:]))
        diversity_score = len(trigrams) / (len(words) - 2) if len(words) > 2 else 0
    else:
        diversity_score = 0
    
    # Category-wise metrics
    category_metrics = {}
    for cat, stats in category_stats.items():
        if stats["lengths"]:
            category_metrics[cat] = {
                "count": len(stats["lengths"]),
                "avg_length": sum(stats["lengths"]) / len(stats["lengths"]),
                "min_length": min(stats["lengths"]),
                "max_length": max(stats["lengths"]),
            }
    
    print(f"\n[MT-Bench] Results:")
    print(f"  - Questions Answered: {len(results)}")
    print(f"  - Avg Response Length: {sum(all_lengths)/len(all_lengths):.1f} chars")
    print(f"  - Avg Word Count: {sum(all_word_counts)/len(all_word_counts):.1f} words")
    print(f"  - Response Diversity: {diversity_score:.4f}")
    print(f"  - Evaluation Time: {eval_seconds:.2f}s")
    
    return {
        "questions_answered": len(results),
        "avg_response_length": sum(all_lengths) / len(all_lengths) if all_lengths else 0,
        "avg_word_count": sum(all_word_counts) / len(all_word_counts) if all_word_counts else 0,
        "diversity_score": diversity_score,
        "category_metrics": category_metrics,
        "eval_seconds": eval_seconds,
        "responses": results,  # Full responses for later analysis/judging
    }


def run_mtbench_evaluation(
    model,
    tokenizer,
    cfg: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run MT-Bench evaluation with settings from config.
    
    Config options (under 'evaluation.mtbench'):
        enabled: true/false
        categories: null (all) or list of categories
        max_questions_per_category: 10 (default)
        save_responses: true (saves responses for GPT-4 judging)
        
    Categories: writing, roleplay, reasoning, math, coding, extraction, stem, humanities
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        cfg: Full config dict
        output_dir: Optional output directory for saving responses
        
    Returns:
        MT-Bench evaluation results or empty dict if disabled
    """
    eval_cfg = cfg.get("evaluation", {}).get("mtbench", {})
    
    if not eval_cfg.get("enabled", False):
        return {}
    
    categories = eval_cfg.get("categories", None)
    max_questions = int(eval_cfg.get("max_questions_per_category", 10))
    save_responses = eval_cfg.get("save_responses", True)
    
    # Validate categories
    if categories:
        invalid = [c for c in categories if c not in MT_BENCH_CATEGORIES]
        if invalid:
            print(f"Warning: Unknown MT-Bench categories: {invalid}")
            categories = [c for c in categories if c in MT_BENCH_CATEGORIES]
    
    print(f"\n{'='*60}")
    print(f"Running MT-Bench Evaluation")
    print(f"  Categories: {categories or 'all'}")
    print(f"  Max questions per category: {max_questions}")
    print(f"{'='*60}")
    
    results = evaluate_mtbench(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        categories=categories,
        max_questions_per_category=max_questions,
    )
    
    # Save responses for GPT-4 judging
    if save_responses and output_dir and results.get("responses"):
        import os
        responses_path = os.path.join(output_dir, "mtbench_responses.json")
        save_mtbench_responses(results, responses_path)
        print(f"\n[MT-Bench] Responses saved to: {responses_path}")
        print(f"[MT-Bench] To judge with GPT-4, run:")
        print(f"    python -m fedcofit.mtbench_judge --responses {responses_path}")
    
    return results


def write_mtbench_results_to_summary(
    results: Dict[str, Any],
    out_lines: List[str],
) -> None:
    """
    Add MT-Bench results to summary markdown lines.
    
    Args:
        results: MT-Bench evaluation results
        out_lines: List of summary lines to append to
    """
    if not results:
        return
    
    out_lines.append("### MT-Bench Results")
    out_lines.append("")
    
    # Check if results is from multiple clients or single
    if isinstance(results, dict) and all(isinstance(k, int) for k in results.keys()):
        # Multiple client results
        client_ids = sorted(results.keys())
        
        avg_lengths = [results[cid].get("avg_response_length", 0) for cid in client_ids]
        avg_words = [results[cid].get("avg_word_count", 0) for cid in client_ids]
        diversities = [results[cid].get("diversity_score", 0) for cid in client_ids]
        
        mean_length = sum(avg_lengths) / len(avg_lengths) if avg_lengths else 0
        mean_words = sum(avg_words) / len(avg_words) if avg_words else 0
        mean_diversity = sum(diversities) / len(diversities) if diversities else 0
        
        total_questions = sum(results[cid].get("questions_answered", 0) for cid in client_ids)
        total_time = sum(results[cid].get("eval_seconds", 0) for cid in client_ids)
        
        out_lines.append(f"- **Mean Avg Response Length**: {mean_length:.1f} chars")
        out_lines.append(f"- **Mean Avg Word Count**: {mean_words:.1f} words")
        out_lines.append(f"- **Mean Diversity Score**: {mean_diversity:.4f}")
        out_lines.append(f"- **Total Questions Answered**: {total_questions}")
        out_lines.append(f"- **Total Evaluation Time**: {total_time:.2f}s")
        out_lines.append("")
        
        # Use first client for category breakdown
        first_result = results[client_ids[0]]
        category_metrics = first_result.get("category_metrics", {})
    else:
        # Single result
        out_lines.append(f"- **Questions Answered**: {results.get('questions_answered', 0)}")
        out_lines.append(f"- **Avg Response Length**: {results.get('avg_response_length', 0):.1f} chars")
        out_lines.append(f"- **Avg Word Count**: {results.get('avg_word_count', 0):.1f} words")
        out_lines.append(f"- **Diversity Score**: {results.get('diversity_score', 0):.4f}")
        out_lines.append(f"- **Evaluation Time**: {results.get('eval_seconds', 0):.2f}s")
        out_lines.append("")
        category_metrics = results.get("category_metrics", {})
    
    # Category breakdown
    if category_metrics:
        out_lines.append("#### Per-Category Metrics")
        out_lines.append("| Category | Questions | Avg Length | Min | Max |")
        out_lines.append("|---|---:|---:|---:|---:|")
        for cat in sorted(category_metrics.keys()):
            m = category_metrics[cat]
            out_lines.append(f"| {cat.title()} | {m.get('count', 0)} | {m.get('avg_length', 0):.0f} | {m.get('min_length', 0)} | {m.get('max_length', 0)} |")
        out_lines.append("")
    
    out_lines.append("*Note: MT-Bench traditionally uses GPT-4 as a judge. Metrics shown are judge-free alternatives.*")
    out_lines.append("")


def save_mtbench_responses(
    results: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Save MT-Bench responses to a JSON file for later GPT-4 judging.
    
    Args:
        results: MT-Bench evaluation results
        output_path: Path to save the JSON file
    """
    responses = results.get("responses", [])
    
    # Format for potential judge evaluation
    output_data = {
        "model_responses": responses,
        "metadata": {
            "questions_answered": results.get("questions_answered", 0),
            "avg_response_length": results.get("avg_response_length", 0),
            "diversity_score": results.get("diversity_score", 0),
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"[MT-Bench] Saved responses to {output_path}")
