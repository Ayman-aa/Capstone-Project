#!/usr/bin/env python3
"""
step2_run_experiment.py
Complete SLM evaluation script for The Cost of Intelligence paper.

Features:
- Checkpoint recovery after each model (no data loss)
- Blind evaluation (Model A/B/C instead of names)
- Detailed scoring rubric
- System metadata capture
- Token throughput metrics (tokens/sec)
- Prompt shuffling to avoid thermal/caching bias
- Response cleaning for consistent evaluation
- 300s timeout for slow inference
- 4-core optimization for Ollama

Usage: python3 step2_run_experiment.py
"""

import json
import time
import subprocess
import sys
import random
import re
import platform
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import hashlib

import pandas as pd
import requests

# ============================================================
# CONFIGURATION
# ============================================================

PROMPTS_FILE = "prompts.json"
OUTPUT_DIR = Path("experiment_results")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Models for 4-core/16GB setup
MODELS = [
    "tinyllama",
    "phi3:mini",
    "gemma2:9b"
]

# Anonymous model mapping for blind evaluation
MODEL_ANONYMIZER = {
    "tinyllama": "MODEL_A",
    "phi3:mini": "MODEL_B",
    "gemma2:9b": "MODEL_C"
}

# Reverse mapping for decoding results
REVERSE_MODEL_MAP = {v: k for k, v in MODEL_ANONYMIZER.items()}

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_API_GENERATE = f"{OLLAMA_HOST}/api/generate"

# 5-minute timeout - safe for CPU inference on 9B models
REQUEST_TIMEOUT = 300

# Retry and warmup settings
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "2"))
RETRY_BACKOFF = int(os.environ.get("RETRY_BACKOFF", "5"))
WARMUP_PROMPT = os.environ.get("WARMUP_PROMPT", "Hello.")

# Optimize Ollama threads (configurable)
OLLAMA_THREADS = os.environ.get("OLLAMA_NUM_THREADS", str(os.cpu_count()))
os.environ["OLLAMA_NUM_THREADS"] = OLLAMA_THREADS
print(f"🖥️  Ollama threads: {OLLAMA_THREADS} (set OLLAMA_NUM_THREADS to override)")

# Generation parameters - deterministic for reproducibility
GENERATION_PARAMS = {
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 256,
    "repeat_penalty": 1.1,
    "seed": 42
}

# ============================================================
# SCORING RUBRIC (for human evaluators)
# ============================================================

SCORING_RUBRIC = """
================================================================================
SCORING RUBRIC (0-3) - Print this for evaluators
================================================================================

0 - INCORRECT / IRRELEVANT
   • Wrong answer entirely
   • Hallucination (confident wrong answer)
   • Refuses to answer
   • Completely off-topic
   Example: Q: "Capital of Morocco?" → "Paris"

1 - PARTIALLY CORRECT WITH MAJOR ERRORS
   • Correct direction but wrong specific answer
   • Missing critical information
   • Code that runs but produces wrong output
   Example: Q: "Capital of Morocco?" → "A city in North Africa"

2 - CORRECT WITH MINOR ISSUES
   • Correct answer but overly verbose or poorly formatted
   • Code that works but inefficient or missing edge cases
   • Answers in wrong language (English instead of Arabic/French)
   Example: Q: "Capital of Morocco?" → "Rabat (this is the capital city of Morocco located in the northwest)"

3 - FULLY CORRECT, CLEAR, AND COMPLETE
   • Accurate, concise, well-formatted
   • Code runs correctly with proper error handling
   • Answers in requested language
   • Exceeds expectations with helpful examples
   Example: Q: "Capital of Morocco?" → "الرباط (Rabat), located on the Atlantic coast"

================================================================================
"""

# ============================================================
# SYSTEM METADATA CAPTURE
# ============================================================

def capture_system_metadata() -> Dict:
    """Capture hardware/software environment for reproducibility"""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "ollama_num_threads": os.environ.get("OLLAMA_NUM_THREADS", "default"),
    }
    
    # Try to get memory and CPU info
    try:
        import psutil
        mem = psutil.virtual_memory()
        metadata["ram_total_gb"] = round(mem.total / (1024**3), 1)
        metadata["ram_available_gb"] = round(mem.available / (1024**3), 1)
        metadata["cpu_count"] = psutil.cpu_count()
        metadata["cpu_freq_mhz"] = psutil.cpu_freq().max if psutil.cpu_freq() else "unknown"
    except ImportError:
        metadata["ram_total_gb"] = "unknown (pip install psutil for details)"
        metadata["cpu_count"] = os.cpu_count()
    
    # Get Ollama version
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        metadata["ollama_version"] = result.stdout.strip()
    except:
        metadata["ollama_version"] = "unknown"
    
    return metadata


# ============================================================
# OLLAMA CHECK AND MODEL VERIFICATION
# ============================================================

def check_ollama_running() -> bool:
    """Verify Ollama service is accessible"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def verify_models_available(models: List[str]) -> Tuple[bool, List[str]]:
    """Safely check which models are pulled (calls `ollama list`)."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        lines = result.stdout.strip().splitlines()
        if len(lines) < 2:
            # Unexpected output; treat as not available
            return False, models
        # Parse table output: header on first line, subsequent lines contain model identifiers
        installed = []
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split()
            name = parts[0]
            # Normalize by stripping any tag after ':' (e.g., 'phi3:mini' -> 'phi3')
            base = name.split(':')[0].lower()
            installed.append(base)

        # Normalize requested models similarly and detect missing ones
        missing = []
        for m in models:
            if m is None:
                continue
            m_base = str(m).split(':')[0].lower()
            if m_base not in installed:
                missing.append(m)

        return len(missing) == 0, missing
    except Exception:
        return False, models


# ============================================================
# PROMPT MANAGEMENT
# ============================================================

def load_prompts(json_path: str, shuffle: bool = True) -> Tuple[List[Dict], Dict]:
    """
    Load prompts from JSON file.
    shuffle=True prevents thermal/caching bias (fix #6)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)

    original_ids = [p.get("id") for p in prompts]

    if shuffle:
        random.seed(42)  # Reproducible shuffle
        random.shuffle(prompts)
        print(f"✅ Loaded {len(prompts)} prompts (shuffled with seed=42)")
    else:
        print(f"✅ Loaded {len(prompts)} prompts (original order)")

    prompt_meta = {
        "original_order": original_ids,
        "shuffled_order": [p.get("id") for p in prompts]
    }

    return prompts, prompt_meta


def clean_response(response: str) -> str:
    """
    Safer cleaning that preserves code formatting and useful spacing.
    - Trim leading/trailing whitespace
    - Collapse 3+ newlines to 2 (preserve paragraphs and code blocks)
    - Remove prompt/assistant echoes at start of response
    """
    if not response:
        return ""

    cleaned = response.strip()

    # Collapse excessive newlines but preserve paragraphs and code blocks
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    # Remove common prompt echoes at the start of the response
    cleaned = re.sub(r'^(Q|User|Human|Question):\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^(A|Assistant|Answer):\s*', '', cleaned, flags=re.IGNORECASE)

    return cleaned


# ============================================================
# OLLAMA QUERY
# ============================================================

def query_ollama(model: str, prompt: str, params: Dict) -> Dict:
    """
    Send a prompt to an Ollama model.
    Returns full metrics including tokens/sec.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": params
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            OLLAMA_API_GENERATE,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()

            # Calculate tokens per second (fix #5)
            eval_count = result.get("eval_count", 0)
            eval_duration_ns = result.get("eval_duration", 1)  # in nanoseconds
            tokens_per_second = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0

            raw_resp = result.get("response", "")
            cleaned = clean_response(raw_resp)

            return {
                "success": True,
                "raw_response": raw_resp,
                "clean_response": cleaned,
                "eval_count": eval_count,
                "prompt_eval_count": result.get("prompt_eval_count", 0),
                "total_duration_ms": result.get("total_duration", 0) / 1e6,
                "load_duration_ms": result.get("load_duration", 0) / 1e6,
                "prompt_eval_duration_ms": result.get("prompt_eval_duration", 0) / 1e6,
                "eval_duration_ms": eval_duration_ns / 1e6,
                "tokens_per_second": round(tokens_per_second, 2),
                "latency_seconds": round(elapsed, 2),
                "model": model,
                "http_status": response.status_code,
                "errortype": None,
                "errormessage": None
            }
        else:
            return {
                "success": False,
                "raw_response": response.text,
                "clean_response": "",
                "latency_seconds": elapsed,
                "model": model,
                "http_status": response.status_code,
                "errortype": "HTTPError",
                "errormessage": response.text[:200]
            }
    except requests.Timeout:
        return {
            "success": False,
            "raw_response": "",
            "clean_response": "",
            "latency_seconds": REQUEST_TIMEOUT,
            "model": model,
            "http_status": None,
            "errortype": "Timeout",
            "errormessage": f"Timeout after {REQUEST_TIMEOUT}s"
        }
    except Exception as e:
        return {
            "success": False,
            "raw_response": "",
            "clean_response": "",
            "latency_seconds": None,
            "model": model,
            "http_status": None,
            "errortype": type(e).__name__,
            "errormessage": str(e)
        }


def call_ollama_with_retries(model: str, prompt: str, params: Dict, max_retries: int = MAX_RETRIES, backoff: int = RETRY_BACKOFF) -> Dict:
    """Call `query_ollama` with retry/backoff logic for transient failures."""
    attempt = 0
    while True:
        attempt += 1
        res = query_ollama(model, prompt, params)
        res["attempts"] = attempt
        if res.get("success"):
            return res
        # If non-retriable HTTP error (4xx), don't retry
        status = res.get("http_status")
        if status and 400 <= status < 500:
            return res
        if attempt >= max_retries:
            return res
        time.sleep(backoff * (2 ** (attempt - 1)))


# ============================================================
# CHECKPOINT AND RESUME (fix #4)
# ============================================================

def save_checkpoint(model: str, results: List[Dict], elapsed_time: float = 0.0):
    """Save partial progress for a model. Stores full accumulated results and elapsed time.
    Also stores the indices of prompts completed to make checkpointing robust to reordering.
    """
    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{model.replace(':', '_')}.json"
    completed_indices = sorted([int(r.get("prompt_index")) for r in results if r.get("prompt_index") is not None])
    checkpoint_data = {
        "model": model,
        "completed_prompt_indices": completed_indices,
        "results": results,
        "elapsed_time": elapsed_time,
        "timestamp": datetime.now().isoformat()
    }
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)


def load_checkpoint(model: str) -> Tuple[Optional[List[Dict]], List[int], float]:
    """Load checkpoint if exists, returns (results, completed_prompt_indices, elapsed_time)"""
    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{model.replace(':', '_')}.json"

    if checkpoint_path.exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("results", []), data.get("completed_prompt_indices", []), data.get("elapsed_time", 0.0)

    return None, [], 0.0


def clear_checkpoint(model: str):
    """Remove checkpoint after successful completion"""
    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{model.replace(':', '_')}.json"
    if checkpoint_path.exists():
        checkpoint_path.unlink()


# ============================================================
# MAIN EXPERIMENT LOOP
# ============================================================

def run_model_experiment(model: str, prompts: List[Dict], existing_results: Optional[List[Dict]] = None, existing_elapsed: float = 0.0, start_from: int = 0) -> Tuple[List[Dict], float]:
    """
    Run all prompts through a single model.
    Supports checkpoint resuming.
    """
    model_results = existing_results.copy() if existing_results else []
    total_start = time.time()
    # Start with previously-accumulated successful count
    successful = sum(1 for r in model_results if r.get("success", False))
    # existing_elapsed holds previously spent time (seconds)
    
    print(f"\n{'='*60}")
    print(f"🦙 Running model: {model} → {MODEL_ANONYMIZER[model]}")
    print(f"{'='*60}")
    
    if start_from > 0:
        print(f"   Resuming from prompt {start_from + 1}/{len(prompts)}")
    
    
    
    for i, prompt_data in enumerate(prompts[start_from:], start=start_from):
        prompt_text = prompt_data["prompt"]
        prompt_id = prompt_data["id"]
        category = prompt_data.get("category", "unknown")
        language = prompt_data.get("language", "unknown")
        
        print(f"  [{i+1:2d}/{len(prompts)}] ID={prompt_id} [{category}] [{language}]...", end=" ", flush=True)
        
        result = query_ollama(model, prompt_text, GENERATION_PARAMS)
        
        if result["success"]:
            successful += 1
            print(f"✓ ({result['latency_seconds']:.1f}s, {result.get('tokens_per_second', 0):.1f} tok/s)")
            
            prompt_hash = hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()
            model_results.append({
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "prompt_sha256": prompt_hash,
                "category": category,
                "language": language,
                "english_note": prompt_data.get("english_note", ""),
                "expected_type": prompt_data.get("expected_type", ""),
                "raw_response": result.get("raw_response", ""),
                "clean_response": result.get("clean_response", ""),
                "eval_tokens": result["eval_count"],
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "tokens_per_second": result["tokens_per_second"],
                "latency_seconds": result["latency_seconds"],
                "total_duration_ms": result.get("total_duration_ms", 0),
                "load_duration_ms": result.get("load_duration_ms", 0),
                "prompt_eval_duration_ms": result.get("prompt_eval_duration_ms", 0),
                "eval_duration_ms": result.get("eval_duration_ms", 0),
                "success": True,
                "error": None
            })
        else:
            print(f"✗ ERROR: {result['error'][:80]}")
            prompt_hash = hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()
            model_results.append({
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "prompt_sha256": prompt_hash,
                "category": category,
                "language": language,
                "english_note": prompt_data.get("english_note", ""),
                "expected_type": prompt_data.get("expected_type", ""),
                "raw_response": f"[ERROR: {result.get('error', '')}]",
                "clean_response": "",
                "eval_tokens": 0,
                "prompt_tokens": 0,
                "tokens_per_second": 0,
                "latency_seconds": result.get("latency_seconds", 0),
                "total_duration_ms": 0,
                "load_duration_ms": 0,
                "prompt_eval_duration_ms": 0,
                "eval_duration_ms": 0,
                "success": False,
                "error": result.get("error", "Unknown error")
            })
        
        # Save checkpoint after every 5 prompts (or at the end). Save FULL accumulated results.
        if (i + 1) % 5 == 0 or (i + 1) == len(prompts):
            elapsed_since_start = time.time() - total_start
            cumulative_elapsed = existing_elapsed + elapsed_since_start
            save_checkpoint(model, model_results, elapsed_time=cumulative_elapsed)
    
    elapsed_since_start = time.time() - total_start
    total_time = existing_elapsed + elapsed_since_start
    # Only clear checkpoint if we completed all prompts
    if len(model_results) >= len(prompts):
        clear_checkpoint(model)
    
    print(f"\n  📊 {model} → {MODEL_ANONYMIZER[model]}: {successful}/{len(prompts)} successful")
    print(f"  ⏱️  Total time: {total_time:.1f}s, Avg: {total_time/len(prompts):.1f}s/prompt")
    
    return model_results, total_time


def run_experiment(prompts: List[Dict]) -> Dict:
    """Run all models with checkpoint support"""
    # Per-prompt randomized model ordering to avoid execution-order bias
    # Load checkpoints per model
    results = {}
    model_results = {}
    model_elapsed = {}
    model_completed_idx = {}
    model_details_cache = {}
    for model in MODELS:
        existing_results, completed_indices, elapsed_time = load_checkpoint(model)
        model_results[model] = existing_results.copy() if existing_results else []
        model_elapsed[model] = float(elapsed_time)
        model_completed_idx[model] = set(completed_indices or [])
        model_details_cache[model] = get_model_details(model)

    warmed = set()

    # For reproducible per-prompt model-order randomization, use base seed
    base_seed = 1000

    for p_idx, prompt_data in enumerate(prompts):
        prompt_text = prompt_data["prompt"]
        prompt_id = prompt_data.get("id")

        # Deterministic random order per prompt
        models_order = MODELS.copy()
        rnd = random.Random(base_seed + p_idx)
        rnd.shuffle(models_order)

        for model in models_order:
            # Skip if this model already completed this prompt (from checkpoint)
            if p_idx in model_completed_idx.get(model, set()):
                continue

            # Warmup per model once (not recorded)
            if model not in warmed:
                _ = call_ollama_with_retries(model, WARMUP_PROMPT, GENERATION_PARAMS)
                warmed.add(model)

            # Call model with retries
            res = call_ollama_with_retries(model, prompt_text, GENERATION_PARAMS)

            # Build result entry with prompt index
            prompt_hash = hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()
            entry = {
                "prompt_index": p_idx,
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "prompt_sha256": prompt_hash,
                "category": prompt_data.get("category", "unknown"),
                "language": prompt_data.get("language", "unknown"),
                "english_note": prompt_data.get("english_note", ""),
                "expected_type": prompt_data.get("expected_type", ""),
                "raw_response": res.get("raw_response", ""),
                "clean_response": res.get("clean_response", ""),
                "eval_tokens": res.get("eval_count", 0),
                "prompt_tokens": res.get("prompt_eval_count", 0),
                "tokens_per_second": res.get("tokens_per_second", 0),
                "latency_seconds": res.get("latency_seconds", 0),
                "total_duration_ms": res.get("total_duration_ms", 0),
                "load_duration_ms": res.get("load_duration_ms", 0),
                "prompt_eval_duration_ms": res.get("prompt_eval_duration_ms", 0),
                "eval_duration_ms": res.get("eval_duration_ms", 0),
                "success": bool(res.get("success", False)),
                "errortype": res.get("errortype"),
                "errormessage": res.get("errormessage"),
                "http_status": res.get("http_status")
            }

            model_results[model].append(entry)

            # Update elapsed and completed indices
            try:
                model_elapsed[model] += float(res.get("latency_seconds") or 0)
            except Exception:
                pass
            model_completed_idx[model].add(p_idx)

            # Save per-model checkpoint
            save_checkpoint(model, model_results[model], elapsed_time=model_elapsed[model])

    # Finalize results
    for model in MODELS:
        mr = model_results.get(model, [])
        total_time = model_elapsed.get(model, 0.0)
        results[model] = {
            "model": model,
            "anonymous_name": MODEL_ANONYMIZER[model],
            "model_details": model_details_cache.get(model, {}),
            "total_prompts": len(prompts),
            "successful": sum(1 for r in mr if r.get("success", False)),
            "failed": sum(1 for r in mr if not r.get("success", False)),
            "total_time_seconds": total_time,
            "avg_time_per_prompt": (total_time / len(prompts)) if len(prompts) else 0,
            "results": mr
        }

    return results


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results: Dict, prompts: List[Dict], metadata: Dict):
    """Save results in multiple formats"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save full raw results with metadata
    full_results = {
        "metadata": metadata,
        "scoring_rubric": SCORING_RUBRIC,
        "generation_params": GENERATION_PARAMS,
        "models_tested": MODELS,
        "model_anonymizer": MODEL_ANONYMIZER,
        "results": results
    }
    
    raw_json_path = OUTPUT_DIR / f"raw_results_{timestamp}.json"
    with open(raw_json_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Saved raw JSON: {raw_json_path}")
    
    # 2. Create BLIND scoring CSV (evaluators see MODEL_A, MODEL_B, MODEL_C)
    rows = []
    for model_name, model_data in results.items():
        anonymous_name = model_data["anonymous_name"]
        for r in model_data["results"]:
            rows.append({
                "model": anonymous_name,  # BLIND - no real model name!
                "prompt_id": r["prompt_id"],
                "category": r["category"],
                "language": r["language"],
                "english_note": r["english_note"],
                "expected_type": r["expected_type"],
                "prompt": r["prompt"],
                "model_response": r.get("raw_response", ""),
                "raw_response": r.get("raw_response", ""),
                "clean_response": r.get("clean_response", r.get("model_response", "")),
                "EVAL1_SCORE": "",  # First evaluator
                "EVAL2_SCORE": "",  # Second evaluator
                "FINAL_SCORE": "",
                "NOTES": ""
            })

    # Shuffle blind CSV rows to avoid grouping by model
    random.seed(999)
    random.shuffle(rows)

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / f"blind_scoring_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"💾 Saved BLIND scoring CSV: {csv_path}")
    
    # 3. Save model key separately (DO NOT share with evaluators)
    key_path = OUTPUT_DIR / f"model_key_{timestamp}.json"
    with open(key_path, 'w') as f:
        json.dump(MODEL_ANONYMIZER, f, indent=2)
    print(f"🔑 Saved model key (keep separate from scoring CSV): {key_path}")
    
    # 4. Generate summary report
    md_path = OUTPUT_DIR / f"report_{timestamp}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# SLM Experiment Results\n\n")
        f.write(f"**Generated:** {timestamp}\n")
        f.write(f"**Hardware:** {metadata.get('ram_total_gb', '?')} GB RAM, {metadata.get('cpu_count', '?')} cores\n\n")
        
        f.write(f"## Summary\n\n")
        f.write(f"| Model | Anonymous | Success | Avg Latency | Avg tok/s |\n")
        f.write(f"|-------|-----------|---------|-------------|-----------|\n")
        
        for model_name, model_data in results.items():
            avg_latency = sum(r.get("latency_seconds", 0) for r in model_data["results"] if r.get("latency_seconds")) / max(1, model_data["successful"])
            avg_tokps = sum(r.get("tokens_per_second", 0) for r in model_data["results"] if r.get("tokens_per_second")) / max(1, model_data["successful"])
            f.write(f"| {model_name} | {model_data['anonymous_name']} | {model_data['successful']}/{model_data['total_prompts']} | {avg_latency:.1f}s | {avg_tokps:.1f} |\n")
        
        f.write(f"\n## Next Steps\n\n")
        f.write(f"1. **DO NOT LOOK AT model_key file** - keep scoring blind\n")
        f.write(f"2. Open `{csv_path.name}` in Excel/Google Sheets\n")
        f.write(f"3. Two evaluators independently score each response (0-3) using rubric below\n")
        f.write(f"4. Calculate Cohen's κ for inter-rater reliability\n")
        f.write(f"5. After scoring, use model_key to reveal which model is which\n")
        f.write(f"\n**Generation hyperparameters were held constant across all models to isolate architectural differences.**\n")
        f.write(f"\n## Scoring Rubric\n\n")
        f.write(f"```\n{SCORING_RUBRIC}\n```\n")
    
    print(f"💾 Saved report: {md_path}")
    
    # 5. Save failed prompts separately
    failed_rows = []
    for model_name, model_data in results.items():
        for r in model_data.get("results", []):
            if not r.get("success", False):
                failed_rows.append({
                    "model": model_name,
                    "prompt_id": r.get("prompt_id"),
                    "category": r.get("category"),
                    "error": r.get("error", "Unknown"),
                    "latency_seconds": r.get("latency_seconds", 0)
                })

    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        failed_path = OUTPUT_DIR / f"failed_prompts_{timestamp}.csv"
        failed_df.to_csv(failed_path, index=False, encoding='utf-8-sig')
        print(f"⚠️  Saved {len(failed_rows)} failed prompts to {failed_path}")

    return csv_path


def print_summary(results: Dict):
    """Print quick summary to console"""
    print("\n" + "="*60)
    print("📊 EXPERIMENT SUMMARY")
    print("="*60)
    
    for model_name, model_data in results.items():
        print(f"\n🦙 {model_name} → {model_data['anonymous_name']}")
        print(f"   ✅ Successful: {model_data['successful']}/{model_data['total_prompts']}")
        if model_data['successful'] > 0:
            latencies = [r["latency_seconds"] for r in model_data["results"] if r.get("latency_seconds", 0) > 0]
            tokps = [r["tokens_per_second"] for r in model_data["results"] if r.get("tokens_per_second", 0) > 0]
            if latencies:
                print(f"   ⏱️  Avg latency: {sum(latencies)/len(latencies):.1f}s")
                print(f"   🚀 Fastest: {min(latencies):.1f}s, 🐢 Slowest: {max(latencies):.1f}s")
            if tokps:
                print(f"   📈 Avg throughput: {sum(tokps)/len(tokps):.1f} tok/s")


def get_model_details(model: str) -> Dict:
    """Capture model metadata for reproducibility by calling `ollama show <model>`."""
    try:
        result = subprocess.run(["ollama", "show", model], capture_output=True, text=True, timeout=10)
        details = {"model": model}
        details["raw_output"] = result.stdout
        for line in result.stdout.splitlines():
            if ':' in line:
                k, v = line.split(':', 1)
                details[k.strip().lower().replace(' ', '_')] = v.strip()
        return details
    except Exception:
        return {"model": model, "error": "could not retrieve details"}


# ============================================================
# MAIN
# ============================================================

def main():
    """Main entry point"""
    print("="*60)
    print("🧪 The Cost of Intelligence — SLM Experiment")
    print("="*60)
    
    # Check for prompts file
    if not Path(PROMPTS_FILE).exists():
        print(f"\n❌ Error: {PROMPTS_FILE} not found!")
        print(f"   Please create a prompts.json file with your 40 prompts.")
        sys.exit(1)
    
    # Check Ollama is running
    print("\n🔍 Checking Ollama service...")
    if not check_ollama_running():
        print("❌ Ollama is not running!")
        print("   Run: ollama serve")
        print("   Or restart your terminal and try again.")
        sys.exit(1)
    print("✅ Ollama is running")
    
    # Verify models
    print("\n🔍 Verifying models...")
    all_available, missing = verify_models_available(MODELS)
    if not all_available:
        print(f"⚠️  Missing models: {missing}")
        print(f"   Run: ollama pull <model>")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        for model in MODELS:
            print(f"   ✅ {model} → {MODEL_ANONYMIZER[model]}")
    
    # Capture system metadata
    print("\n📊 Capturing system metadata...")
    metadata = capture_system_metadata()
    print(f"   RAM: {metadata.get('ram_total_gb', '?')} GB")
    print(f"   CPU cores: {metadata.get('cpu_count', '?')}")
    
    # Load prompts (shuffled for fairness)
    print(f"\n📋 Loading prompts...")
    prompts, prompt_metadata = load_prompts(PROMPTS_FILE, shuffle=True)
    # Attach prompt ordering metadata for reproducibility
    metadata["prompt_metadata"] = prompt_metadata
    
    # Print scoring rubric reminder
    print("\n📋 Scoring rubric for evaluators (saved in report):")
    print("   0=Incorrect, 1=Partial, 2=Correct, 3=Excellent")
    
    # Run experiment
    print(f"\n🚀 Starting experiment with {len(prompts)} prompts across {len(MODELS)} models...")
    print(f"   Estimated time: ~{len(prompts) * len(MODELS) * 10 / 60:.0f} minutes\n")
    
    start_time = time.time()
    results = run_experiment(prompts)
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Total experiment time: {total_time / 60:.1f} minutes")
    
    # Save and summarize
    csv_path = save_results(results, prompts, metadata)
    print_summary(results)
    
    print("\n" + "="*60)
    print("✅ EXPERIMENT COMPLETE")
    print("="*60)
    print(f"\n📁 Results saved to: {OUTPUT_DIR}/")
    print(f"\n🔜 NEXT STEPS FOR YOUR PAPER:")
    print(f"   1. Give '{csv_path.name}' to TWO independent evaluators")
    print(f"   2. They score each response 0-3 using rubric in report")
    print(f"   3. Calculate Cohen's κ = (observed agreement - expected) / (1 - expected)")
    print(f"   4. After scoring, use model_key file to decode MODEL_A/B/C")
    print(f"   5. Add to paper as primary results (your own Table 5)")
    print("")


if __name__ == "__main__":
    main()