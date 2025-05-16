# ---------------- evaluate_models_unified_fixed_multithread_claude.py ----------------
"""
Evaluates Claude LLM on Personalized Follow-up Text Generation (Task 3.3 - predicting body text)
using multiple threads.

- Only Claude model is supported.
- Task 3.4 related code has been removed.
- Dataset is split into N parts for N threads.
- Includes retry mechanism for HTTP 427 errors from Claude API.
- Logs and results are saved in a 'MultiThread' directory.
- Each thread has its own log file in 'MultiThread/thread_logs'.
- Final statistics are aggregated from all threads, including per-domain stats, in the main log.
"""
import csv
import os
import sys
import time
import random
from datetime import datetime as dt
from copy import deepcopy
import re
import json
import logging
import warnings
import threading # Added for multithreading
from math import ceil # Added for chunking
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Import API Libraries ---
# Only Anthropic is needed now
try:
    from anthropic import AnthropicVertex
    from anthropic import APIError as AnthropicAPIError
    from anthropic import RateLimitError as AnthropicRateLimitError
except ImportError:
    print("ERROR: 'anthropic' or 'anthropic-vertex' library not found. Claude models will not be available.")
    print("Please install it: pip install anthropic anthropic-vertex")
    AnthropicVertex = None
    AnthropicAPIError = None
    AnthropicRateLimitError = None
    sys.exit(1)

# --- Import Evaluation Libraries ---
MULTI_EVAL_ENABLED = True  # Default attempt state
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
except ImportError:
    print(
        "ERROR: 'sentence-transformers' or 'torch' not found. Please install them: pip install sentence-transformers torch")
    sys.exit(1)

try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        try:
            nltk.download('punkt', quiet=True)
            print("'punkt' downloaded successfully.")
        except Exception as nltk_e:
            print(f"ERROR: Failed to download NLTK 'punkt' tokenizer: {nltk_e}. ROUGE/BLEU/METEOR disabled.")
            MULTI_EVAL_ENABLED = False
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("NLTK 'wordnet' resource not found (for METEOR). Downloading...")
        try:
            nltk.download('wordnet', quiet=True)
            print("'wordnet' downloaded successfully.")
        except Exception as nltk_e:
            print(f"ERROR: Failed to download NLTK 'wordnet': {nltk_e}. METEOR will be disabled.")
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        print("NLTK 'omw-1.4' resource not found (optional for METEOR). Downloading...")
        try:
            nltk.download('omw-1.4', quiet=True)
            print("'omw-1.4' downloaded successfully.")
        except Exception as nltk_e:
            print(f"WARNING: Failed to download NLTK 'omw-1.4': {nltk_e}. METEOR might have reduced coverage.")

except ImportError:
    print(
        "ERROR: 'rouge-score' or 'nltk' not found. Install them (`pip install rouge-score nltk`) for ROUGE/BLEU/METEOR evaluation.")
    MULTI_EVAL_ENABLED = False
    print("INFO: MULTI_EVAL (ROUGE/BLEU/METEOR) disabled due to missing libraries.")

try:
    import numpy as np
except ImportError:
    print("ERROR: 'numpy' not found. Please install it: pip install numpy")
    sys.exit(1)
# --- End Imports ---

# --- >> Consolidated Flags << ---
FLAG_33 = True  # Run Task 3.3 (Predict Body Text) - This is fixed
FLAG_34 = False # Task 3.4 is removed
test_flag = False  # If True, run only first 5 prompts per thread (approx)
CUT = False  # If True (and test_flag=False), run only first 500 prompts (approx, total)
MULTI_EVAL = MULTI_EVAL_ENABLED  # Use status determined during import
WRITE_OUTPUT = True  # If True, save best SBERT candidate output separately (per model)
TASK_NAME = "Predict_Body_Text_3.3"
NUM_THREADS = 12 # Number of threads to use

# --- Output Directory ---
OUTPUT_DIR = "MultiThread" # All outputs will go here
os.makedirs(OUTPUT_DIR, exist_ok=True)
THREAD_LOG_DIR = os.path.join(OUTPUT_DIR, "thread_logs") # Subdirectory for thread-specific logs
os.makedirs(THREAD_LOG_DIR, exist_ok=True)


# --- Main Logging Setup (for aggregated results and overall progress) ---
main_log_filename = os.path.join(OUTPUT_DIR, f"EVAL-MultiThread_Claude-{dt.now():%Y%m%d_%H%M%S}.log")
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s') # Changed to name for clarity
main_logger = logging.getLogger("MainEvaluator") # Specific name for the main logger
main_logger.setLevel(logging.INFO)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING) # Quieten SentenceTransformer's own logger

if main_logger.hasHandlers(): main_logger.handlers.clear()

file_handler = logging.FileHandler(main_log_filename, encoding='utf-8')
file_handler.setFormatter(log_formatter)
main_logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
main_logger.addHandler(console_handler)
# --- End Main Logging Setup ---


# --- General Configuration ---
INPUT_JSONL_PATH = "WithoutConversationPrompts_BodyPrediction_v2.jsonl"
GENERATION_MAX_TOKENS = 256
NUM_CANDIDATES = 1 # For Claude, we make 1 call per prompt.
GENERATION_TEMPERATURE = 0.7

main_logger.info(f"Running Task 3.3: Predict Body Text from {INPUT_JSONL_PATH}")
if not MULTI_EVAL:
    main_logger.warning("MULTI_EVAL (ROUGE/BLEU/METEOR) is disabled.")
else:
    main_logger.info("MULTI_EVAL (ROUGE/BLEU/METEOR) is enabled.")

API_DELAY_SECONDS = 1
PROMPT_DELAY_SECONDS = 0
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'
LOW_SIM_THRESHOLD = 0.05
HIGH_SIM_THRESHOLD = 0.9

# --- GCP Project and Location (used by Claude on Vertex) ---
YOUR_PROJECT_ID = '' # Replace with your Project ID
YOUR_LOCATION_CLAUDE = '' # or your specific Claude region

# --- Model Definitions (Claude Only) ---
MODEL_CONFIGS = {
    "": {
        "api_type": "anthropic_vertex",
        "project_id": YOUR_PROJECT_ID,
        "region": YOUR_LOCATION_CLAUDE,
        "model_id": ""
    }
}
# --- End Model Definitions ---

# --- Global SBERT Model and ROUGE Scorer (initialized once) ---
sbert_model_global = None
rouge_scorer_instance_global = None
meteor_available_global = False

# --- Helper Functions ---
def get_thread_concatenated_text(thread_nodes): # Unused
    if not thread_nodes or not isinstance(thread_nodes, list): return ""
    return " ".join([node.get("body", "") for node in thread_nodes if
                     isinstance(node, dict) and isinstance(node.get("body"), str) and node.get("body", "").strip()])

def get_thread_embedding(thread_nodes, sbert_model_instance, device_to_use): # Unused
    if not thread_nodes or not isinstance(thread_nodes, list): return None
    bodies = [node.get("body", "") for node in thread_nodes if
              isinstance(node, dict) and isinstance(node.get("body"), str) and node.get("body", "").strip()]
    if not bodies: return None
    try:
        with torch.no_grad():
            embeddings = sbert_model_instance.encode(bodies, convert_to_tensor=True, device=device_to_use,
                                                     show_progress_bar=False, batch_size=128)
            mean_embedding = torch.mean(embeddings.to('cpu'), dim=0)
        return mean_embedding
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "MPS backend out of memory" in str(e):
            main_logger.error(f"OOM Error during SBERT encoding: {e}.") # Use main_logger for global issues
        else:
            main_logger.error(f"Runtime error during SBERT encoding/mean calculation: {e}")
        return None
    except Exception as e:
        main_logger.error(f"Error during SBERT encoding/mean calculation: {e}");
        return None

def call_anthropic_vertex_with_retry(client, model_id, prompt_text, max_gen_tokens, temperature, worker_logger_instance):
    if AnthropicVertex is None: return [], "Anthropic SDK not imported."
    
    candidate_responses = []
    error_message = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            message = client.messages.create(
                max_tokens=max_gen_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": prompt_text,
                    }
                ],
                model=model_id,
                temperature=temperature,
            )

            if message.content and isinstance(message.content, list) and message.content[0].text:
                candidate_responses.append(message.content[0].text)
            elif isinstance(message.content, str):
                 candidate_responses.append(message.content)
            else:
                error_message = f"No valid content received from Anthropic model {model_id}. Response: {message.model_dump_json(indent=2)}"
                worker_logger_instance.warning(error_message)
                candidate_responses = []
                break 

            if message.stop_reason not in ["end_turn", "max_tokens", "stop_sequence"]:
                worker_logger_instance.warning(f"Anthropic model {model_id} finished with reason: '{message.stop_reason}'. Output might be incomplete.")
            error_message = None 
            break 
        except AnthropicRateLimitError as e:
            error_message = f"AnthropicRateLimitError for {model_id}: {e}"
        except AnthropicAPIError as e:
            err_code = getattr(e, 'status_code', 'N/A')
            err_body = getattr(e, 'body', {})
            err_msg_detail = str(err_body.get('error', {}).get('message', str(e))) if err_body else str(e)
            error_message = f"AnthropicAPIError for {model_id}: Status={err_code}, Message={err_msg_detail[:250]}..."
            if err_code == 427:
                if attempt < MAX_RETRIES:
                    worker_logger_instance.warning(f"Attempt {attempt + 1}/{MAX_RETRIES + 1} for {model_id} encountered HTTP 427. Sleeping for 5s before retry...")
                    time.sleep(5)
                    continue 
                else: 
                    worker_logger_instance.error(f"Final API Error (HTTP 427) after {attempt + 1} attempts for {model_id}: {error_message}")
                    break 
            if err_code in [400, 401, 403]:
                 worker_logger_instance.warning(f"Non-retryable client error {err_code} for {model_id}: {error_message}")
                 break
            elif err_code in [500, 503] and attempt < MAX_RETRIES:
                pass
            else:
                if attempt >= MAX_RETRIES:
                    worker_logger_instance.error(f"Max retries reached for {model_id} with error {err_code}: {error_message}")
                else:
                    worker_logger_instance.warning(f"Unhandled or non-retryable AnthropicAPIError {err_code} for {model_id}: {error_message}")
                break
        except Exception as e:
            error_message = f"Unexpected Error during Anthropic API call for {model_id}: {type(e).__name__} - {str(e)[:150]}..."
            break 

        if error_message and attempt < MAX_RETRIES:
            worker_logger_instance.warning(
                f"Attempt {attempt + 1}/{MAX_RETRIES + 1} failed for Anthropic model {model_id}: {error_message}. Retrying in {RETRY_DELAY_SECONDS}s...")
            time.sleep(RETRY_DELAY_SECONDS)
        elif error_message:
            worker_logger_instance.error(f"Final API Error after {attempt + 1} attempts for Anthropic model {model_id}: {error_message}")
            candidate_responses = []
    
    return candidate_responses, error_message

def log_domain_metrics(current_domain_stats, logger_instance_to_use, meteor_is_available_flag, active_model_count=1):
    if not current_domain_stats:
        logger_instance_to_use.info("  No domain statistics collected yet.")
        return

    for domain, stats in sorted(current_domain_stats.items()):
        logger_instance_to_use.info(f"  --- Domain: {domain} ---")
        total_model_attempts = stats.get('total_model_attempts_on_prompts', 0)
        logger_instance_to_use.info(f"    Total Model Attempts: {total_model_attempts}")
        true_label_errors = stats.get('prompts_with_true_label_errors', 0)
        if true_label_errors > 0:
            logger_instance_to_use.info(f"    Prompts with True Label Processing Errors: {true_label_errors}")
        api_errors = stats.get('api_errors_count', 0)
        post_api_eval_errors = stats.get('post_api_eval_errors_count', 0)
        logger_instance_to_use.info(f"    API Call Errors: {api_errors}")
        logger_instance_to_use.info(f"    Post-API Eval Errors: {post_api_eval_errors}")
        sbert_evals = stats.get('successful_sbert_evals', 0)
        avg_sbert = (stats.get('sum_sbert_scores', 0.0) / sbert_evals) if sbert_evals > 0 else 0.0
        logger_instance_to_use.info(f"    Avg SBERT Score: {avg_sbert:.4f} (over {sbert_evals} evals)")
        if MULTI_EVAL:
            rl_evals = stats.get('successful_rougeL_evals', 0)
            avg_rl = (stats.get('sum_rougeL_scores', 0.0) / rl_evals) if rl_evals > 0 else 0.0
            logger_instance_to_use.info(f"    Avg ROUGE-L F1: {avg_rl:.4f} (over {rl_evals} evals)")
            r1_evals = stats.get('successful_rouge1_evals', 0)
            avg_r1 = (stats.get('sum_rouge1_scores', 0.0) / r1_evals) if r1_evals > 0 else 0.0
            logger_instance_to_use.info(f"    Avg ROUGE-1 F1: {avg_r1:.4f} (over {r1_evals} evals)")
            bleu_evals = stats.get('successful_bleu_evals', 0)
            avg_bleu = (stats.get('sum_bleu_scores', 0.0) / bleu_evals) if bleu_evals > 0 else 0.0
            logger_instance_to_use.info(f"    Avg BLEU Score: {avg_bleu:.4f} (over {bleu_evals} evals)")
            if meteor_is_available_flag:
                meteor_evals = stats.get('successful_meteor_evals', 0)
                avg_meteor = (stats.get('sum_meteor_scores', 0.0) / meteor_evals) if meteor_evals > 0 else 0.0
                logger_instance_to_use.info(f"    Avg METEOR Score: {avg_meteor:.4f} (over {meteor_evals} evals)")
        logger_instance_to_use.info("-" * 20)

# --- Worker Function for Threading ---
def process_prompts_worker(
        thread_id, prompts_chunk_with_indices, model_name, model_config_details,
        global_sbert_model, global_rouge_scorer, global_meteor_available,
        global_settings, device_to_use):
    
    thread_name = f"WorkerThread-{thread_id}"
    worker_logger = logging.getLogger(thread_name)
    worker_logger.setLevel(logging.INFO)
    
    if not worker_logger.hasHandlers(): # Setup handler only if not already configured (e.g. by previous run in interactive session)
        thread_log_filename = os.path.join(THREAD_LOG_DIR, f"worker_thread_{thread_id}.log")
        thread_file_handler = logging.FileHandler(thread_log_filename, encoding='utf-8')
        thread_file_handler.setFormatter(log_formatter) # Use the same global formatter
        worker_logger.addHandler(thread_file_handler)
        # Optionally, add a console handler for thread logs if desired, but can be noisy.
        # thread_console_handler = logging.StreamHandler(sys.stdout)
        # thread_console_handler.setFormatter(log_formatter)
        # worker_logger.addHandler(thread_console_handler)

    worker_logger.info(f"Worker {thread_id} started, assigned {len(prompts_chunk_with_indices)} prompts.")

    api_client = None
    try:
        if model_config_details["api_type"] == "anthropic_vertex":
            if AnthropicVertex:
                api_client = AnthropicVertex(region=model_config_details["region"], project_id=model_config_details["project_id"])
            else:
                raise ConnectionError("Anthropic SDK not available for Claude.")
        else:
            raise ValueError(f"Unsupported api_type '{model_config_details['api_type']}' in worker.")
    except Exception as e:
        worker_logger.error(f"Client Init Error for {model_name}: {e}")
        main_logger.error(f"Worker {thread_id}: Client Init Error for {model_name}: {e}") # Also log to main for visibility
        return ({'total': len(prompts_chunk_with_indices), 'errors': len(prompts_chunk_with_indices)}, 
                {}, [], [], [], set())

    worker_model_stats = {'successful_preds': 0, 'total_similarity': 0.0,
                          'successful_rougeL': 0, 'total_rougeL': 0.0,
                          'successful_rouge1': 0, 'total_rouge1': 0.0,
                          'successful_bleu': 0, 'total_bleu': 0.0,
                          'successful_meteor': 0, 'total_meteor': 0.0,
                          'errors': 0, 'total': 0, 'true_label_errors_count': 0}
    worker_domain_stats = {}
    worker_model_results_list = [] 
    worker_combined_json_lines = [] 
    worker_best_candidate_lines = [] 
    worker_confusing_prompt_indices = set()

    for local_idx, (original_prompt_idx, prompt_entry) in enumerate(prompts_chunk_with_indices):
        user = prompt_entry.get("user", "Unknown")
        current_prompt_text = prompt_entry["prompt"]
        true_label_input = prompt_entry["true_label"]
        target_post_id = f"Prompt_{original_prompt_idx + 1}"
        current_subreddit = prompt_entry.get("subreddit", "Unknown_Subreddit") 
        
        worker_logger.info(
            f"Processing Prompt {local_idx + 1}/{len(prompts_chunk_with_indices)} (Global Index: {original_prompt_idx + 1}, User: {user}, Subreddit: {current_subreddit})")

        if current_subreddit not in worker_domain_stats:
            worker_domain_stats[current_subreddit] = {
                'total_model_attempts_on_prompts': 0, 'prompts_with_true_label_errors': 0,
                'successful_sbert_evals': 0, 'sum_sbert_scores': 0.0,
                'successful_rougeL_evals': 0, 'sum_rougeL_scores': 0.0,
                'successful_rouge1_evals': 0, 'sum_rouge1_scores': 0.0,
                'successful_bleu_evals': 0, 'sum_bleu_scores': 0.0,
                'successful_meteor_evals': 0, 'sum_meteor_scores': 0.0,
                'api_errors_count': 0, 'post_api_eval_errors_count': 0
            }

        prompt_results_for_combined_json = {
            "prompt_index": original_prompt_idx, "user": user, "task": TASK_NAME, 
            "input_prompt": current_prompt_text, "true_label": true_label_input, 
            "target_post": None, "subreddit": current_subreddit, "model_outputs": {}
        }

        true_label_embeddings = None; true_label_text_single = ""; true_label_tokenized_list = []; true_label_tokenized_for_meteor = []
        processing_error_occurred = False

        try:
            if not isinstance(true_label_input, str) or not true_label_input.strip(): 
                raise ValueError("True label text is empty or not a string.")
            true_label_text_single = true_label_input
            true_label_embeddings = global_sbert_model.encode(true_label_text_single, convert_to_tensor=True, device=device_to_use)
            if global_settings['MULTI_EVAL']:
                true_label_tokenized_list = [nltk.word_tokenize(true_label_text_single.lower())]
                if not true_label_tokenized_list[0]: raise ValueError("True label tokenization for BLEU resulted in empty list.")
                if global_meteor_available:
                    tokenized_ref = nltk.word_tokenize(true_label_text_single.lower())
                    if not tokenized_ref: raise ValueError("True label tokenization for METEOR resulted in empty list.")
                    true_label_tokenized_for_meteor = [tokenized_ref]
        except ValueError as ve:
            worker_logger.error(f"Error processing true label for prompt global_idx {original_prompt_idx + 1}: {ve} - Skipping.")
            processing_error_occurred = True; prompt_results_for_combined_json["error_processing_true_label"] = str(ve)
        except Exception as e:
            worker_logger.error(f"Unexpected error processing true label for prompt global_idx {original_prompt_idx + 1}: {e} - Skipping.")
            processing_error_occurred = True; prompt_results_for_combined_json["error_processing_true_label"] = f"Unexpected error: {e}"

        worker_model_stats['total'] += 1
        worker_domain_stats[current_subreddit]['total_model_attempts_on_prompts'] += 1
        
        if processing_error_occurred:
            worker_model_stats['errors'] += 1
            worker_model_stats['true_label_errors_count'] +=1
            worker_domain_stats[current_subreddit]['prompts_with_true_label_errors'] += 1
            worker_domain_stats[current_subreddit]['post_api_eval_errors_count'] += 1
            worker_model_results_list.append({
                "prompt_index": original_prompt_idx, "user": user, "prompt_preview": current_prompt_text[:100] + "...",
                "true_label_preview": "ERROR", "prediction": "N/A", "best_sbert_score": "N/A",
                "best_rougeL_score": "N/A", "best_rouge1_score": "N/A", "best_bleu_score": "N/A",
                "best_meteor_score": "N/A", "api_error": f"Skipped: True Label Error: {prompt_results_for_combined_json['error_processing_true_label']}"
            })
            prompt_results_for_combined_json["model_outputs"][model_name] = {"best_candidate_text": None, "best_sbert_score": None, "api_error": f"Skipped: True Label Error"}
            worker_combined_json_lines.append(deepcopy(prompt_results_for_combined_json))
            # Log current thread averages after this error
            # (Metrics won't change for this specific post, but total count increases)
        else: # No true label processing error, proceed with model call
            api_error = None; candidate_responses = []; max_similarity_score = -1.0; best_sbert_candidate_raw_string = ""
            current_sbert_score, current_rougeL_score, current_rouge1_score, current_bleu_score, current_meteor_score = -1.0, -1.0, -1.0, -1.0, -1.0
            
            worker_logger.info(f"  Model {model_name[:30]:<30} | Requesting candidate...")
            start_time = time.time()
            candidate_responses, api_error = call_anthropic_vertex_with_retry(
                api_client, model_config_details["model_id"], current_prompt_text, 
                global_settings['GENERATION_MAX_TOKENS'], global_settings['GENERATION_TEMPERATURE'], worker_logger
            )
            duration = time.time() - start_time
            best_sbert_candidate_preview = "Error"

            if api_error:
                worker_logger.warning(f"  Model {model_name[:30]:<30} | API Error! ({str(api_error)[:60]}...)")
                worker_model_stats['errors'] += 1
                worker_domain_stats[current_subreddit]['api_errors_count'] += 1
                best_sbert_candidate_preview = f"Error: {api_error}"
            elif not candidate_responses or not candidate_responses[0].strip():
                no_cand_msg = "No candidates received or candidate is empty"
                worker_logger.warning(f"  Model: {model_name[:30]:<30} | Pred Error! {no_cand_msg}.")
                worker_model_stats['errors'] += 1
                worker_domain_stats[current_subreddit]['post_api_eval_errors_count'] += 1
                best_sbert_candidate_preview = f"Error: {no_cand_msg}"; api_error = no_cand_msg
            else:
                best_sbert_candidate_raw_string = candidate_responses[0]
                try:
                    with torch.no_grad():
                        candidate_embedding = global_sbert_model.encode(best_sbert_candidate_raw_string, convert_to_tensor=True, device=device_to_use)
                        similarity_tensor = util.cos_sim(candidate_embedding, true_label_embeddings.to(device_to_use))
                        max_similarity_score = similarity_tensor.item(); current_sbert_score = max_similarity_score
                    worker_model_stats['successful_preds'] += 1; worker_model_stats['total_similarity'] += max_similarity_score
                    worker_domain_stats[current_subreddit]['successful_sbert_evals'] += 1; worker_domain_stats[current_subreddit]['sum_sbert_scores'] += max_similarity_score

                    if global_settings['MULTI_EVAL']:
                        cand_text_for_eval = best_sbert_candidate_raw_string
                        if cand_text_for_eval.strip():
                            if global_rouge_scorer:
                                try:
                                    scores_dict = global_rouge_scorer.score(true_label_text_single, cand_text_for_eval)
                                    current_rougeL_score = scores_dict['rougeL'].fmeasure; current_rouge1_score = scores_dict['rouge1'].fmeasure
                                except Exception as r_e: worker_logger.warning(f"ROUGE error: {r_e}")
                            candidate_tokens = nltk.word_tokenize(cand_text_for_eval.lower())
                            if candidate_tokens:
                                try: current_bleu_score = sentence_bleu(true_label_tokenized_list, candidate_tokens, smoothing_function=SmoothingFunction().method1)
                                except Exception as b_e: worker_logger.warning(f"BLEU error: {b_e}")
                                if global_meteor_available and true_label_tokenized_for_meteor:
                                    try: current_meteor_score = meteor_score(true_label_tokenized_for_meteor, candidate_tokens)
                                    except Exception as m_e: worker_logger.warning(f"METEOR error: {m_e}")
                        if current_rougeL_score >= 0.0: worker_model_stats['successful_rougeL'] += 1; worker_model_stats['total_rougeL'] += current_rougeL_score; worker_domain_stats[current_subreddit]['successful_rougeL_evals'] += 1; worker_domain_stats[current_subreddit]['sum_rougeL_scores'] += current_rougeL_score
                        if current_rouge1_score >= 0.0: worker_model_stats['successful_rouge1'] += 1; worker_model_stats['total_rouge1'] += current_rouge1_score; worker_domain_stats[current_subreddit]['successful_rouge1_evals'] += 1; worker_domain_stats[current_subreddit]['sum_rouge1_scores'] += current_rouge1_score
                        if current_bleu_score >= 0.0: worker_model_stats['successful_bleu'] += 1; worker_model_stats['total_bleu'] += current_bleu_score; worker_domain_stats[current_subreddit]['successful_bleu_evals'] += 1; worker_domain_stats[current_subreddit]['sum_bleu_scores'] += current_bleu_score
                        if global_meteor_available and current_meteor_score >= 0.0: worker_model_stats['successful_meteor'] += 1; worker_model_stats['total_meteor'] += current_meteor_score; worker_domain_stats[current_subreddit]['successful_meteor_evals'] += 1; worker_domain_stats[current_subreddit]['sum_meteor_scores'] += current_meteor_score
                    
                    best_sbert_candidate_preview = best_sbert_candidate_raw_string[:100] + "..."
                    log_info = f"Best SBERT: {max_similarity_score:.4f}"
                    if global_settings['MULTI_EVAL']: log_info += f" | R-L: {current_rougeL_score:.4f} | R-1: {current_rouge1_score:.4f} | BLEU: {current_bleu_score:.4f}"
                    if global_meteor_available: log_info += f" | METEOR: {current_meteor_score:.4f}"
                    log_info += f" | Time: {duration:.2f}s"
                    worker_logger.info(f"  Model {model_name[:30]:<30} | {log_info}")
                    if global_settings['WRITE_OUTPUT']: worker_best_candidate_lines.append({"prompt_index": original_prompt_idx, "user": user, "target_id": target_post_id, "true_label": true_label_input, "subreddit": current_subreddit, "best_sbert_candidate_text": best_sbert_candidate_raw_string, "best_sbert_score": max_similarity_score})
                except (ValueError, RuntimeError, Exception) as eval_e:
                    eval_err_msg = f"Eval Error ({type(eval_e).__name__}): {str(eval_e)[:100]}..."; worker_logger.error(f"  {eval_err_msg} for {model_name}!");
                    worker_model_stats['errors'] += 1; worker_domain_stats[current_subreddit]['post_api_eval_errors_count'] += 1
                    best_sbert_candidate_preview = eval_err_msg; max_similarity_score = -1.0; current_rougeL_score = -1.0; current_rouge1_score = -1.0; current_bleu_score = -1.0; current_meteor_score = -1.0
                    api_error = eval_err_msg # Treat as API error for reporting

            worker_model_results_list.append({
                "prompt_index": original_prompt_idx, "user": user, "prompt_preview": current_prompt_text[:100] + "...",
                "true_label_preview": true_label_input[:80] + "...", "prediction": best_sbert_candidate_preview,
                "best_sbert_score": f"{max_similarity_score:.6f}" if max_similarity_score != -1.0 else "N/A",
                "best_rougeL_score": f"{current_rougeL_score:.6f}" if global_settings['MULTI_EVAL'] and current_rougeL_score >= 0.0 else "N/A",
                "best_rouge1_score": f"{current_rouge1_score:.6f}" if global_settings['MULTI_EVAL'] and current_rouge1_score >= 0.0 else "N/A",
                "best_bleu_score": f"{current_bleu_score:.6f}" if global_settings['MULTI_EVAL'] and current_bleu_score >= 0.0 else "N/A",
                "best_meteor_score": f"{current_meteor_score:.6f}" if global_settings['MULTI_EVAL'] and global_meteor_available and current_meteor_score >= 0.0 else "N/A",
                "api_error": str(api_error) if api_error else ""})
            prompt_results_for_combined_json["model_outputs"][model_name] = {"best_candidate_text": best_sbert_candidate_raw_string if max_similarity_score != -1.0 and not api_error else None, "best_sbert_score": max_similarity_score if max_similarity_score != -1.0 and not api_error else None, "api_error": str(api_error) if api_error else None}
            worker_combined_json_lines.append(deepcopy(prompt_results_for_combined_json))

            if current_sbert_score != -1.0 and (current_sbert_score < global_settings['LOW_SIM_THRESHOLD'] or current_sbert_score > global_settings['HIGH_SIM_THRESHOLD']):
                worker_confusing_prompt_indices.add(original_prompt_idx)
                worker_logger.warning(f"--- Prompt global_idx {original_prompt_idx + 1} flagged as CONFUSING (SBERT: {current_sbert_score:.4f}) ---")

        # Log current thread's average metrics after each post
        avg_sbert_thread = (worker_model_stats['total_similarity'] / worker_model_stats['successful_preds']) if worker_model_stats['successful_preds'] > 0 else 0.0
        avg_rl_thread = (worker_model_stats['total_rougeL'] / worker_model_stats['successful_rougeL']) if worker_model_stats['successful_rougeL'] > 0 else 0.0
        avg_r1_thread = (worker_model_stats['total_rouge1'] / worker_model_stats['successful_rouge1']) if worker_model_stats['successful_rouge1'] > 0 else 0.0
        avg_bleu_thread = (worker_model_stats['total_bleu'] / worker_model_stats['successful_bleu']) if worker_model_stats['successful_bleu'] > 0 else 0.0
        avg_meteor_thread = (worker_model_stats['total_meteor'] / worker_model_stats['successful_meteor']) if global_meteor_available and worker_model_stats['successful_meteor'] > 0 else 0.0
        
        metrics_log_str = f"  Thread Cumulative Averages (after {local_idx + 1} processed in this thread): "
        metrics_log_str += f"SBERT: {avg_sbert_thread:.4f} ({worker_model_stats['successful_preds']}/{worker_model_stats['total'] - worker_model_stats['true_label_errors_count']} evals)"
        if global_settings['MULTI_EVAL']:
            metrics_log_str += f", R-L: {avg_rl_thread:.4f} ({worker_model_stats['successful_rougeL']} evals)"
            metrics_log_str += f", R-1: {avg_r1_thread:.4f} ({worker_model_stats['successful_rouge1']} evals)"
            metrics_log_str += f", BLEU: {avg_bleu_thread:.4f} ({worker_model_stats['successful_bleu']} evals)"
            if global_meteor_available:
                metrics_log_str += f", METEOR: {avg_meteor_thread:.4f} ({worker_model_stats['successful_meteor']} evals)"
        metrics_log_str += f". API/Eval Errors so far: {worker_model_stats['errors'] - worker_model_stats['true_label_errors_count']}. True Label Errors: {worker_model_stats['true_label_errors_count']}."
        worker_logger.info(metrics_log_str)


        if device_to_use != 'cpu':
            try: 
                if 'candidate_embedding' in locals(): del candidate_embedding
                if 'similarity_tensor' in locals(): del similarity_tensor
                torch.cuda.empty_cache()
            except (NameError, Exception): pass
        
        if global_settings['API_DELAY_SECONDS'] > 0: time.sleep(global_settings['API_DELAY_SECONDS'])
    
    if global_settings['PROMPT_DELAY_SECONDS'] > 0 and local_idx < len(prompts_chunk_with_indices) -1 : # Check local_idx scope if needed
        worker_logger.info(f"--- Delaying {global_settings['PROMPT_DELAY_SECONDS']}s between prompts (end of batch in this case) ---") # This delay will now only apply if the loop didn't finish
        time.sleep(global_settings['PROMPT_DELAY_SECONDS'])

    worker_logger.info(f"Worker {thread_id} finished processing. Successful SBERT Evals: {worker_model_stats['successful_preds']}, Total Processed in Thread: {worker_model_stats['total']}, API/Eval Errors: {worker_model_stats['errors'] - worker_model_stats['true_label_errors_count']}, True Label Errors: {worker_model_stats['true_label_errors_count']}")
    return (worker_model_stats, worker_domain_stats, worker_model_results_list, 
            worker_combined_json_lines, worker_best_candidate_lines, worker_confusing_prompt_indices)

# --- Main Evaluation Logic ---
def main():
    global sbert_model_global, rouge_scorer_instance_global, meteor_available_global

    if not os.path.exists(INPUT_JSONL_PATH): 
        main_logger.error(f"Input JSONL file not found at {INPUT_JSONL_PATH}"); sys.exit(1)
    
    main_logger.info(f"Running Task: {TASK_NAME}")
    main_logger.info(f"Input file: {INPUT_JSONL_PATH}")
    main_logger.info(f"Output directory: {OUTPUT_DIR}, Thread logs in: {THREAD_LOG_DIR}")
    main_logger.info(f"Number of threads: {NUM_THREADS}")
    main_logger.info(f"Generating 1 candidate per prompt.") # Claude behavior
    main_logger.info(f"Multi-Eval (ROUGE/BLEU/METEOR) enabled: {MULTI_EVAL}")
    main_logger.info(f"Write Output (Best Candidate Per Model) enabled: {WRITE_OUTPUT}")

    main_logger.info(f"Loading SBERT model: {SBERT_MODEL_NAME}...")
    try:
        device = 'cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
        main_logger.info(f"Using device: {device}")
        sbert_model_global = SentenceTransformer(SBERT_MODEL_NAME, device=device)
        main_logger.info("SBERT model loaded successfully.")
    except Exception as e:
        main_logger.error(f"Error loading SBERT model: {e}"); sys.exit(1)

    if MULTI_EVAL:
        try:
            rouge_scorer_instance_global = rouge_scorer.RougeScorer(['rougeL', 'rouge1'], use_stemmer=True)
            main_logger.info("ROUGE (L,1) scorer initialized.")
        except Exception as e: main_logger.error(f"Failed to initialize ROUGE scorer: {e}. ROUGE evaluation disabled.")
        if not meteor_available_global: main_logger.warning("METEOR scoring will be disabled due to missing NLTK resources.")
    else: main_logger.info("MULTI_EVAL is disabled. ROUGE/BLEU/METEOR scores will not be calculated.")

    timestamp_str = dt.now().strftime("%Y%m%d_%H%M%S")
    run_identifier = f"{'_test' if test_flag else ('_cut500' if CUT else '')}"
    combined_output_filename = os.path.join(OUTPUT_DIR, f"input_best_output_best_match_{TASK_NAME}{run_identifier}_{timestamp_str}.jsonl")
    main_logger.info(f"Aggregated combined input/output/match will be saved to: {combined_output_filename}")
    
    prompts_data = []
    main_logger.info(f"Reading prompts from {INPUT_JSONL_PATH}...")
    try:
        with open(INPUT_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip();
                if not line: continue
                try:
                    data_dict = json.loads(line)
                    if not data_dict.get("prompt") or not data_dict.get("true_label") or not isinstance(data_dict["true_label"], str): 
                        main_logger.warning(f"Skip L{line_num}. Invalid 'prompt' or 'true_label'. Data: {str(data_dict)[:100]}"); continue
                    prompts_data.append({"original_index": len(prompts_data), "data": data_dict})
                except (json.JSONDecodeError, Exception) as e: main_logger.warning(f"Skip L{line_num}. Parse/Validation error: {e}. Line: {line[:100]}...")
    except FileNotFoundError: main_logger.error(f"Input file not found: {INPUT_JSONL_PATH}"); sys.exit(1)
    except Exception as e: main_logger.error(f"Error reading input file {INPUT_JSONL_PATH}: {e}"); sys.exit(1)
    
    if not prompts_data: main_logger.error("No valid prompts found. Exiting."); sys.exit(0)
    main_logger.info(f"Read {len(prompts_data)} valid prompts.")

    limit = (5 * NUM_THREADS) if test_flag else (500 if CUT else None)
    if limit: main_logger.info(f"{'TEST' if test_flag else 'CUT'} MODE: Limiting to approx {limit} total prompts.")
    
    prompts_to_process_with_indices = [{"original_index": entry["original_index"], **entry["data"]} for entry in (prompts_data[:limit] if limit else prompts_data)]
    num_actual_prompts = len(prompts_to_process_with_indices)
    if num_actual_prompts == 0: main_logger.error("No prompts to process after filtering. Exiting."); sys.exit(0)
    main_logger.info(f"Total prompts to process across all threads: {num_actual_prompts}")
    run_mode_desc = f"TEST MODE - First ~{limit}" if test_flag else (f"CUT MODE - First {limit}" if CUT else "FULL RUN")
    main_logger.info(f"\n{'=' * 10} Starting Interleaved Evaluation ({run_mode_desc} - {TASK_NAME}) {'=' * 10}")

    chunk_size = ceil(num_actual_prompts / NUM_THREADS)
    prompt_chunks_with_original_indices = []
    for i in range(NUM_THREADS):
        start_idx = i * chunk_size; end_idx = min((i + 1) * chunk_size, num_actual_prompts)
        if start_idx < end_idx: chunk = [(entry["original_index"], entry) for entry in prompts_to_process_with_indices[start_idx:end_idx]]; prompt_chunks_with_original_indices.append(chunk)
        else: prompt_chunks_with_original_indices.append([])

    active_model_name = list(MODEL_CONFIGS.keys())[0]
    model_config_data = MODEL_CONFIGS[active_model_name]
    global_settings_for_worker = {'MULTI_EVAL': MULTI_EVAL, 'WRITE_OUTPUT': WRITE_OUTPUT, 'GENERATION_MAX_TOKENS': GENERATION_MAX_TOKENS, 'GENERATION_TEMPERATURE': GENERATION_TEMPERATURE, 'LOW_SIM_THRESHOLD': LOW_SIM_THRESHOLD, 'HIGH_SIM_THRESHOLD': HIGH_SIM_THRESHOLD, 'API_DELAY_SECONDS': API_DELAY_SECONDS, 'PROMPT_DELAY_SECONDS': PROMPT_DELAY_SECONDS}

    aggregated_model_stats = {'successful_preds': 0, 'total_similarity': 0.0, 'successful_rougeL': 0, 'total_rougeL': 0.0, 'successful_rouge1': 0, 'total_rouge1': 0.0, 'successful_bleu': 0, 'total_bleu': 0.0, 'successful_meteor': 0, 'total_meteor': 0.0, 'errors': 0, 'total': 0, 'true_label_errors_count':0}
    aggregated_domain_stats = {}; aggregated_model_results_list = []; all_combined_json_lines = []; all_best_candidate_lines = []; all_confusing_prompt_indices = set()

    with ThreadPoolExecutor(max_workers=NUM_THREADS, thread_name_prefix='WorkerThread') as executor: # Thread name prefix matches logger name for worker
        futures = []
        for i in range(NUM_THREADS):
            if not prompt_chunks_with_original_indices[i]:
                main_logger.info(f"Skipping submission for Thread {i+1} as it has no prompts.")
                continue
            futures.append(executor.submit(process_prompts_worker, i + 1, prompt_chunks_with_original_indices[i], active_model_name, model_config_data, sbert_model_global, rouge_scorer_instance_global, meteor_available_global, global_settings_for_worker, device))

        for future in as_completed(futures):
            try:
                res_model_stats, res_domain_stats, res_model_results, res_combined_lines, res_best_cand_lines, res_confusing_indices = future.result()
                for key, val in res_model_stats.items(): aggregated_model_stats[key] = aggregated_model_stats.get(key, 0) + val
                for domain, stats_part in res_domain_stats.items():
                    if domain not in aggregated_domain_stats: aggregated_domain_stats[domain] = deepcopy(stats_part)
                    else:
                        for key, value in stats_part.items():
                            if isinstance(value, (int, float)): aggregated_domain_stats[domain][key] = aggregated_domain_stats[domain].get(key, 0) + value
                aggregated_model_results_list.extend(res_model_results); all_combined_json_lines.extend(res_combined_lines); all_best_candidate_lines.extend(res_best_cand_lines); all_confusing_prompt_indices.update(res_confusing_indices)
            except Exception as exc: main_logger.error(f'A worker thread generated an exception: {exc}', exc_info=True)
            else: main_logger.info(f'A worker thread completed and results aggregated.')
    
    aggregated_model_results_list.sort(key=lambda x: x.get('prompt_index', float('inf')))
    all_combined_json_lines.sort(key=lambda x: x.get('prompt_index', float('inf')))
    all_best_candidate_lines.sort(key=lambda x: x.get('prompt_index', float('inf')))

    if WRITE_OUTPUT and all_best_candidate_lines:
        best_cand_filename = os.path.join(OUTPUT_DIR, f"best_candidates_{active_model_name.replace('@','_')}_{TASK_NAME}{run_identifier}.jsonl")
        try:
            with open(best_cand_filename, 'w', encoding='utf-8') as f_bc:
                for line_data in all_best_candidate_lines: f_bc.write(json.dumps(line_data, ensure_ascii=False) + '\n')
            main_logger.info(f"Saved aggregated best candidates to {best_cand_filename}")
        except IOError as e: main_logger.error(f"Could not write {best_cand_filename}: {e}")
    if all_combined_json_lines:
        try:
            with open(combined_output_filename, 'w', encoding='utf-8') as f_comb:
                for line_data in all_combined_json_lines: f_comb.write(json.dumps(line_data, ensure_ascii=False) + '\n')
            main_logger.info(f"Saved aggregated combined input/output/match to {combined_output_filename}")
        except (IOError, TypeError) as e: main_logger.error(f"Could not write/serialize {combined_output_filename}: {e}")

    main_logger.info(f"\n{'=' * 10} Final Aggregated Domain Metrics ({run_mode_desc}) {'=' * 10}")
    log_domain_metrics(aggregated_domain_stats, main_logger, meteor_available_global, active_model_count=1)

    main_logger.info(f"\n{'=' * 10} Evaluation Finished ({run_mode_desc}) - Final Metrics & Results {'=' * 10}")
    if all_confusing_prompt_indices: main_logger.info(f"Identified {len(all_confusing_prompt_indices)} confusing prompts (0-indexed): {sorted(list(all_confusing_prompt_indices))}")
    else: main_logger.info("No confusing prompts identified.")
    
    overall_summary_list = []
    model_name = active_model_name; stats = aggregated_model_stats; results = aggregated_model_results_list
    safe_model_name = re.sub(r'[\\/*?:"<>|@]', '_', model_name)
    eval_output_filename = os.path.join(OUTPUT_DIR, f"eval_{safe_model_name}_{TASK_NAME}{run_identifier}.csv")
    main_logger.info(f"\n--- Final Aggregated Results for: {model_name} ---")

    if results:
        main_logger.info(f"  Writing {len(results)} aggregated results to {eval_output_filename}...")
        try:
            with open(eval_output_filename, 'w', newline='', encoding='utf-8-sig') as f:
                fieldnames = ["prompt_index", "user", "prompt_preview", "true_label_preview", "prediction", "best_sbert_score", "best_rougeL_score", "best_rouge1_score", "best_bleu_score", "best_meteor_score", "api_error"]
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore'); writer.writeheader(); writer.writerows(results)
        except IOError as e: main_logger.error(f"Error writing {eval_output_filename}: {e}")
    else: main_logger.warning(f"  No results recorded for {model_name}.")

    total_processed = stats['total']
    true_label_errs = stats['true_label_errors_count']
    api_or_pred_errors = stats['errors'] - true_label_errs # API/Eval errors, excluding true label processing ones
    successful_sbert_preds = stats['successful_preds']
    
    # Total prompts attempted by API (total - true label errors)
    total_api_attempts = total_processed - true_label_errs

    avg_best_sbert_score = (stats['total_similarity'] / successful_sbert_preds) if successful_sbert_preds > 0 else 0.0
    # Success rate based on prompts that went to API
    success_rate_sbert = (successful_sbert_preds / total_api_attempts) * 100 if total_api_attempts > 0 else 0.0

    avg_best_rougeL_score = (stats['total_rougeL'] / stats['successful_rougeL']) if stats.get('successful_rougeL',0) > 0 else 0.0
    avg_best_rouge1_score = (stats['total_rouge1'] / stats['successful_rouge1']) if stats.get('successful_rouge1',0) > 0 else 0.0
    avg_best_bleu_score = (stats['total_bleu'] / stats['successful_bleu']) if stats.get('successful_bleu',0) > 0 else 0.0
    avg_best_meteor_score = (stats['total_meteor'] / stats['successful_meteor']) if meteor_available_global and stats.get('successful_meteor', 0) > 0 else 0.0

    filtered_results = [r for r in results if r.get('prompt_index', -1) not in all_confusing_prompt_indices and not (isinstance(r.get('prediction'), str) and r['prediction'].startswith("Error:")) and not r.get('api_error')]
    successful_sbert_preds_filtered = sum(1 for r in filtered_results if r.get('best_sbert_score', 'N/A') != 'N/A' and isinstance(r.get('best_sbert_score'), str))
    total_similarity_filtered = sum(float(r['best_sbert_score']) for r in filtered_results if r.get('best_sbert_score', 'N/A') != 'N/A' and isinstance(r.get('best_sbert_score'), str))
    avg_best_sbert_score_filtered = (total_similarity_filtered / successful_sbert_preds_filtered) if successful_sbert_preds_filtered > 0 else 0.0
    
    # For filtered success rate, need to determine how many non-confusing prompts were attempted by API
    prompts_for_filtered_rate = total_api_attempts - len(all_confusing_prompt_indices.intersection(set(r['prompt_index'] for r in results if not (r.get('api_error') or (isinstance(r.get('prediction'), str) and r['prediction'].startswith("Error:")) )))) # A bit complex, simplifies to total non-confusing API attempts
    success_rate_sbert_filtered = (successful_sbert_preds_filtered / prompts_for_filtered_rate) * 100 if prompts_for_filtered_rate > 0 else 0.0


    avg_best_rougeL_score_filtered, avg_best_rouge1_score_filtered, avg_best_bleu_score_filtered, avg_best_meteor_score_filtered = 0.0, 0.0, 0.0, 0.0
    if MULTI_EVAL:
        s_rl_f = sum(1 for r in filtered_results if r.get('best_rougeL_score', 'N/A') != 'N/A'); t_rl_f = sum(float(r['best_rougeL_score']) for r in filtered_results if r.get('best_rougeL_score', 'N/A') != 'N/A')
        avg_best_rougeL_score_filtered = (t_rl_f / s_rl_f) if s_rl_f > 0 else 0.0
        s_r1_f = sum(1 for r in filtered_results if r.get('best_rouge1_score', 'N/A') != 'N/A'); t_r1_f = sum(float(r['best_rouge1_score']) for r in filtered_results if r.get('best_rouge1_score', 'N/A') != 'N/A')
        avg_best_rouge1_score_filtered = (t_r1_f / s_r1_f) if s_r1_f > 0 else 0.0
        s_b_f = sum(1 for r in filtered_results if r.get('best_bleu_score', 'N/A') != 'N/A'); t_b_f = sum(float(r['best_bleu_score']) for r in filtered_results if r.get('best_bleu_score', 'N/A') != 'N/A')
        avg_best_bleu_score_filtered = (t_b_f / s_b_f) if s_b_f > 0 else 0.0
        if meteor_available_global:
            s_m_f = sum(1 for r in filtered_results if r.get('best_meteor_score', 'N/A') != 'N/A'); t_m_f = sum(float(r['best_meteor_score']) for r in filtered_results if r.get('best_meteor_score', 'N/A') != 'N/A')
            avg_best_meteor_score_filtered = (t_m_f / s_m_f) if s_m_f > 0 else 0.0

    summary_dict = {
        "model_name": model_name, "task": TASK_NAME, "total_prompts_dataset": total_processed,
        "prompts_with_true_label_errors": true_label_errs,
        "prompts_attempted_by_api": total_api_attempts,
        "successful_api_plus_evals": successful_sbert_preds, 
        "api_eval_errors_on_attempted": api_or_pred_errors,
        "avg_best_sbert_score": f"{avg_best_sbert_score:.6f}", 
        "overall_success_rate_of_api_attempts": f"{success_rate_sbert:.2f}%",
        "avg_best_rougeL_score": f"{avg_best_rougeL_score:.6f}" if MULTI_EVAL else "N/A",
        "avg_best_rouge1_score": f"{avg_best_rouge1_score:.6f}" if MULTI_EVAL else "N/A",
        "avg_best_bleu_score": f"{avg_best_bleu_score:.6f}" if MULTI_EVAL else "N/A",
        "avg_best_meteor_score": f"{avg_best_meteor_score:.6f}" if MULTI_EVAL and meteor_available_global else "N/A",
        "confusing_samples_excluded_count": len(all_confusing_prompt_indices),
        "successful_evals_filtered": successful_sbert_preds_filtered,
        "avg_best_sbert_score_filtered": f"{avg_best_sbert_score_filtered:.6f}",
        "overall_success_rate_filtered": f"{success_rate_sbert_filtered:.2f}%",
        "avg_best_rougeL_score_filtered": f"{avg_best_rougeL_score_filtered:.6f}" if MULTI_EVAL else "N/A",
        "avg_best_rouge1_score_filtered": f"{avg_best_rouge1_score_filtered:.6f}" if MULTI_EVAL else "N/A",
        "avg_best_bleu_score_filtered": f"{avg_best_bleu_score_filtered:.6f}" if MULTI_EVAL else "N/A",
        "avg_best_meteor_score_filtered": f"{avg_best_meteor_score_filtered:.6f}" if MULTI_EVAL and meteor_available_global else "N/A",
        "output_file": eval_output_filename
    }
    overall_summary_list.append(summary_dict)

    main_logger.info(f"  Final Summary for {model_name} ({TASK_NAME}):")
    main_logger.info(f"    Total Prompts in Dataset Subset: {summary_dict['total_prompts_dataset']}")
    main_logger.info(f"    Prompts with True Label Processing Errors (skipped before API): {summary_dict['prompts_with_true_label_errors']}")
    main_logger.info(f"    Prompts Attempted by API: {summary_dict['prompts_attempted_by_api']}")
    main_logger.info(f"    Successful API calls + Evaluations: {summary_dict['successful_api_plus_evals']}")
    main_logger.info(f"    API/Post-API Eval Errors (on attempted prompts): {summary_dict['api_eval_errors_on_attempted']}")
    main_logger.info(f"    Overall Success Rate (Successful Evals / Prompts Attempted by API): {summary_dict['overall_success_rate_of_api_attempts']}")
    main_logger.info(f"    --- SBERT Similarity (Avg Best) ---")
    main_logger.info(f"    Overall: {summary_dict['avg_best_sbert_score']}")
    main_logger.info(f"    Filtered: {summary_dict['avg_best_sbert_score_filtered']} (on {summary_dict['successful_evals_filtered']} successful non-confusing evals)")
    if MULTI_EVAL:
        main_logger.info(f"    --- ROUGE-L F1 --- Overall: {summary_dict['avg_best_rougeL_score']}, Filtered: {summary_dict['avg_best_rougeL_score_filtered']}")
        main_logger.info(f"    --- ROUGE-1 F1 --- Overall: {summary_dict['avg_best_rouge1_score']}, Filtered: {summary_dict['avg_best_rouge1_score_filtered']}")
        main_logger.info(f"    --- BLEU Score --- Overall: {summary_dict['avg_best_bleu_score']}, Filtered: {summary_dict['avg_best_bleu_score_filtered']}")
        if meteor_available_global: main_logger.info(f"    --- METEOR Score --- Overall: {summary_dict['avg_best_meteor_score']}, Filtered: {summary_dict['avg_best_meteor_score_filtered']}")
    main_logger.info(f"    Per-Model Results File: {summary_dict['output_file']}")
    main_logger.info("-" * 30)

    main_logger.info(f"\n{'=' * 20} Final Evaluation Summary (Claude){' [' + run_mode_desc + ']'} {'=' * 20}")
    for summary_item in overall_summary_list:
        log_str = f"Model: {summary_item['model_name']:<30} | Task: {summary_item['task']:<20} | API Success: {summary_item['overall_success_rate_of_api_attempts']:>7s} | Avg SBERT: {summary_item['avg_best_sbert_score']}"
        if MULTI_EVAL: log_str += f" | R-L: {summary_item['avg_best_rougeL_score']} | R-1: {summary_item['avg_best_rouge1_score']} | BLEU: {summary_item['avg_best_bleu_score']}"
        if meteor_available_global: log_str += f" | METEOR: {summary_item['avg_best_meteor_score']}"
        main_logger.info(log_str)

    summary_filename = os.path.join(OUTPUT_DIR, f"evaluation_summary_{TASK_NAME}{run_identifier}.csv")
    main_logger.info(f"\nSaving overall summary to {summary_filename}...")
    try:
        with open(summary_filename, 'w', newline='', encoding='utf-8-sig') as f:
            if overall_summary_list:
                writer = csv.DictWriter(f, fieldnames=list(overall_summary_list[0].keys())); writer.writeheader(); writer.writerows(overall_summary_list)
            else: f.write("No summary data.\n")
        main_logger.info(f"Overall summary saved.")
    except (IOError, Exception) as e: main_logger.error(f"Error writing summary file {summary_filename}: {e}")

    main_logger.info(f"\nMultithreaded evaluation ({run_mode_desc}) complete.")
    main_logger.info(f"Main log file: {main_log_filename}")
    main_logger.info(f"Individual thread logs are in: {THREAD_LOG_DIR}")

if __name__ == "__main__":
    nltk_punkt_available = False; nltk_wordnet_omw_available = False
    if MULTI_EVAL_ENABLED: # Check NLTK only if MULTI_EVAL could be enabled
        try: nltk.data.find('tokenizers/punkt'); nltk_punkt_available = True; main_logger.info("NLTK 'punkt' found.")
        except LookupError: main_logger.info("NLTK 'punkt' not found. Attempting download..."); nltk.download('punkt', quiet=True); nltk_punkt_available = True; main_logger.info("'punkt' downloaded.")
        except Exception as e: main_logger.error(f"NLTK 'punkt' check/download failed: {e}")
        try: nltk.data.find('corpora/wordnet'); nltk.data.find('corpora/omw-1.4'); nltk_wordnet_omw_available = True; main_logger.info("NLTK 'wordnet'/'omw-1.4' found.")
        except LookupError: main_logger.info("NLTK 'wordnet'/'omw-1.4' not found. Downloading..."); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True); nltk_wordnet_omw_available = True; main_logger.info("'wordnet'/'omw-1.4' downloaded.")
        except Exception as e: main_logger.warning(f"NLTK 'wordnet'/'omw-1.4' check/download failed: {e}")

        if not nltk_punkt_available: MULTI_EVAL = False; meteor_available_global = False; main_logger.warning("MULTI_EVAL disabled: 'punkt' unavailable.")
        elif not nltk_wordnet_omw_available: meteor_available_global = Fa_lse; main_logger.warning("METEOR disabled: 'wordnet'/'omw-1.4' unavailable.")
        else: meteor_available_global = True # MULTI_EVAL remains as per MULTI_EVAL_ENABLED
    else: # MULTI_EVAL_ENABLED was false from the start
        MULTI_EVAL = False; meteor_available_global = False
        main_logger.info("MULTI_EVAL was initially disabled or pre-requisites (e.g. rouge_score, nltk) not met.")
    main()