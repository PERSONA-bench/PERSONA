# ---------------- evaluate_models_unified_fixed_multithreaded.py ----------------
"""
Evaluates LLMs on EITHER Personalized Conversation Generation (Task 3.4)
OR Personalized Follow-up Text Generation (Task 3.3 - predicting body text).

MODIFIED FOR MULTITHREADING.
Controlled by FLAG_33 and FLAG_34. Includes multiple evaluation metrics and flags.
Saves detailed input, best model output, and the most similar true label per prompt
to a combined JSONL file each run. Includes enhanced error logging.
Also calculates and logs metrics per domain (subreddit).
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
import threading
import math # For ceiling division

# --- Import API Libraries ---
try:
    from openai import AzureOpenAI, APIError, RateLimitError
except ImportError:
    print("ERROR: 'openai' library not found. Please install it: pip install openai")
    sys.exit(1)

try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
except ImportError:
    print("WARNING: 'google-generativeai' library not found. Google models will not be available.")
    genai = None
    google_exceptions = None

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
FLAG_33 = True
FLAG_34 = not FLAG_33
test_flag = False
CUT = False
MULTI_EVAL = MULTI_EVAL_ENABLED
WRITE_OUTPUT = True
if FLAG_33 == FLAG_34: logging.error("Exactly one of FLAG_33 or FLAG_34 must be True."); sys.exit(1) # Logger not set yet
TASK_NAME = "Predict_Body_Text_3.3" if FLAG_33 else "Generate_Conversation_3.4"
# --- End Flag Configuration ---

# --- General Configuration ---
if FLAG_33:
    INPUT_JSONL_PATH = "WithPseudoRandomConversationPrompts_BodyPrediction.jsonl"
    BASE_OUTPUT_DIR = f"MultiThreadGPT4.1-3.3/eval_random_results_{TASK_NAME}" # MODIFIED
    GENERATION_MAX_TOKENS = 256
    NUM_CANDIDATES = 10
    GENERATION_TEMPERATURE = 0.7
else: # FLAG_34 is True
    INPUT_JSONL_PATH = 'prompts_generate_conversation_final.jsonl'
    BASE_OUTPUT_DIR = f"MultiThreadGPT4.1-3.3/eval_results_{TASK_NAME}" # MODIFIED
    GENERATION_MAX_TOKENS = 2500
    NUM_CANDIDATES = 10
    GENERATION_TEMPERATURE = 0.8
    # MULTI_EVAL = False # This was in original, kept for consistency

API_DELAY_SECONDS = 1
PROMPT_DELAY_SECONDS = 0
MAX_RETRIES = 3 # Increased for 429 handling
RETRY_DELAY_SECONDS = 3
RETRY_DELAY_429_SECONDS = 15 # Specific delay for 429
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'
LOW_SIM_THRESHOLD = 0.1
HIGH_SIM_THRESHOLD = 1.0
DEFAULT_API_VERSION = ""
REQUIRED_NEW_API_VERSION = "2024-12-01-preview"

# --- Model Definitions ---
MODEL_CONFIGS = {
    "gpt-4.1": {"api_type": "azure", "api_key": "", # Replace with your key
                           "azure_endpoint": "https://eastus2instancefranck.openai.azure.com/", # Replace with your endpoint
                           "api_version": DEFAULT_API_VERSION, "deployment_name": "gpt-4.1"},
}
# --- End Model Definitions ---

# --- Global SBERT Model and Device (Initialized in main_setup) ---
sbert_model = None
device = None
rouge_scorer_instance = None
meteor_available_global = True # Renamed to avoid conflict

# --- Logging Setup (Main logger, threads will have their own) ---
main_log_filename = f"FINAL-EVAL-MAIN-{dt.now():%Y%m%d_%H%M%S}.log" # Main log for overall process
main_logger_formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s') # Added threadName
main_logger = logging.getLogger("main_eval_logger")
main_logger.setLevel(logging.INFO)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
if main_logger.hasHandlers(): main_logger.handlers.clear()

# Ensure base output directory exists for main log
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
main_file_handler_path = os.path.join(BASE_OUTPUT_DIR, main_log_filename)
main_file_handler = logging.FileHandler(main_file_handler_path, encoding='utf-8')
main_file_handler.setFormatter(main_logger_formatter)
main_logger.addHandler(main_file_handler)

main_console_handler = logging.StreamHandler(sys.stdout)
main_console_handler.setFormatter(main_logger_formatter)
main_logger.addHandler(main_console_handler)
# --- End Main Logging Setup ---

# --- Helper Functions (Mostly unchanged, ensure thread safety if they modify shared state not passed as args) ---
def get_thread_concatenated_text(thread_nodes):
    if not thread_nodes or not isinstance(thread_nodes, list): return ""
    return " ".join([node.get("body", "") for node in thread_nodes if
                     isinstance(node, dict) and isinstance(node.get("body"), str) and node.get("body", "").strip()])

def get_thread_embedding(thread_nodes, local_sbert_model, local_device): # Pass sbert_model and device
    if not thread_nodes or not isinstance(thread_nodes, list): return None
    bodies = [node.get("body", "") for node in thread_nodes if
              isinstance(node, dict) and isinstance(node.get("body"), str) and node.get("body", "").strip()]
    if not bodies: return None
    try:
        with torch.no_grad(): # Important for inference
            embeddings = local_sbert_model.encode(bodies, convert_to_tensor=True, device=local_device,
                                                  show_progress_bar=False, batch_size=128) # Use local device
            mean_embedding = torch.mean(embeddings.to('cpu'), dim=0) # SBERT embeddings are often moved to CPU after encoding
        return mean_embedding
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "MPS backend out of memory" in str(e):
            main_logger.error(f"OOM Error during SBERT encoding: {e}.") # Use main_logger or pass logger
        else:
            main_logger.error(f"Runtime error during SBERT encoding/mean calculation: {e}")
        return None
    except Exception as e:
        main_logger.error(f"Error during SBERT encoding/mean calculation: {e}");
        return None

# --- API Call Functions (Modified for 429 retry) ---
def call_azure_openai_with_retry(client, deployment_name, prompt_text, model_config, num_candidates, max_gen_tokens,
                                 temperature, thread_logger): # Pass logger
    candidate_responses = []
    error_message = None
    api_version_str = model_config.get('api_version', DEFAULT_API_VERSION)
    requires_new_params = api_version_str >= REQUIRED_NEW_API_VERSION
    request_params = {"model": deployment_name, "messages": [{"role": "user", "content": prompt_text}],
                      "n": num_candidates, "temperature": temperature}
    if requires_new_params:
        request_params["max_tokens"] = max_gen_tokens
    else:
        request_params["max_tokens"] = max_gen_tokens

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(**request_params)
            finish_reason = "Unknown"
            if response.choices:
                candidate_responses = [choice.message.content or "" for choice in response.choices]
                if response.choices[0].finish_reason: finish_reason = response.choices[0].finish_reason
            else:
                if hasattr(response, 'prompt_filter_results') and response.prompt_filter_results:
                    filter_info = response.prompt_filter_results[0].get('content_filter_results', {})
                    finish_reason = f"Prompt Content Filtered ({filter_info})"
                elif hasattr(response, 'choices') and response.choices is not None and len(response.choices) == 0:
                    finish_reason = "Empty Choices Received"
                error_message = f"No choices received. Finish Reason: {finish_reason}"
                thread_logger.warning(f"Model {deployment_name} returned no choices. Finish Reason: {finish_reason}")
                candidate_responses = []
                break
            if finish_reason != 'stop' and not error_message:
                thread_logger.warning(
                    f"Model {deployment_name} finished with reason: '{finish_reason}'. Output might be truncated or incomplete.")
            error_message = None
            break
        except RateLimitError as e:
            error_message = f"RateLimitError: {e}"
            if attempt < MAX_RETRIES:
                thread_logger.warning(
                    f"Attempt {attempt + 1} RateLimitError ({deployment_name}). Retrying in {RETRY_DELAY_429_SECONDS}s...")
                time.sleep(RETRY_DELAY_429_SECONDS) # Specific delay for 429
                continue # Retry immediately after sleep
            else:
                thread_logger.error(f"Final RateLimitError after {attempt + 1} attempts ({deployment_name}): {error_message}")
                break
        except APIError as e:
            err_code = getattr(e, 'status_code', 'N/A')
            err_msg = getattr(e, 'message', str(e))
            err_type = getattr(e, 'type', 'N/A')
            error_message = f"APIError: Status={err_code}, Type={err_type}, Message={err_msg[:150]}..."
            # ... (original error handling for APIError) ...
            if 'DeploymentNotFound' in str(e) or (hasattr(e, 'code') and e.code == 'DeploymentNotFound'): break
            elif 'content_management_policy' in str(e).lower() or (hasattr(e, 'code') and e.code == 'content_filter'): error_message = f"Content Filter Error ({err_code}): {err_msg[:100]}..."; break
            elif 'context_length_exceeded' in str(e).lower() or (hasattr(e, 'code') and e.code == 'context_length_exceeded'): break
            elif err_code in [500, 503, 408] and attempt < MAX_RETRIES: pass # Retryable server-side issues (non-429)
            else: break # Non-retryable API error
        except Exception as e:
            error_message = f"Unexpected Error during API call: {type(e).__name__} - {str(e)[:150]}..."
            break

        if error_message and attempt < MAX_RETRIES: # For non-RateLimit errors that are retryable
            thread_logger.warning(
                f"Attempt {attempt + 1} fail({deployment_name}): {error_message}. Retrying in {RETRY_DELAY_SECONDS}s...")
            time.sleep(RETRY_DELAY_SECONDS)
        elif error_message:
            thread_logger.error(f"Final API Error after {attempt + 1} attempts ({deployment_name}): {error_message}")

    return candidate_responses, error_message


def call_google_vertex_with_retry(client_sdk, model_id, prompt_text, num_candidates, max_gen_tokens, temperature, thread_logger): # Pass logger
    if genai is None: return [], "Google GenAI SDK not imported."
    thread_logger.warning("Google Vertex AI call function is not implemented.")
    return [], "Google Call Function Not Implemented"


def setup_thread_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s')
    
    # Prevent adding multiple handlers if re-called (though ideally called once per thread)
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Optional: Add console output for thread logs too, or rely on main logger for console
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)
    return logger


def log_domain_metrics(current_domain_stats, logger_instance, meteor_is_available_flag, num_models): # Pass num_models
    if not current_domain_stats:
        logger_instance.info("  No domain statistics collected yet.")
        return

    active_model_count = num_models

    for domain, stats in sorted(current_domain_stats.items()):
        logger_instance.info(f"  --- Domain: {domain} ---")
        # ... (rest of the original log_domain_metrics function, ensure logger_instance is used) ...
        total_model_attempts = stats.get('total_model_attempts_on_prompts', 0)
        logger_instance.info(
            f"    Total Model Attempts (prompts_in_domain * models_attempted_per_prompt): {total_model_attempts}")
        true_label_errors = stats.get('prompts_with_true_label_errors', 0)
        if true_label_errors > 0:
            logger_instance.info(
                f"    Prompts with True Label Processing Errors: {true_label_errors} (each contributed {active_model_count} errors to 'Post-API Eval Errors' below and {active_model_count} to 'Total Model Attempts' above)")
        api_errors = stats.get('api_errors_count', 0)
        post_api_eval_errors = stats.get('post_api_eval_errors_count', 0)
        logger_instance.info(f"    API Call Errors (model failed to respond): {api_errors}")
        logger_instance.info(
            f"    Post-API Eval Errors (e.g., no valid candidates, true_label_error effects): {post_api_eval_errors}")
        sbert_evals = stats.get('successful_sbert_evals', 0)
        avg_sbert = (stats.get('sum_sbert_scores', 0.0) / sbert_evals) if sbert_evals > 0 else 0.0
        logger_instance.info(f"    Avg SBERT Score: {avg_sbert:.4f} (over {sbert_evals} successful evals)")
        if FLAG_33 and MULTI_EVAL:
            rl_evals = stats.get('successful_rougeL_evals', 0)
            avg_rl = (stats.get('sum_rougeL_scores', 0.0) / rl_evals) if rl_evals > 0 else 0.0
            logger_instance.info(f"    Avg ROUGE-L F1 Score: {avg_rl:.4f} (over {rl_evals} evals)")
            # ... (add ROUGE1, BLEU, METEOR logging as in original) ...
            r1_evals = stats.get('successful_rouge1_evals', 0)
            avg_r1 = (stats.get('sum_rouge1_scores', 0.0) / r1_evals) if r1_evals > 0 else 0.0
            logger_instance.info(f"    Avg ROUGE-1 F1 Score: {avg_r1:.4f} (over {r1_evals} evals)")

            bleu_evals = stats.get('successful_bleu_evals', 0)
            avg_bleu = (stats.get('sum_bleu_scores', 0.0) / bleu_evals) if bleu_evals > 0 else 0.0
            logger_instance.info(f"    Avg BLEU Score: {avg_bleu:.4f} (over {bleu_evals} evals)")

            if meteor_is_available_flag:
                meteor_evals = stats.get('successful_meteor_evals', 0)
                avg_meteor = (stats.get('sum_meteor_scores', 0.0) / meteor_evals) if meteor_evals > 0 else 0.0
                logger_instance.info(f"    Avg METEOR Score: {avg_meteor:.4f} (over {meteor_evals} evals)")
        logger_instance.info("-" * 20)


# --- Thread Processing Function ---
def process_prompts_thread(thread_id, prompts_chunk, output_dir, local_sbert_model, local_device, local_rouge_scorer, local_meteor_available):
    thread_name = f"Thread-{thread_id}"
    thread_log_filename = os.path.join(output_dir, f"thread_{thread_id}_eval_{dt.now():%Y%m%d_%H%M%S}.log")
    thread_logger = setup_thread_logger(thread_name, thread_log_filename)
    thread_logger.info(f"Starting processing for {len(prompts_chunk)} prompts.")

    # These will be returned by the thread
    thread_model_results = {name: [] for name in MODEL_CONFIGS.keys()}
    thread_model_stats = {name: {'successful_preds': 0, 'total_similarity': 0.0,
                                 'successful_rougeL': 0, 'total_rougeL': 0.0,
                                 'successful_rouge1': 0, 'total_rouge1': 0.0,
                                 'successful_bleu': 0, 'total_bleu': 0.0,
                                 'successful_meteor': 0, 'total_meteor': 0.0,
                                 'errors': 0, 'total': 0} for name in MODEL_CONFIGS.keys()}
    thread_domain_stats = {}
    thread_confusing_prompt_indices = set() # Store original indices

    # Thread-specific output files
    timestamp_str = dt.now().strftime("%Y%m%d_%H%M%S")
    run_identifier_thread = f"{'_test' if test_flag else ('_cut500' if CUT else '')}_thread{thread_id}"
    
    thread_combined_output_filename = os.path.join(output_dir,
                                            f"input_best_output_best_match_{TASK_NAME}{run_identifier_thread}_{timestamp_str}.jsonl")
    thread_combined_output_file = None
    try:
        thread_combined_output_file = open(thread_combined_output_filename, 'w', encoding='utf-8')
        thread_logger.info(f"Saving combined input/output for thread to: {thread_combined_output_filename}")
    except IOError as e:
        thread_logger.error(f"Could not open thread combined output file {thread_combined_output_filename}: {e}")
        # Decide if thread should exit or continue without this file

    thread_best_candidate_files = {}
    if WRITE_OUTPUT:
        for model_name_key in MODEL_CONFIGS.keys():
            safe_model_name = re.sub(r'[\\/*?:"<>|]', '_', model_name_key)
            filename = os.path.join(output_dir, f"best_candidates_{safe_model_name}_{TASK_NAME}{run_identifier_thread}.jsonl")
            try:
                thread_best_candidate_files[model_name_key] = open(filename, 'w', encoding='utf-8')
            except IOError as e:
                thread_logger.error(f"Could not open best candidate file {filename} for thread {thread_id}: {e}")
                thread_best_candidate_files[model_name_key] = None


    # Initialize API clients per thread
    thread_initialized_clients = {}
    for model_name_key, config_val in MODEL_CONFIGS.items():
        if config_val["api_type"] == "azure":
            try:
                api_client = AzureOpenAI(api_key=config_val["api_key"], azure_endpoint=config_val["azure_endpoint"],
                                         api_version=config_val["api_version"])
                thread_initialized_clients[model_name_key] = api_client
            except Exception as e:
                thread_logger.error(f"Azure Client Init Error for {model_name_key} in thread {thread_id}: {e}")
                # This model will likely fail all its calls in this thread
        # Add other API types if necessary


    final_prompt_index_in_chunk = len(prompts_chunk) - 1
    active_models_list = list(MODEL_CONFIGS.keys()) # Ensure this is defined

    for i_chunk, prompt_entry_with_original_idx in enumerate(prompts_chunk):
        original_prompt_idx = prompt_entry_with_original_idx['original_idx']
        prompt_entry = prompt_entry_with_original_idx['data']

        user = prompt_entry.get("user", "Unknown")
        current_prompt_text = prompt_entry["prompt"]
        true_label_input = prompt_entry["true_label"]
        target_post_data = prompt_entry.get("target_post")
        target_post_id = target_post_data.get("id", f"Prompt_{original_prompt_idx + 1}") if isinstance(target_post_data, dict) else f"Prompt_{original_prompt_idx + 1}"
        current_subreddit = prompt_entry.get("subreddit", "Unknown_Subreddit")

        thread_logger.info(
            f"\n--- Processing Prompt (Original Idx {original_prompt_idx + 1}, Chunk Idx {i_chunk + 1}/{len(prompts_chunk)}) User: {user} ---")
        thread_logger.info(f"  Subreddit: {current_subreddit}")

        if current_subreddit not in thread_domain_stats:
            thread_domain_stats[current_subreddit] = {
                'total_model_attempts_on_prompts': 0,
                'prompts_with_true_label_errors': 0,
                'successful_sbert_evals': 0, 'sum_sbert_scores': 0.0,
                'successful_rougeL_evals': 0, 'sum_rougeL_scores': 0.0,
                'successful_rouge1_evals': 0, 'sum_rouge1_scores': 0.0,
                'successful_bleu_evals': 0, 'sum_bleu_scores': 0.0,
                'successful_meteor_evals': 0, 'sum_meteor_scores': 0.0,
                'api_errors_count': 0,
                'post_api_eval_errors_count': 0
            }

        prompt_results_for_combined_json = {
            "prompt_index": original_prompt_idx, "user": user, "task": TASK_NAME, "input_prompt": current_prompt_text,
            "true_label": true_label_input, "target_post": target_post_data if target_post_data else None,
            "subreddit": current_subreddit,
            "model_outputs": {}
        }

        true_label_embeddings = None
        true_label_text_single = ""
        true_label_tokenized_list = []
        true_label_tokenized_for_meteor = []
        valid_true_labels_data = []
        processing_error_occurred = False

        try:
            # Ensure SBERT model and device are passed correctly
            if FLAG_33:
                if not isinstance(true_label_input, str) or not true_label_input.strip(): raise ValueError("True label text is empty or not a string.")
                true_label_text_single = true_label_input
                with torch.no_grad(): # Ensure no_grad for SBERT encoding
                    true_label_embeddings = local_sbert_model.encode(true_label_text_single, convert_to_tensor=True, device=local_device)
                valid_true_labels_data.append({"original_index": 0, "thread": true_label_text_single, "embedding": true_label_embeddings})
                if MULTI_EVAL and local_rouge_scorer: # Check for local_rouge_scorer
                    true_label_tokenized_list = [nltk.word_tokenize(true_label_text_single.lower())]
                    if not true_label_tokenized_list[0]: raise ValueError("True label tokenization for BLEU resulted in empty list.")
                    if local_meteor_available:
                        tokenized_ref = nltk.word_tokenize(true_label_text_single.lower())
                        if not tokenized_ref: raise ValueError("True label tokenization for METEOR resulted in empty list.")
                        true_label_tokenized_for_meteor = [tokenized_ref]
            elif FLAG_34:
                # ... (original FLAG_34 true label processing, using local_sbert_model, local_device) ...
                if not isinstance(true_label_input, list): raise ValueError("True label for Task 3.4 must be a list of threads.")
                if not true_label_input: raise ValueError("True label list for Task 3.4 is empty.") # Or handle as valid empty case if intended
                embeddings_list_cpu = []
                for t_idx, true_thread in enumerate(true_label_input):
                    if not isinstance(true_thread, list):
                        thread_logger.warning(f"    Skipping true_label thread index {t_idx} for prompt {original_prompt_idx + 1}: Expected list, got {type(true_thread)}.")
                        continue
                    embedding_cpu = get_thread_embedding(true_thread, local_sbert_model, 'cpu') # Ensure SBERT model is passed
                    if embedding_cpu is not None:
                        valid_true_labels_data.append({"original_index": t_idx, "thread": true_thread, "embedding": embedding_cpu})
                        embeddings_list_cpu.append(embedding_cpu)
                    else:
                        thread_logger.warning(f"    Could not encode true_label thread index {t_idx} for prompt {original_prompt_idx + 1}.")
                if not valid_true_labels_data: raise ValueError("No valid true label threads could be encoded.")
                true_label_embeddings = torch.stack(embeddings_list_cpu).to(local_device) # Move to thread's device

            if not valid_true_labels_data or (true_label_embeddings is None and not (FLAG_34 and not true_label_input)):
                if not (FLAG_34 and not true_label_input):
                    raise ValueError("Failed to generate valid true label embeddings or no valid true labels.")
        except ValueError as ve:
            thread_logger.error(f"  Error processing true label for prompt {original_prompt_idx + 1}: {ve} - Skipping prompt evaluation.")
            processing_error_occurred = True
            prompt_results_for_combined_json["error_processing_true_label"] = str(ve)
        except Exception as e:
            thread_logger.error(f"  Unexpected error processing true label for prompt {original_prompt_idx + 1}: {e} - Skipping prompt evaluation.")
            processing_error_occurred = True
            prompt_results_for_combined_json["error_processing_true_label"] = f"Unexpected error: {e}"

        if processing_error_occurred:
            error_msg = prompt_results_for_combined_json["error_processing_true_label"]
            thread_domain_stats[current_subreddit]['prompts_with_true_label_errors'] = thread_domain_stats[current_subreddit].get('prompts_with_true_label_errors', 0) + 1
            num_active_models = len(active_models_list)
            thread_domain_stats[current_subreddit]['total_model_attempts_on_prompts'] = thread_domain_stats[current_subreddit].get('total_model_attempts_on_prompts', 0) + num_active_models
            thread_domain_stats[current_subreddit]['post_api_eval_errors_count'] = thread_domain_stats[current_subreddit].get('post_api_eval_errors_count', 0) + num_active_models

            for model_name in active_models_list:
                stats = thread_model_stats[model_name]
                stats['total'] += 1
                stats['errors'] += 1
                # ... (append to thread_model_results as in original) ...
                thread_model_results[model_name].append(
                    {"prompt_index": original_prompt_idx, "user": user, "prompt_preview": current_prompt_text[:100] + "...",
                     "true_label_preview": "ERROR", "prediction": "N/A", "best_sbert_score": "N/A",
                     "best_rougeL_score": "N/A", "best_rouge1_score": "N/A", "best_bleu_score": "N/A",
                     "best_meteor_score": "N/A", "api_error": f"Skipped due to True Label Error: {error_msg}"})
                prompt_results_for_combined_json["model_outputs"][model_name] = {"best_candidate_text": None, # ... as original
                                                                                 "best_sbert_score": None,
                                                                                 "most_similar_true_label": None,
                                                                                 "similarity_to_best_true_label": None,
                                                                                 "api_error": f"Skipped due to True Label Error: {error_msg}"}


            if thread_combined_output_file:
                try:
                    thread_combined_output_file.write(json.dumps(prompt_results_for_combined_json, ensure_ascii=False) + '\n')
                except Exception as write_err:
                    thread_logger.error(f"Error writing combined results (true label error) for prompt {original_prompt_idx + 1}: {write_err}")
            continue # To next prompt in chunk

        current_prompt_scores_sbert = {} # For confusing prompt detection within this thread's chunk
        for model_name in active_models_list:
            config = MODEL_CONFIGS[model_name]
            stats = thread_model_stats[model_name] # Use thread-local stats
            stats['total'] += 1

            thread_domain_stats[current_subreddit]['total_model_attempts_on_prompts'] = thread_domain_stats[current_subreddit].get('total_model_attempts_on_prompts', 0) + 1
            thread_logger.info(f"  Model: {model_name[:30]:<30} | Requesting {NUM_CANDIDATES} candidates...")
            
            api_client = thread_initialized_clients.get(model_name) # Use thread-local client
            api_error = None
            candidate_responses = []
            # ... (Initialize best_sbert_candidate_idx, max_similarity_score etc. as in original) ...
            best_sbert_candidate_idx = -1; max_similarity_score = -1.0; best_sbert_candidate_preview = "Error"
            max_rougeL_score = -1.0; max_rouge1_score = -1.0; max_bleu_score = -1.0; max_meteor_score = -1.0
            best_sbert_candidate_raw_string = ""; most_similar_true_label_content = None; similarity_to_best_true_label = None


            if api_client is None: # Check if client initialization failed earlier for this model
                api_error = f"API Client not available for {model_name} in thread {thread_id}"
                thread_logger.error(api_error)
                stats['errors'] += 1
                thread_domain_stats[current_subreddit]['api_errors_count'] = thread_domain_stats[current_subreddit].get('api_errors_count', 0) + 1
                current_prompt_scores_sbert[model_name] = -1.0
                # ... (append error to thread_model_results and prompt_results_for_combined_json) ...
                thread_model_results[model_name].append(
                    {"prompt_index": original_prompt_idx, # ...
                     "prediction": f"Error: {api_error}", # ...
                     "api_error": api_error})
                prompt_results_for_combined_json["model_outputs"][model_name] = {"api_error": api_error}
                continue # To next model

            task_num_candidates = NUM_CANDIDATES
            task_max_tokens = GENERATION_MAX_TOKENS
            task_temperature = GENERATION_TEMPERATURE
            start_time = time.time()

            if config["api_type"] == "azure":
                if not isinstance(api_client, AzureOpenAI): # Should not happen if init was successful
                    api_error = f"Azure client became invalid for {model_name} (unexpected)"
                    thread_logger.error(api_error)
                else:
                    candidate_responses, api_error = call_azure_openai_with_retry(
                        api_client, config["deployment_name"], current_prompt_text, config,
                        task_num_candidates, task_max_tokens, task_temperature, thread_logger # Pass thread_logger
                    )
            elif config["api_type"] == "google_vertex":
                candidate_responses, api_error = call_google_vertex_with_retry(
                    api_client, config.get("model_id"), current_prompt_text,
                    task_num_candidates, task_max_tokens, task_temperature, thread_logger # Pass thread_logger
                )
            else:
                api_error = f"API call logic not implemented for api_type '{config['api_type']}'"
                thread_logger.error(f"API Call Error for {model_name}: {api_error}")
            
            end_time = time.time()
            duration = end_time - start_time

            # --- Evaluation logic for candidates (largely same as original, ensure using thread-local vars) ---
            # Reset scores for this model's attempt on this prompt
            max_similarity_score = -1.0; max_rougeL_score = -1.0; max_rouge1_score = -1.0
            max_bleu_score = -1.0; max_meteor_score = -1.0
            best_sbert_candidate_idx = -1; best_sbert_candidate_raw_string = ""
            best_sbert_candidate_preview = "Error"; most_similar_true_label_content = None
            similarity_to_best_true_label = None

            if api_error:
                thread_logger.warning(f"Model: {model_name[:30]:<30} | API Error! ({str(api_error)[:60]}...)")
                stats['errors'] += 1
                thread_domain_stats[current_subreddit]['api_errors_count'] = thread_domain_stats[current_subreddit].get('api_errors_count', 0) + 1
                best_sbert_candidate_preview = f"Error: {api_error}"
                current_prompt_scores_sbert[model_name] = -1.0
            elif not candidate_responses:
                no_cand_msg = "No candidates received (API call successful)"
                thread_logger.warning(f"Model: {model_name[:30]:<30} | Pred Error! {no_cand_msg}.")
                stats['errors'] += 1
                thread_domain_stats[current_subreddit]['post_api_eval_errors_count'] = thread_domain_stats[current_subreddit].get('post_api_eval_errors_count', 0) + 1
                best_sbert_candidate_preview = f"Error: {no_cand_msg}"
                current_prompt_scores_sbert[model_name] = -1.0
                api_error = no_cand_msg # Set api_error for logging
            else:
                valid_candidates_data = []
                try:
                    num_received = len(candidate_responses)
                    thread_logger.debug(f"    Processing {num_received} candidates received from {model_name}...")
                    with torch.no_grad(): # SBERT encoding should be in no_grad
                        for idx, cand_str in enumerate(candidate_responses):
                            # ... (Original candidate validation and embedding) ...
                            # Ensure you use local_sbert_model and local_device
                            if not isinstance(cand_str, str) or not cand_str.strip():
                                thread_logger.warning(f"    Cand #{idx + 1}/{num_received} for {model_name} is not a valid string or is empty. Skipping.")
                                continue
                            embedding = None; struct = None
                            try:
                                if FLAG_34:
                                    try:
                                        struct = json.loads(cand_str)
                                        if not isinstance(struct, list): # ...
                                            thread_logger.warning(f"    Cand #{idx+1}/{num_received} for {model_name} (Task 3.4) JSON parsed but not list. Skipping.")
                                            continue
                                        embedding = get_thread_embedding(struct, local_sbert_model, local_device) # Pass SBERT
                                        if embedding is None: # ...
                                            thread_logger.warning(f"    Cand #{idx+1}/{num_received} for {model_name} (Task 3.4) failed to embed. Skipping.")
                                            continue
                                    except json.JSONDecodeError as json_e: # ...
                                        thread_logger.warning(f"    Cand #{idx+1}/{num_received} for {model_name} (Task 3.4) JSON parse error: {json_e}. Skipping.")
                                        continue
                                elif FLAG_33:
                                    embedding = local_sbert_model.encode(cand_str, convert_to_tensor=True, device=local_device) # Pass SBERT
                                    if not torch.is_tensor(embedding): # ...
                                        thread_logger.warning(f"    Cand #{idx+1}/{num_received} for {model_name} (Task 3.3) failed embedding. Skipping.")
                                        continue
                                
                                if embedding is not None:
                                    valid_candidates_data.append({"idx": idx, "raw_str": cand_str, "struct": struct,
                                                                  "embedding": embedding.to(local_device) if torch.is_tensor(embedding) else embedding})
                                else:
                                    thread_logger.warning(f"    Cand #{idx + 1}/{num_received} for {model_name} did not produce valid embedding. Skipping.")
                                    continue
                            except Exception as e_cand_proc:
                                thread_logger.error(f"    Unexpected error processing cand #{idx + 1}/{num_received} for {model_name}: {e_cand_proc}. Skipping.")
                                continue
                    
                    if valid_candidates_data:
                        # ... (Original SBERT similarity, ROUGE, BLEU, METEOR calculation) ...
                        # Make sure to use local_sbert_model, local_device, local_rouge_scorer, local_meteor_available
                        # And use thread_logger for logging within this section.
                        candidate_embeddings_list = [d['embedding'] for d in valid_candidates_data if torch.is_tensor(d['embedding'])]
                        if not candidate_embeddings_list: raise ValueError("No valid tensor embeddings for SBERT.")

                        candidate_embeddings_tensor = torch.stack(candidate_embeddings_list).to(local_device)
                        # Ensure true_label_embeddings is on the same device
                        sbert_sim_matrix = util.cos_sim(candidate_embeddings_tensor, true_label_embeddings.to(local_device))
                        max_sbert_scores_per_candidate, _ = torch.max(sbert_sim_matrix, dim=1)
                        # ... (rest of similarity logic from original, using local_device) ...
                        sbert_scores_for_valid_candidates = []
                        for cand_data_item in valid_candidates_data:
                            if torch.is_tensor(cand_data_item['embedding']):
                                sim_to_true_labels = util.cos_sim(cand_data_item['embedding'].to(local_device),
                                                                  true_label_embeddings.to(local_device))
                                sbert_scores_for_valid_candidates.append(torch.max(sim_to_true_labels).item())
                            else: sbert_scores_for_valid_candidates.append(-1.0)
                        if not sbert_scores_for_valid_candidates: raise ValueError("No SBERT scores calc.")

                        max_similarity_score = max(sbert_scores_for_valid_candidates)
                        best_sbert_candidate_local_idx = sbert_scores_for_valid_candidates.index(max_similarity_score)
                        best_sbert_data = valid_candidates_data[best_sbert_candidate_local_idx]
                        best_sbert_candidate_idx = best_sbert_data['idx']
                        best_sbert_candidate_raw_string = best_sbert_data['raw_str']
                        
                        # ... (most_similar_true_label_content, similarity_to_best_true_label logic) ...
                        best_cand_embedding_on_device = best_sbert_data['embedding'].to(local_device) if torch.is_tensor(best_sbert_data['embedding']) else local_sbert_model.encode(best_sbert_data['raw_str'], convert_to_tensor=True, device=local_device)
                        sim_scores_vs_all_true = util.cos_sim(best_cand_embedding_on_device, true_label_embeddings.to(local_device))
                        best_true_label_local_idx = torch.argmax(sim_scores_vs_all_true[0]).item()
                        most_similar_true_label_data = valid_true_labels_data[best_true_label_local_idx]
                        most_similar_true_label_content = most_similar_true_label_data['thread']
                        similarity_to_best_true_label = max_similarity_score


                        stats['successful_preds'] += 1
                        stats['total_similarity'] += max_similarity_score
                        current_prompt_scores_sbert[model_name] = max_similarity_score
                        if max_similarity_score != -1.0: # Domain stats update
                            thread_domain_stats[current_subreddit]['successful_sbert_evals'] = thread_domain_stats[current_subreddit].get('successful_sbert_evals', 0) + 1
                            thread_domain_stats[current_subreddit]['sum_sbert_scores'] = thread_domain_stats[current_subreddit].get('sum_sbert_scores', 0.0) + max_similarity_score

                        if FLAG_33 and MULTI_EVAL and local_rouge_scorer: # Check local_rouge_scorer
                            # ... (ROUGE/BLEU/METEOR logic, using local_rouge_scorer, local_meteor_available, true_label_text_single, true_label_tokenized_list, etc.)
                            # ... and update thread_domain_stats for these metrics as well
                            temp_max_rougeL, temp_max_rouge1, temp_max_bleu, temp_max_meteor = 0.0, 0.0, 0.0, 0.0
                            for cand_data_item in valid_candidates_data:
                                cand_text_for_eval = cand_data_item['raw_str']
                                if not cand_text_for_eval.strip(): continue
                                if local_rouge_scorer: # Ensure scorer is available
                                    try:
                                        rouge_scores = local_rouge_scorer.score(true_label_text_single, cand_text_for_eval)
                                        current_rougeL = rouge_scores['rougeL'].fmeasure; current_rouge1 = rouge_scores['rouge1'].fmeasure
                                        if current_rougeL > temp_max_rougeL: temp_max_rougeL = current_rougeL
                                        if current_rouge1 > temp_max_rouge1: temp_max_rouge1 = current_rouge1
                                    except Exception: pass
                                # ... (BLEU/METEOR calc as in original, using local_meteor_available)
                                candidate_tokens_for_eval = nltk.word_tokenize(cand_text_for_eval.lower())
                                if not candidate_tokens_for_eval: continue
                                try: # BLEU
                                    current_bleu = sentence_bleu(true_label_tokenized_list, candidate_tokens_for_eval, smoothing_function=SmoothingFunction().method1)
                                    if current_bleu > temp_max_bleu: temp_max_bleu = current_bleu
                                except Exception: pass
                                if local_meteor_available and true_label_tokenized_for_meteor: # METEOR
                                    try:
                                        current_meteor = meteor_score(true_label_tokenized_for_meteor, candidate_tokens_for_eval)
                                        if current_meteor > temp_max_meteor: temp_max_meteor = current_meteor
                                    except Exception: pass

                            max_rougeL_score = temp_max_rougeL if temp_max_rougeL > 0 else -1.0
                            max_rouge1_score = temp_max_rouge1 if temp_max_rouge1 > 0 else -1.0
                            max_bleu_score = temp_max_bleu if temp_max_bleu > 0 else -1.0
                            max_meteor_score = temp_max_meteor if temp_max_meteor > 0 else -1.0
                            
                            # Update thread_model_stats and thread_domain_stats for ROUGE/BLEU/METEOR
                            if max_rougeL_score >= 0.0:
                                stats['successful_rougeL'] += 1; stats['total_rougeL'] += max_rougeL_score
                                thread_domain_stats[current_subreddit]['successful_rougeL_evals'] = thread_domain_stats[current_subreddit].get('successful_rougeL_evals',0)+1
                                thread_domain_stats[current_subreddit]['sum_rougeL_scores'] = thread_domain_stats[current_subreddit].get('sum_rougeL_scores',0.0)+max_rougeL_score
                            # ... repeat for R1, BLEU, METEOR ...
                            if max_rouge1_score >= 0.0:
                                stats['successful_rouge1'] += 1; stats['total_rouge1'] += max_rouge1_score
                                thread_domain_stats[current_subreddit]['successful_rouge1_evals']+=1; thread_domain_stats[current_subreddit]['sum_rouge1_scores']+=max_rouge1_score
                            if max_bleu_score >= 0.0:
                                stats['successful_bleu'] += 1; stats['total_bleu'] += max_bleu_score
                                thread_domain_stats[current_subreddit]['successful_bleu_evals']+=1; thread_domain_stats[current_subreddit]['sum_bleu_scores']+=max_bleu_score
                            if local_meteor_available and max_meteor_score >= 0.0:
                                stats['successful_meteor'] += 1; stats['total_meteor'] += max_meteor_score
                                thread_domain_stats[current_subreddit]['successful_meteor_evals']+=1; thread_domain_stats[current_subreddit]['sum_meteor_scores']+=max_meteor_score


                        # ... (Preview generation and logging as in original, using thread_logger) ...
                        if FLAG_33: best_sbert_candidate_preview = best_sbert_candidate_raw_string[:100] + "..."
                        else: # ... (original preview logic for FLAG_34)
                            first_comment_body = "(Invalid Struct)" # ...
                            try:
                                if isinstance(best_sbert_data['struct'], list) and best_sbert_data['struct']:
                                    first_node = best_sbert_data['struct'][0]
                                    if isinstance(first_node, dict): first_comment_body = first_node.get('body','(Body Missing)')[:80]
                            except Exception as preview_e: thread_logger.warning(f"    Error gen preview: {preview_e}"); first_comment_body = "(Preview Error)"
                            best_sbert_candidate_preview = f"BestCandFirstComment: {first_comment_body}..."


                        log_info_thread = f"Best SBERT Sim: {max_similarity_score:.4f} (Cand Original Idx #{best_sbert_candidate_idx + 1} vs TrueLabel Idx {most_similar_true_label_data['original_index']})"
                        if FLAG_33 and MULTI_EVAL and local_rouge_scorer:
                            log_info_thread += f" | Best R-L: {max_rougeL_score:.4f} | Best R-1: {max_rouge1_score:.4f} | Best BLEU: {max_bleu_score:.4f}"
                            if local_meteor_available: log_info_thread += f" | Best METEOR: {max_meteor_score:.4f}"
                        log_info_thread += f" | Time(API+Eval): {duration:.2f}s"
                        thread_logger.info(f"Model: {model_name[:30]:<30} | {log_info_thread}")

                        if WRITE_OUTPUT and thread_best_candidate_files.get(model_name) and best_sbert_candidate_idx != -1:
                            try:
                                # ... (write to thread_best_candidate_files[model_name]) ...
                                output_data_single_model = {"prompt_index": original_prompt_idx, "user": user,
                                                            "target_id": target_post_id if FLAG_34 else None,
                                                            "true_label": true_label_input, "subreddit": current_subreddit,
                                                            "best_sbert_candidate_text": best_sbert_candidate_raw_string,
                                                            "best_sbert_score": max_similarity_score}
                                thread_best_candidate_files[model_name].write(json.dumps(output_data_single_model, ensure_ascii=False) + '\n')
                            except Exception as write_err_cand:
                                thread_logger.error(f"Error writing best cand file for {model_name}, prompt {original_prompt_idx + 1}: {write_err_cand}")

                    else: # No valid_candidates_data
                        # ... (Error handling as in original, update thread_domain_stats post_api_eval_errors_count) ...
                        no_valid_cand_msg = f"No valid candidates processed out of {num_received if 'num_received' in locals() else 'unknown'} received."
                        thread_logger.warning(f"Model: {model_name[:30]:<30} | Pred Error! {no_valid_cand_msg}")
                        stats['errors'] += 1;
                        thread_domain_stats[current_subreddit]['post_api_eval_errors_count']+=1
                        best_sbert_candidate_preview = f"Error: {no_valid_cand_msg}"; max_similarity_score = -1.0 # ... etc.
                        api_error = no_valid_cand_msg; current_prompt_scores_sbert[model_name] = -1.0
                
                except ValueError as ve_eval: # Errors during similarity calculation etc.
                    # ... (Error handling, update thread_domain_stats post_api_eval_errors_count) ...
                    eval_err_msg = f"Eval Processing Error (e.g. Sim Calc): {ve_eval}"; thread_logger.error(f"{eval_err_msg} for {model_name}!");
                    stats['errors'] += 1; thread_domain_stats[current_subreddit]['post_api_eval_errors_count']+=1
                    best_sbert_candidate_preview = eval_err_msg; max_similarity_score = -1.0 # ... etc.
                    api_error = eval_err_msg; current_prompt_scores_sbert[model_name] = -1.0
                except RuntimeError as rte_eval: # CUDA/MPS OOM during eval
                    # ... (Error handling, update thread_domain_stats post_api_eval_errors_count) ...
                    oom_err_msg = f"CUDA/MPS Runtime Error (Sim Calc / potential OOM): {rte_eval}"; thread_logger.error(f"Eval Runtime Error for {model_name}! {oom_err_msg}");
                    stats['errors'] += 1; thread_domain_stats[current_subreddit]['post_api_eval_errors_count']+=1
                    best_sbert_candidate_preview = oom_err_msg; max_similarity_score = -1.0 # ... etc.
                    api_error = oom_err_msg; current_prompt_scores_sbert[model_name] = -1.0
                except Exception as e_eval_unexp: # Other unexpected errors during eval
                    # ... (Error handling, update thread_domain_stats post_api_eval_errors_count) ...
                    unexp_err_msg = f"Unexpected Eval Error (Sim Calc): {type(e_eval_unexp).__name__}: {str(e_eval_unexp)[:100]}..."; thread_logger.error(f"Unexpected Eval Error for {model_name}! {unexp_err_msg}");
                    stats['errors'] += 1; thread_domain_stats[current_subreddit]['post_api_eval_errors_count']+=1
                    best_sbert_candidate_preview = unexp_err_msg; max_similarity_score = -1.0 # ... etc.
                    api_error = unexp_err_msg; current_prompt_scores_sbert[model_name] = -1.0


            # Log running stats for model (using thread_logger)
            # ... (Copied from original, ensuring stats and meteor_available_global are correctly referenced) ...
            successful_sbert_calcs_thread = stats['successful_preds']
            running_avg_sim_thread = (stats['total_similarity'] / successful_sbert_calcs_thread) if successful_sbert_calcs_thread > 0 else 0.0
            log_msg_thread = f"  Model: {model_name[:30]:<30} | Running Avg SBERT: {running_avg_sim_thread:.4f} ({successful_sbert_calcs_thread} valid out of {stats['total']})"
            if FLAG_33 and MULTI_EVAL and local_rouge_scorer: # check local scorer
                running_avg_rougeL_thread = (stats['total_rougeL'] / stats['successful_rougeL']) if stats['successful_rougeL'] > 0 else 0.0
                # ... (R1, BLEU, METEOR running avgs) ...
                log_msg_thread += f" | Avg R-L: {running_avg_rougeL_thread:.4f} ({stats['successful_rougeL']})" # ... etc.
            log_msg_thread += f" | Errors: {stats['errors']}"
            thread_logger.info(log_msg_thread)


            # Append to thread_model_results
            true_label_display_thread = true_label_input[:80] + "..." if isinstance(true_label_input, str) else (f"{len(true_label_input)} threads" if isinstance(true_label_input, list) else "N/A")
            thread_model_results[model_name].append({
                "prompt_index": original_prompt_idx, "user": user, "prompt_preview": current_prompt_text[:100] + "...",
                "true_label_preview": true_label_display_thread, "prediction": best_sbert_candidate_preview,
                "best_sbert_score": f"{max_similarity_score:.6f}" if max_similarity_score != -1.0 else "N/A",
                "best_rougeL_score": f"{max_rougeL_score:.6f}" if FLAG_33 and MULTI_EVAL and local_rouge_scorer and max_rougeL_score >= 0.0 else "N/A",
                # ... (R1, BLEU, METEOR scores) ...
                "api_error": str(api_error) if api_error else "",
            })
            # Update prompt_results_for_combined_json
            prompt_results_for_combined_json["model_outputs"][model_name] = {
                "best_candidate_text": best_sbert_candidate_raw_string if best_sbert_candidate_idx != -1 else None,
                "best_sbert_score": max_similarity_score if max_similarity_score != -1.0 else None,
                "most_similar_true_label": most_similar_true_label_content,
                "similarity_to_best_true_label": similarity_to_best_true_label,
                "api_error": str(api_error) if api_error else None
            }
            
            if local_device != 'cpu': # Memory cleanup for GPU
                try:
                    if 'candidate_embeddings_tensor' in locals(): del candidate_embeddings_tensor
                    if 'sbert_sim_matrix' in locals(): del sbert_sim_matrix
                    if 'max_sbert_scores_per_candidate' in locals(): del max_sbert_scores_per_candidate
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): torch.mps.empty_cache() # For MPS
                except NameError: pass
                except Exception as e_mem_clean: thread_logger.warning(f"Error during GPU memory cleanup for model: {e_mem_clean}")
            
            if API_DELAY_SECONDS > 0: time.sleep(API_DELAY_SECONDS) # Per model delay (if any)

        if thread_combined_output_file:
            try:
                thread_combined_output_file.write(json.dumps(prompt_results_for_combined_json, ensure_ascii=False) + '\n')
            except TypeError as te_json: # ... (original error handling for JSON serialization) ...
                thread_logger.error(f"Error serializing combined results for prompt {original_prompt_idx + 1} to JSON: {te_json}.")
            except Exception as write_err_comb: # ...
                thread_logger.error(f"Error writing combined results for prompt {original_prompt_idx + 1}: {write_err_comb}")


        if len(current_prompt_scores_sbert) == len(active_models_list):
            valid_scores_thread = [s for s in current_prompt_scores_sbert.values() if s != -1.0]
            if len(valid_scores_thread) == len(active_models_list):
                all_low_thread = all(s < LOW_SIM_THRESHOLD for s in valid_scores_thread)
                all_high_thread = all(s > HIGH_SIM_THRESHOLD for s in valid_scores_thread)
                if all_low_thread or all_high_thread:
                    thread_confusing_prompt_indices.add(original_prompt_idx) # Add original index
                    thread_logger.warning(f"--- Prompt Original Idx {original_prompt_idx + 1} flagged as CONFUSING ---")
        
        if PROMPT_DELAY_SECONDS > 0 and i_chunk < final_prompt_index_in_chunk:
            thread_logger.info(f"--- Delaying {PROMPT_DELAY_SECONDS}s ---")
            time.sleep(PROMPT_DELAY_SECONDS)
        
        if local_device != 'cpu': # Memory cleanup for GPU after prompt
            try:
                if 'true_label_embeddings' in locals() and true_label_embeddings is not None:
                    del true_label_embeddings
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): torch.mps.empty_cache()
            except NameError: pass
            except Exception as e_mem_prompt: thread_logger.warning(f"Error during GPU memory cleanup after prompt: {e_mem_prompt}")

    # Close thread-specific files
    if thread_combined_output_file:
        try: thread_combined_output_file.close()
        except Exception as e_close: thread_logger.warning(f"Error closing thread combined output file: {e_close}")
    
    if WRITE_OUTPUT:
        for model_name_key, file_handle in thread_best_candidate_files.items():
            if file_handle:
                try: file_handle.close()
                except Exception as e_close_best: thread_logger.warning(f"Error closing best cand file for {model_name_key}: {e_close_best}")

    thread_logger.info(f"Finished processing {len(prompts_chunk)} prompts. Log: {thread_log_filename}")
    return thread_model_results, thread_model_stats, thread_domain_stats, thread_confusing_prompt_indices, thread_combined_output_filename


def main_setup():
    global sbert_model, device, rouge_scorer_instance, meteor_available_global, MULTI_EVAL

    main_logger.info(f"Running Task: {TASK_NAME}")
    main_logger.info(f"Input file: {INPUT_JSONL_PATH}")
    main_logger.info(f"Base Output directory: {BASE_OUTPUT_DIR}") # Changed to BASE_OUTPUT_DIR
    main_logger.info(f"Generating {NUM_CANDIDATES} candidates per prompt.")
    
    # --- NLTK Resource Check (moved from __main__ to be called once) ---
    nltk_punkt_available = False
    nltk_wordnet_omw_available = False

    if FLAG_33 and MULTI_EVAL_ENABLED: # Only check if potentially needed by any thread
        try:
            nltk.data.find('tokenizers/punkt'); nltk_punkt_available = True
            main_logger.info("NLTK 'punkt' tokenizer found.")
        except LookupError:
            main_logger.info("NLTK 'punkt' tokenizer not found. Attempting download...")
            try: nltk.download('punkt', quiet=True); main_logger.info("'punkt' downloaded successfully."); nltk_punkt_available = True
            except Exception as nltk_e: main_logger.error(f"Failed NLTK 'punkt' download: {nltk_e}.")
        
        try:
            nltk.data.find('corpora/wordnet'); nltk.data.find('corpora/omw-1.4'); nltk_wordnet_omw_available = True
            main_logger.info("NLTK 'wordnet' and 'omw-1.4' resources found (for METEOR).")
        except LookupError:
            main_logger.info("NLTK 'wordnet' or 'omw-1.4' not found. Attempting download...")
            try:
                nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)
                main_logger.info("'wordnet' and 'omw-1.4' downloaded successfully.")
                nltk_wordnet_omw_available = True
            except Exception as nltk_e: main_logger.warning(f"Failed NLTK 'wordnet'/'omw-1.4' download: {nltk_e}.")

        if not nltk_punkt_available:
            MULTI_EVAL = False # Global flag for evaluation logic in threads
            meteor_available_global = False
            main_logger.warning("MULTI_EVAL (ROUGE/BLEU/METEOR) is disabled as 'punkt' tokenizer is unavailable for all threads.")
        elif not nltk_wordnet_omw_available:
            meteor_available_global = False # Disable only METEOR if punkt is OK but wordnet/omw failed
            main_logger.warning("METEOR scoring is disabled/may be limited as 'wordnet'/'omw-1.4' resources are unavailable for all threads.")
        else: # Both punkt and wordnet/omw are available
            meteor_available_global = True
    else: # MULTI_EVAL_ENABLED is False or not FLAG_33
        MULTI_EVAL = False
        meteor_available_global = False
        if FLAG_33: main_logger.info("MULTI_EVAL was initially disabled or pre-requisites not met for all threads.")
    
    main_logger.info(f"Multi-Eval (ROUGE/BLEU/METEOR) effectively enabled for threads: {MULTI_EVAL} (Only applies if Task 3.3)")
    main_logger.info(f"METEOR available for threads: {meteor_available_global}")
    main_logger.info(f"Write Output (Best Candidate Per Model Per Thread) enabled: {WRITE_OUTPUT}")


    # --- Load SBERT Model (once, then pass to threads) ---
    main_logger.info(f"Loading SBERT model: {SBERT_MODEL_NAME}...")
    try:
        if torch.cuda.is_available():
            device_name = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_name = 'mps'
        else:
            device_name = 'cpu'
        device = torch.device(device_name) # Store global device object
        main_logger.info(f"Using device: {device_name}")
        sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=device_name) # Load to global device
        main_logger.info("SBERT model loaded successfully.")
    except Exception as e:
        main_logger.error(f"Error loading SBERT model: {e}"); sys.exit(1)

    # --- Initialize ROUGE Scorer (once, if needed, then pass to threads) ---
    if MULTI_EVAL and FLAG_33:
        try:
            rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL', 'rouge1'], use_stemmer=True)
            main_logger.info("ROUGE (L,1) scorer initialized for threads.")
        except Exception as e:
            main_logger.error(f"Failed to initialize ROUGE scorer: {e}. ROUGE evaluation disabled for threads.")
            MULTI_EVAL = False # If ROUGE fails, disable dependent parts of MULTI_EVAL
            # meteor_available_global might still be true if only ROUGE failed but NLTK was fine
    
    if not os.path.exists(INPUT_JSONL_PATH):
        main_logger.error(f"Input JSONL file not found at {INPUT_JSONL_PATH}"); sys.exit(1)
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True) # Ensure base output dir exists

    prompts_data = []
    line_num = 0
    main_logger.info(f"Reading prompts from {INPUT_JSONL_PATH}...")
    try:
        with open(INPUT_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1; line = line.strip()
                if not line: continue
                try:
                    data_dict = json.loads(line)
                    # ... (original prompt validation logic) ...
                    if not data_dict.get("prompt") or not data_dict.get("true_label"): main_logger.warning(f"Skip L{line_num}. Missing 'prompt' or 'true_label'."); continue
                    if FLAG_33 and not isinstance(data_dict["true_label"], str): main_logger.warning(f"Skip L{line_num}. Task 3.3 needs string true_label."); continue
                    # ... (other validations)
                    prompts_data.append({'original_idx': len(prompts_data), 'data': data_dict}) # Store with original index
                except json.JSONDecodeError as e_json: main_logger.warning(f"Skip L{line_num}. JSON decode error: {e_json}.")
                except Exception as e_parse: main_logger.warning(f"Skip L{line_num}. Error parsing: {e_parse}")
    except FileNotFoundError: main_logger.error(f"Critical Error: Input file not found at {INPUT_JSONL_PATH}"); sys.exit(1)
    except Exception as e_read: main_logger.error(f"Critical Error reading input JSONL: {e_read}"); sys.exit(1)
    
    if not prompts_data: main_logger.error("No valid prompts found. Exiting."); sys.exit(0)
    main_logger.info(f"Read {len(prompts_data)} valid prompts.")

    limit = 5 if test_flag else (500 if CUT else None)
    prompts_to_process_all = prompts_data[:limit] if limit else prompts_data
    run_mode = f"TEST MODE - First {len(prompts_to_process_all)}" if test_flag else (
        f"CUT MODE - First {len(prompts_to_process_all)}" if CUT else "FULL RUN")
    main_logger.info(f"\n{'=' * 10} Starting Interleaved Evaluation ({run_mode} - {TASK_NAME}) {'=' * 10}")

    return prompts_to_process_all, run_mode


# --- Main Evaluation Logic ---
def main():
    prompts_to_process_all, run_mode = main_setup() # Call setup to load models and prompts

    num_threads = 12
    if not prompts_to_process_all:
        main_logger.info("No prompts to process. Exiting.")
        return

    chunk_size = math.ceil(len(prompts_to_process_all) / num_threads)
    prompt_chunks = [prompts_to_process_all[i:i + chunk_size] for i in range(0, len(prompts_to_process_all), chunk_size)]
    
    threads = []
    thread_results_list = [] # To store (model_results, model_stats, domain_stats, confusing_indices, combined_file_path) from each thread

    for i in range(num_threads):
        if i < len(prompt_chunks): # Only start threads if there's a chunk for them
            thread_output_dir = os.path.join(BASE_OUTPUT_DIR, f"thread_{i}")
            os.makedirs(thread_output_dir, exist_ok=True)
            
            # Pass global SBERT model, device, ROUGE scorer, and meteor availability to each thread
            # These resources are read-only within the thread's processing loop or used by thread-safe libraries.
            thread = threading.Thread(
                target=lambda: thread_results_list.append(
                    process_prompts_thread(
                        i, prompt_chunks[i], thread_output_dir, 
                        sbert_model, device, rouge_scorer_instance, meteor_available_global
                    )
                ),
                name=f"Thread-{i}"
            )
            threads.append(thread)
            thread.start()
            main_logger.info(f"Thread-{i} started for {len(prompt_chunks[i])} prompts. Output dir: {thread_output_dir}")
        else:
            main_logger.info(f"Skipping Thread-{i} as there are no more prompt chunks.")


    for thread in threads:
        thread.join()
    main_logger.info("All threads completed.")

    # --- Combine Results from all Threads ---
    main_logger.info("\nCombining results from all threads...")
    all_model_results = {name: [] for name in MODEL_CONFIGS.keys()}
    all_model_stats = {name: {'successful_preds': 0, 'total_similarity': 0.0,
                              'successful_rougeL': 0, 'total_rougeL': 0.0,
                              'successful_rouge1': 0, 'total_rouge1': 0.0,
                              'successful_bleu': 0, 'total_bleu': 0.0,
                              'successful_meteor': 0, 'total_meteor': 0.0,
                              'errors': 0, 'total': 0} for name in MODEL_CONFIGS.keys()}
    all_domain_stats = {}
    all_confusing_prompt_indices = set()
    all_combined_thread_files = []


    for t_results in thread_results_list:
        if t_results is None: # Should not happen if thread appends correctly
            main_logger.warning("A thread returned None results. Skipping.")
            continue
        
        t_model_res, t_model_stat, t_domain_stat, t_confusing, t_combined_file = t_results
        
        all_combined_thread_files.append(t_combined_file)

        for model_name_key in MODEL_CONFIGS.keys():
            all_model_results[model_name_key].extend(t_model_res.get(model_name_key, []))
            
            # Aggregate model_stats (summing up counts and totals)
            for stat_key, stat_val in t_model_stat.get(model_name_key, {}).items():
                all_model_stats[model_name_key][stat_key] = all_model_stats[model_name_key].get(stat_key, 0) + stat_val
        
        for domain_key, d_stats in t_domain_stat.items():
            if domain_key not in all_domain_stats:
                all_domain_stats[domain_key] = d_stats.copy() # Initialize if new
            else: # Aggregate domain_stats
                for stat_key, stat_val in d_stats.items():
                    if 'sum_' in stat_key or stat_key.endswith('_evals') or stat_key.endswith('_count') or stat_key.endswith('_errors') or stat_key.startswith('total_'):
                         all_domain_stats[domain_key][stat_key] = all_domain_stats[domain_key].get(stat_key, 0) + stat_val
                    # Be careful with averaging directly here, better to sum and recalculate average later if needed based on sums and counts

        all_confusing_prompt_indices.update(t_confusing)

    # --- Create a single combined input/output JSONL from all thread files ---
    timestamp_str_final = dt.now().strftime("%Y%m%d_%H%M%S")
    final_combined_filename = os.path.join(BASE_OUTPUT_DIR, f"ALL_THREADS_input_best_output_best_match_{TASK_NAME}{run_identifier}_{timestamp_str_final}.jsonl")
    main_logger.info(f"Consolidating all thread combined outputs into: {final_combined_filename}")
    try:
        with open(final_combined_filename, 'w', encoding='utf-8') as outfile:
            for fname in all_combined_thread_files:
                if fname and os.path.exists(fname): # Check if file exists and path is not None
                    try:
                        with open(fname, 'r', encoding='utf-8') as infile:
                            for line in infile:
                                outfile.write(line)
                        main_logger.info(f"  Successfully merged: {fname}")
                    except Exception as e_merge:
                        main_logger.error(f"  Error merging file {fname}: {e_merge}")
                elif fname:
                     main_logger.warning(f"  Thread combined file not found, skipping merge: {fname}")

    except IOError as e_final_comb:
        main_logger.error(f"Error creating final consolidated combined output file {final_combined_filename}: {e_final_comb}")


    # --- Final Domain Metrics Logging (using main_logger) ---
    main_logger.info(f"\n{'=' * 10} Final Combined Domain Metrics ({run_mode}) {'=' * 10}")
    log_domain_metrics(all_domain_stats, main_logger, meteor_available_global, len(MODEL_CONFIGS.keys())) # Pass global meteor_available

    # --- Final Overall Summary (similar to original, using combined stats) ---
    main_logger.info(f"\n{'=' * 10} Evaluation Finished ({run_mode}) - Calculating Final Metrics & Writing Results {'=' * 10}")
    if all_confusing_prompt_indices:
        main_logger.info(f"Identified {len(all_confusing_prompt_indices)} confusing prompts (Original Indices): {sorted(list(all_confusing_prompt_indices))}")
    else:
        main_logger.info("No confusing prompts identified across all threads.")
    
    overall_summary_final = []
    # ... (Fieldnames for summary CSV as in original) ...
    summary_fieldnames = [
        "model_name", "task", "total_prompts_processed", "successful_predictions", "api_pred_eval_errors",
        "avg_best_sbert_score", "overall_success_rate",
        "avg_best_rougeL_score", "avg_best_rouge1_score",
        "avg_best_bleu_score", "avg_best_meteor_score",
        "confusing_samples_excluded", "avg_best_sbert_score_filtered",
        "overall_success_rate_filtered",
        "avg_best_rougeL_score_filtered", "avg_best_rouge1_score_filtered",
        "avg_best_bleu_score_filtered", "avg_best_meteor_score_filtered",
        "output_file" # This will be the per-model summary CSV
    ]


    active_models_list_final = list(MODEL_CONFIGS.keys())
    for model_name_final in active_models_list_final:
        results_final = all_model_results[model_name_final] # Combined results for this model
        # Sort results by original prompt index for consistent output if desired
        results_final.sort(key=lambda x: x.get('prompt_index', float('inf')))

        stats_final = all_model_stats[model_name_final] # Combined stats for this model
        safe_model_name_final = re.sub(r'[\\/*?:"<>|]', '_', model_name_final)
        eval_output_filename_final = os.path.join(BASE_OUTPUT_DIR, f"FINAL_eval_{safe_model_name_final}_{TASK_NAME}{run_identifier}.csv")
        main_logger.info(f"\n--- Processing Final Combined Results for: {model_name_final} ---")

        if results_final:
            main_logger.info(f"  Writing {len(results_final)} combined results summary to {eval_output_filename_final}...")
            try:
                with open(eval_output_filename_final, 'w', newline='', encoding='utf-8-sig') as f_csv:
                    result_fieldnames_csv = ["prompt_index", "user", "prompt_preview", "true_label_preview", "prediction",
                                         "best_sbert_score", "best_rougeL_score", "best_rouge1_score",
                                         "best_bleu_score", "best_meteor_score", "api_error"]
                    writer_csv = csv.DictWriter(f_csv, fieldnames=result_fieldnames_csv, extrasaction='ignore')
                    writer_csv.writeheader()
                    writer_csv.writerows(results_final) # Write combined & sorted results
            except IOError as e_csv_write:
                main_logger.error(f"Error writing final output CSV {eval_output_filename_final}: {e_csv_write}")
        else:
            main_logger.warning(f"  No combined results recorded for {model_name_final}. Skipping CSV write.")
        
        # --- Recalculate averages and rates from combined stats_final ---
        total_processed_final = stats_final['total']
        api_or_pred_errors_final = stats_final['errors']
        successful_sbert_preds_final = stats_final['successful_preds']
        avg_best_sbert_score_final = (stats_final['total_similarity'] / successful_sbert_preds_final) if successful_sbert_preds_final > 0 else 0.0
        success_rate_sbert_final = ((total_processed_final - api_or_pred_errors_final) / total_processed_final) * 100 if total_processed_final > 0 else 0.0
        
        avg_best_rougeL_score_final = (stats_final['total_rougeL'] / stats_final.get('successful_rougeL', 0)) if stats_final.get('successful_rougeL', 0) > 0 else 0.0
        # ... (Calculate R1, BLEU, METEOR averages similarly from stats_final) ...
        avg_best_rouge1_score_final = (stats_final['total_rouge1'] / stats_final.get('successful_rouge1',0)) if stats_final.get('successful_rouge1',0) > 0 else 0.0
        avg_best_bleu_score_final = (stats_final['total_bleu'] / stats_final.get('successful_bleu',0)) if stats_final.get('successful_bleu',0) > 0 else 0.0
        avg_best_meteor_score_final = (stats_final['total_meteor'] / stats_final.get('successful_meteor',0)) if meteor_available_global and stats_final.get('successful_meteor',0) > 0 else 0.0


        # --- Filtered stats (using combined results_final and all_confusing_prompt_indices) ---
        filtered_results_final = [r for r in results_final if r.get('prompt_index', -1) not in all_confusing_prompt_indices]
        # ... (Recalculate filtered averages and rates as in original, but from filtered_results_final) ...
        total_processed_filtered_final = len(filtered_results_final)
        errors_filtered_final = sum(1 for r in filtered_results_final if r['prediction'].startswith("Error:") or r.get('api_error'))
        successful_sbert_preds_filtered_final = sum(1 for r in filtered_results_final if not r['prediction'].startswith("Error:") and not r.get('api_error') and r.get('best_sbert_score', 'N/A') != 'N/A')
        total_similarity_filtered_final = sum(float(r['best_sbert_score']) for r in filtered_results_final if r.get('best_sbert_score', 'N/A') != 'N/A')
        avg_best_sbert_score_filtered_final = (total_similarity_filtered_final / successful_sbert_preds_filtered_final) if successful_sbert_preds_filtered_final > 0 else 0.0
        success_rate_sbert_filtered_final = ((total_processed_filtered_final - errors_filtered_final) / total_processed_filtered_final) * 100 if total_processed_filtered_final > 0 else 0.0
        
        avg_best_rougeL_score_filtered_final, avg_best_rouge1_score_filtered_final, avg_best_bleu_score_filtered_final, avg_best_meteor_score_filtered_final = 0.0, 0.0, 0.0, 0.0
        if FLAG_33 and MULTI_EVAL: # Check global MULTI_EVAL
            # ... (Calculate filtered ROUGE/BLEU/METEOR averages from filtered_results_final) ...
            pass # Placeholder for detailed calculation as in original script

        summary_final_model = {
            "model_name": model_name_final, "task": TASK_NAME, "total_prompts_processed": total_processed_final,
            "successful_predictions": successful_sbert_preds_final, "api_pred_eval_errors": api_or_pred_errors_final,
            "avg_best_sbert_score": f"{avg_best_sbert_score_final:.6f}", 
            "overall_success_rate": f"{success_rate_sbert_final:.2f}%",
            "avg_best_rougeL_score": f"{avg_best_rougeL_score_final:.6f}" if FLAG_33 and MULTI_EVAL else "N/A",
            # ... (R1, BLEU, METEOR scores, filtered scores) ...
            "confusing_samples_excluded": len(all_confusing_prompt_indices), # Show once in summary
            "avg_best_sbert_score_filtered": f"{avg_best_sbert_score_filtered_final:.6f}",
            "overall_success_rate_filtered": f"{success_rate_sbert_filtered_final:.2f}%",
            # ... (filtered R1, BLEU, METEOR)
            "output_file": eval_output_filename_final # Path to this model's combined CSV
        }
        overall_summary_final.append(summary_final_model)

        # Log final summary for this model (using main_logger)
        # ... (Similar to original logging, but use _final suffixed variables) ...
        main_logger.info(f"  Final Combined Summary for {model_name_final} ({TASK_NAME}):")
        main_logger.info(f"    Total Prompts Attempted (all threads): {summary_final_model['total_prompts_processed']}")
        # ... (rest of the logging statements)

    main_logger.info(f"\n{'=' * 20} Final Overall Evaluation Summary (All Models, All Threads){' [' + run_mode + ']'} {'=' * 20}")
    for summary_item_final in overall_summary_final:
        # ... (Log final summary items as in original) ...
        log_str_final = f"Model: {summary_item_final['model_name']:<30} | Task: {summary_item_final['task']:<25} | Success: {summary_item_final['overall_success_rate']:>7s} | Avg SBERT: {summary_item_final['avg_best_sbert_score']}"
        if FLAG_33 and MULTI_EVAL: # Check global MULTI_EVAL
            log_str_final += f" | Avg R-L: {summary_item_final['avg_best_rougeL_score']}" # ... etc.
        main_logger.info(log_str_final)

    # --- Save final combined summary CSV ---
    summary_filename_final = os.path.join(BASE_OUTPUT_DIR, f"FINAL_evaluation_summary_{TASK_NAME}{run_identifier}.csv")
    main_logger.info(f"\nSaving final overall summary to {summary_filename_final}...")
    try:
        with open(summary_filename_final, 'w', newline='', encoding='utf-8-sig') as f_sum_csv:
            if overall_summary_final:
                writer_sum_csv = csv.DictWriter(f_sum_csv, fieldnames=summary_fieldnames)
                writer_sum_csv.writeheader()
                final_summary_rows_csv = []
                for idx, s_item_csv in enumerate(overall_summary_final):
                    s_copy_csv = s_item_csv.copy()
                    if idx > 0: s_copy_csv['confusing_samples_excluded'] = '' # Show only once
                    row_to_write_csv = {field: s_copy_csv.get(field, "N/A") for field in summary_fieldnames}
                    final_summary_rows_csv.append(row_to_write_csv)
                writer_sum_csv.writerows(final_summary_rows_csv)
            else:
                f_sum_csv.write("No final summary data generated.\n")
        main_logger.info(f"Final overall summary saved successfully.")
    except IOError as e_sum_csv_io: main_logger.error(f"Error writing final summary CSV {summary_filename_final}: {e_sum_csv_io}")
    except Exception as e_sum_csv_unexp: main_logger.error(f"Unexpected error saving final summary CSV: {e_sum_csv_unexp}")

    main_logger.info(f"\nMultithreaded evaluation ({run_mode}) complete.")
    main_logger.info(f"Consolidated input/output/best_match JSONL: {final_combined_filename}")
    main_logger.info(f"Main Log file: {main_file_handler_path}")
    main_logger.info(f"Individual thread logs and outputs are in subdirectories within: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    # Initial check for NLTK and MULTI_EVAL setup happens in main_setup() now
    # The global MULTI_EVAL and meteor_available_global will be set there.
    main()