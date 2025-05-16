# ---------------- evaluate_models_unified_fixed.py ----------------
"""
Evaluates LLMs on EITHER Personalized Conversation Generation (Task 3.4)
OR Personalized Follow-up Text Generation (Task 3.3 - predicting body text).

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
    # +++ NEW IMPORT FOR METEOR +++
    from nltk.translate.meteor_score import meteor_score
    import nltk

    # NLTK resources needed: punkt (already there), wordnet (for METEOR)
    # omw-1.4 is often good to have with wordnet for multilingual aspects, though not strictly for English METEOR
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
    try:  # +++ NEW NLTK RESOURCE CHECK FOR WORDNET +++
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("NLTK 'wordnet' resource not found (for METEOR). Downloading...")
        try:
            nltk.download('wordnet', quiet=True)
            print("'wordnet' downloaded successfully.")
        except Exception as nltk_e:
            print(f"ERROR: Failed to download NLTK 'wordnet': {nltk_e}. METEOR will be disabled.")
            # We might still run ROUGE/BLEU, so don't set MULTI_EVAL_ENABLED to False here
            # We'll handle METEOR-specific disabling later if needed.
    try:  # +++ NEW NLTK RESOURCE CHECK FOR OMW-1.4 +++
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

# Import numpy (keep it as torch/sentence-transformers might need it)
try:
    import numpy as np
except ImportError:
    print("ERROR: 'numpy' not found. Please install it: pip install numpy")
    sys.exit(1)
# --- End Imports ---

# --- Logging Setup ---
log_filename = f"FINAL-EVAL-{dt.now():%Y%m%d_%H%M%S}.log"
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Set higher level for noisy libraries if needed
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
if logger.hasHandlers(): logger.handlers.clear()
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)
# --- End Logging Setup ---


# --- >> Consolidated Flags << ---
FLAG_33 = True  # Run Task 3.3 (Predict Body Text) # MODIFIED FOR TESTING
FLAG_34 = not FLAG_33  # Run Task 3.4 (Generate Conversation Thread)
test_flag = False  # If True, run only first 5 prompts
CUT = True  # If True (and test_flag=False), run only first 500 prompts
MULTI_EVAL = MULTI_EVAL_ENABLED  # Use status determined during import
WRITE_OUTPUT = True  # If True, save best SBERT candidate output separately (per model)
# --- Ensure only one task is selected ---
if FLAG_33 == FLAG_34: logger.error("Exactly one of FLAG_33 or FLAG_34 must be True."); sys.exit(1)
TASK_NAME = "Predict_Body_Text_3.3" if FLAG_33 else "Generate_Conversation_3.4"
# --- End Flag Configuration ---

# --- General Configuration ---
if FLAG_33:
    INPUT_JSONL_PATH = "WithConversationPrompts_BodyPrediction_v2.jsonl"
    # INPUT_JSONL_PATH = "WithoutConversationPrompts_BodyPrediction_v2.jsonl"
    # INPUT_JSONL_PATH = "WithPseudoRandomConversationPrompts_BodyPrediction.jsonl"
    OUTPUT_DIR = f"eval_random_results_{TASK_NAME}"
    GENERATION_MAX_TOKENS = 256;
    NUM_CANDIDATES = 10;
    GENERATION_TEMPERATURE = 0.7
    logger.info(f"Running Task 3.3: Predict Body Text from {INPUT_JSONL_PATH}")
    if not MULTI_EVAL:
        logger.warning("MULTI_EVAL (ROUGE/BLEU/METEOR) is disabled.")
    else:
        logger.info("MULTI_EVAL (ROUGE/BLEU/METEOR) is enabled.")
else:  # FLAG_34 is True
    INPUT_JSONL_PATH = 'prompts_generate_conversation_final.jsonl'
    OUTPUT_DIR = f"eval_results_{TASK_NAME}"
    GENERATION_MAX_TOKENS = 2500;
    NUM_CANDIDATES = 10;
    GENERATION_TEMPERATURE = 0.8
    logger.info(f"Running Task 3.4: Generate Conversation from {INPUT_JSONL_PATH}")
    MULTI_EVAL = False  # Force disable ROUGE/BLEU for thread generation task
    logger.info("MULTI_EVAL (ROUGE/BLEU/METEOR) is disabled for Task 3.4.")

API_DELAY_SECONDS = 1;
PROMPT_DELAY_SECONDS = 0;
MAX_RETRIES = 1;
RETRY_DELAY_SECONDS = 3
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'
LOW_SIM_THRESHOLD = 0.1;
HIGH_SIM_THRESHOLD = 1.0;
DEFAULT_API_VERSION = "2024-02-15-preview";
REQUIRED_NEW_API_VERSION = "2024-12-01-preview"  # Example, adjust if needed

# --- Model Definitions ---
MODEL_CONFIGS = { # Input yours
}
# --- End Model Definitions ---

# --- Load SBERT Model ---
logger.info(f"Loading SBERT model: {SBERT_MODEL_NAME}...")
try:
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    logger.info(f"Using device: {device}")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=device)
    logger.info("SBERT model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading SBERT model: {e}");
    sys.exit(1)

# --- Initialize ROUGE Scorer (Conditional) ---
rouge_scorer_instance = None
meteor_available = True  # Global flag for METEOR availability

if MULTI_EVAL and FLAG_33:
    try:
        rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL', 'rouge1'], use_stemmer=True)
        logger.info("ROUGE (L,1) scorer initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize ROUGE scorer: {e}. ROUGE evaluation disabled.")
        MULTI_EVAL = False
        meteor_available = False

    # Check if wordnet is available for METEOR (already checked at startup, this is a secondary check)
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')  # Also check for omw-1.4
    except LookupError:
        logger.warning(
            "NLTK 'wordnet' or 'omw-1.4' not found after initial check during main setup. METEOR scoring will be disabled.")
        meteor_available = False


# --- End Initializations ---

# --- Helper Functions ---
def get_thread_concatenated_text(thread_nodes):
    if not thread_nodes or not isinstance(thread_nodes, list): return ""
    return " ".join([node.get("body", "") for node in thread_nodes if
                     isinstance(node, dict) and isinstance(node.get("body"), str) and node.get("body", "").strip()])


def get_thread_embedding(thread_nodes, sbert_model_instance, device_to_use):
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
            logger.error(f"OOM Error during SBERT encoding: {e}.")
        else:
            logger.error(f"Runtime error during SBERT encoding/mean calculation: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during SBERT encoding/mean calculation: {e}");
        return None


# --- API Call Functions ---
def call_azure_openai_with_retry(client, deployment_name, prompt_text, model_config, num_candidates, max_gen_tokens,
                                 temperature):
    candidate_responses = [];
    error_message = None
    api_version_str = model_config.get('api_version', DEFAULT_API_VERSION)
    requires_new_params = api_version_str >= REQUIRED_NEW_API_VERSION
    request_params = {"model": deployment_name, "messages": [{"role": "user", "content": prompt_text}],
                      "n": num_candidates, "temperature": temperature}
    if requires_new_params:
        # Example: Adjust if future API versions change parameter names like "max_tokens"
        # request_params["max_completion_tokens"] = max_gen_tokens # Hypothetical future param
        request_params["max_tokens"] = max_gen_tokens  # Sticking to current known params
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
                error_message = f"No choices received. Finish Reason: {finish_reason}";
                logger.warning(f"Model {deployment_name} returned no choices. Finish Reason: {finish_reason}");
                candidate_responses = [];
                break
            if finish_reason != 'stop' and not error_message:  # Only log warning if not already an error_message
                logger.warning(
                    f"Model {deployment_name} finished with reason: '{finish_reason}'. Output might be truncated or incomplete.")
            error_message = None  # Clear previous attempt's error if successful
            break
        except RateLimitError as e:
            error_message = f"RateLimitError: {e}"
        except APIError as e:
            err_code = getattr(e, 'status_code', 'N/A');
            err_msg = getattr(e, 'message', str(e));
            err_type = getattr(e, 'type', 'N/A')
            error_message = f"APIError: Status={err_code}, Type={err_type}, Message={err_msg[:150]}..."
            if 'DeploymentNotFound' in str(e) or (hasattr(e, 'code') and e.code == 'DeploymentNotFound'):
                break
            elif 'content_management_policy' in str(e).lower() or (hasattr(e, 'code') and e.code == 'content_filter'):
                error_message = f"Content Filter Error ({err_code}): {err_msg[:100]}...";
                break
            elif 'context_length_exceeded' in str(e).lower() or (
                    hasattr(e, 'code') and e.code == 'context_length_exceeded'):
                break
            elif err_code in [500, 503, 429,
                              408] and attempt < MAX_RETRIES:  # Retryable server-side or rate limit issues
                pass
            else:  # Non-retryable API error
                break
        except Exception as e:  # Catch-all for other unexpected errors during API call
            error_message = f"Unexpected Error during API call: {type(e).__name__} - {str(e)[:150]}...";
            break

        if error_message and attempt < MAX_RETRIES:  # If error and retries left
            logger.warning(
                f"Attempt {attempt + 1} fail({deployment_name}): {error_message}. Retrying in {RETRY_DELAY_SECONDS}s...");
            time.sleep(
                RETRY_DELAY_SECONDS)
        elif error_message:  # If error and no retries left
            logger.error(f"Final API Error after {attempt + 1} attempts ({deployment_name}): {error_message}")

    return candidate_responses, error_message


def call_google_vertex_with_retry(client_sdk, model_id, prompt_text, num_candidates, max_gen_tokens, temperature):
    if genai is None: return [], "Google GenAI SDK not imported."
    logger.warning("Google Vertex AI call function is not implemented.")  # Placeholder
    return [], "Google Call Function Not Implemented"


# +++ NEW HELPER FUNCTION for DOMAIN METRICS +++
def log_domain_metrics(current_domain_stats, logger_instance, meteor_is_available_flag):
    if not current_domain_stats:
        logger_instance.info("  No domain statistics collected yet.")
        return

    active_model_count = len(MODEL_CONFIGS)  # Get current number of models

    for domain, stats in sorted(current_domain_stats.items()):
        logger_instance.info(f"  --- Domain: {domain} ---")

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

            r1_evals = stats.get('successful_rouge1_evals', 0)
            avg_r1 = (stats.get('sum_rouge1_scores', 0.0) / r1_evals) if r1_evals > 0 else 0.0
            logger_instance.info(f"    Avg ROUGE-1 F1 Score: {avg_r1:.4f} (over {r1_evals} evals)")

            bleu_evals = stats.get('successful_bleu_evals', 0)
            avg_bleu = (stats.get('sum_bleu_scores', 0.0) / bleu_evals) if bleu_evals > 0 else 0.0
            logger_instance.info(f"    Avg BLEU Score: {avg_bleu:.4f} (over {bleu_evals} evals)")

            if meteor_is_available_flag:  # Use passed flag
                meteor_evals = stats.get('successful_meteor_evals', 0)
                avg_meteor = (stats.get('sum_meteor_scores', 0.0) / meteor_evals) if meteor_evals > 0 else 0.0
                logger_instance.info(f"    Avg METEOR Score: {avg_meteor:.4f} (over {meteor_evals} evals)")
        logger_instance.info("-" * 20)


# --- Main Evaluation Logic ---
def main():
    if not os.path.exists(INPUT_JSONL_PATH): logger.error(
        f"Input JSONL file not found at {INPUT_JSONL_PATH}"); sys.exit(1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Running Task: {TASK_NAME}")
    logger.info(f"Input file: {INPUT_JSONL_PATH}");
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Generating {NUM_CANDIDATES} candidates per prompt.")
    logger.info(f"Multi-Eval (ROUGE/BLEU/METEOR) enabled: {MULTI_EVAL} (Only applies if Task 3.3)")
    logger.info(f"Write Output (Best Candidate Per Model) enabled: {WRITE_OUTPUT}")

    timestamp_str = dt.now().strftime("%Y%m%d_%H%M%S")
    run_identifier = f"{'_test' if test_flag else ('_cut500' if CUT else '')}"
    combined_output_filename = os.path.join(OUTPUT_DIR,
                                            f"input_best_output_best_match_{TASK_NAME}{run_identifier}_{timestamp_str}.jsonl")
    logger.info(f"Saving combined input/best output/best match per prompt to: {combined_output_filename}")
    combined_output_file = None
    try:
        combined_output_file = open(combined_output_filename, 'w', encoding='utf-8')
    except IOError as e:
        logger.error(
            f"Critical Error: Could not open combined output file {combined_output_filename}: {e}. Combined output saving disabled.")

    prompts_data = [];
    line_num = 0
    logger.info(f"Reading prompts from {INPUT_JSONL_PATH}...")
    try:
        with open(INPUT_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1;
                line = line.strip();
                if not line: continue
                try:
                    data_dict = json.loads(line)
                    if not data_dict.get("prompt") or not data_dict.get("true_label"): logger.warning(
                        f"Skip L{line_num}. Missing 'prompt' or 'true_label'. Data: {str(data_dict)[:100]}"); continue
                    if FLAG_33 and not isinstance(data_dict["true_label"], str): logger.warning(
                        f"Skip L{line_num}. Task 3.3 needs string true_label."); continue
                    if FLAG_34 and not isinstance(data_dict["true_label"], list): logger.warning(
                        f"Skip L{line_num}. Task 3.4 needs list true_label (can be empty)."); continue
                    if FLAG_34 and not data_dict.get("target_post"): logger.warning(
                        f"Skip L{line_num}. Task 3.4 needs 'target_post'."); continue
                    prompts_data.append(data_dict)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skip L{line_num}. JSON decode error: {e}. Line: {line[:100]}...")
                except Exception as e:
                    logger.warning(f"Skip L{line_num}. Error parsing/validating line: {e}")
    except FileNotFoundError:
        logger.error(f"Critical Error: Input file not found at {INPUT_JSONL_PATH}");
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical Error reading input JSONL file {INPUT_JSONL_PATH}: {e}");
        sys.exit(1)
    if not prompts_data: logger.error("No valid prompts found after reading. Exiting."); sys.exit(0)
    logger.info(f"Read {len(prompts_data)} valid prompts.")

    limit = 5 if test_flag else (100 if CUT else None)
    prompts_to_process = prompts_data[:limit] if limit else prompts_data
    final_prompt_index = len(prompts_to_process) - 1
    run_mode = f"TEST MODE - First {len(prompts_to_process)}" if test_flag else (
        f"CUT MODE - First {len(prompts_to_process)}" if CUT else "FULL RUN")
    logger.info(f"\n{'=' * 10} Starting Interleaved Evaluation ({run_mode} - {TASK_NAME}) {'=' * 10}")

    active_models = list(MODEL_CONFIGS.keys())
    model_results = {name: [] for name in active_models}
    model_stats = {name: {'successful_preds': 0, 'total_similarity': 0.0,
                          'successful_rougeL': 0, 'total_rougeL': 0.0,
                          'successful_rouge1': 0, 'total_rouge1': 0.0,
                          'successful_bleu': 0, 'total_bleu': 0.0,
                          'successful_meteor': 0, 'total_meteor': 0.0,
                          'errors': 0, 'total': 0} for name in active_models}

    # +++ NEW: For domain (subreddit) specific stats +++
    domain_stats = {}

    initialized_clients = {};
    confusing_prompt_indices = set()

    best_candidate_files = {}
    if WRITE_OUTPUT:
        for model_name in active_models:
            safe_model_name = re.sub(r'[\\/*?:"<>|]', '_', model_name)
            filename = os.path.join(OUTPUT_DIR, f"best_candidates_{safe_model_name}_{TASK_NAME}{run_identifier}.jsonl")
            try:
                best_candidate_files[model_name] = open(filename, 'w', encoding='utf-8')
            except IOError as e:
                logger.error(f"Could not open best candidate file {filename}: {e}");
                best_candidate_files[
                    model_name] = None

    for i, prompt_entry in enumerate(prompts_to_process):
        user = prompt_entry.get("user", "Unknown")
        current_prompt_text = prompt_entry["prompt"]
        true_label_input = prompt_entry["true_label"]
        target_post_data = prompt_entry.get("target_post") # This is still used for target_post_id and other things
        target_post_id = target_post_data.get("id", f"Prompt_{i + 1}") if isinstance(target_post_data, dict) else f"Prompt_{i + 1}"
        
        # +++ CORRECTED: Get current subreddit directly from prompt_entry +++
        current_subreddit = prompt_entry.get("subreddit", "Unknown_Subreddit") 
        
        logger.info(
            f"\n--- Processing Prompt {i + 1}/{len(prompts_to_process)} (User: {user}, Target: {target_post_id if FLAG_34 else 'Single Reply'}) ---")
        logger.info(f"  Subreddit: {current_subreddit}") # Log corrected subreddit 

        # +++ NEW: Initialize domain_stats entry if new +++
        if current_subreddit not in domain_stats:
            domain_stats[current_subreddit] = {
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
            "prompt_index": i, "user": user, "task": TASK_NAME, "input_prompt": current_prompt_text,
            "true_label": true_label_input, "target_post": target_post_data if target_post_data else None,
            "subreddit": current_subreddit,  # Add subreddit to combined output
            "model_outputs": {}
        }

        true_label_embeddings = None;
        true_label_text_single = "";
        true_label_tokenized_list = []
        true_label_tokenized_for_meteor = []
        valid_true_labels_data = [];
        processing_error_occurred = False

        try:
            if FLAG_33:
                if not isinstance(true_label_input, str) or not true_label_input.strip(): raise ValueError(
                    "True label text is empty or not a string.")
                true_label_text_single = true_label_input
                true_label_embeddings = sbert_model.encode(true_label_text_single, convert_to_tensor=True,
                                                           device=device)
                valid_true_labels_data.append(
                    {"original_index": 0, "thread": true_label_text_single, "embedding": true_label_embeddings})
                if MULTI_EVAL:
                    true_label_tokenized_list = [nltk.word_tokenize(true_label_text_single.lower())]
                    if not true_label_tokenized_list[0]: raise ValueError(
                        "True label tokenization for BLEU resulted in empty list.")
                    if meteor_available:
                        tokenized_ref = nltk.word_tokenize(true_label_text_single.lower())
                        if not tokenized_ref: raise ValueError(
                            "True label tokenization for METEOR resulted in empty list.")
                        true_label_tokenized_for_meteor = [tokenized_ref]

            elif FLAG_34:
                if not isinstance(true_label_input, list): raise ValueError(
                    "True label for Task 3.4 must be a list of threads.")
                if not true_label_input: raise ValueError("True label list for Task 3.4 is empty.")
                embeddings_list_cpu = []
                for t_idx, true_thread in enumerate(true_label_input):
                    if not isinstance(true_thread, list): logger.warning(
                        f"    Skipping true_label thread index {t_idx} for prompt {i + 1}: Expected list, got {type(true_thread)}."); continue
                    embedding_cpu = get_thread_embedding(true_thread, sbert_model, 'cpu')
                    if embedding_cpu is not None:
                        valid_true_labels_data.append(
                            {"original_index": t_idx, "thread": true_thread, "embedding": embedding_cpu})
                        embeddings_list_cpu.append(embedding_cpu)
                    else:
                        logger.warning(f"    Could not encode true_label thread index {t_idx} for prompt {i + 1}.")
                if not valid_true_labels_data: raise ValueError("No valid true label threads could be encoded.")
                true_label_embeddings = torch.stack(embeddings_list_cpu).to(device)
            if not valid_true_labels_data or (
                    true_label_embeddings is None and not (
                    FLAG_34 and not true_label_input)):  # Ensure embeddings exist unless it's an empty true_label list for 3.4
                if not (FLAG_34 and not true_label_input):  # If not an empty true label list for 3.4, this is an error
                    raise ValueError("Failed to generate valid true label embeddings or no valid true labels.")
        except ValueError as ve:
            logger.error(f"  Error processing true label for prompt {i + 1}: {ve} - Skipping prompt evaluation.")
            processing_error_occurred = True;
            prompt_results_for_combined_json["error_processing_true_label"] = str(ve)
        except Exception as e:
            logger.error(
                f"  Unexpected error processing true label for prompt {i + 1}: {e} - Skipping prompt evaluation.")
            processing_error_occurred = True;
            prompt_results_for_combined_json["error_processing_true_label"] = f"Unexpected error: {e}"

        if processing_error_occurred:
            error_msg = prompt_results_for_combined_json["error_processing_true_label"]
            # +++ NEW: Update domain stats for true label processing error +++
            domain_stats[current_subreddit]['prompts_with_true_label_errors'] = domain_stats[current_subreddit].get(
                'prompts_with_true_label_errors', 0) + 1
            num_active_models = len(active_models)
            domain_stats[current_subreddit]['total_model_attempts_on_prompts'] = domain_stats[current_subreddit].get(
                'total_model_attempts_on_prompts', 0) + num_active_models
            domain_stats[current_subreddit]['post_api_eval_errors_count'] = domain_stats[current_subreddit].get(
                'post_api_eval_errors_count', 0) + num_active_models

            for model_name in active_models:
                stats = model_stats[model_name];
                stats['total'] += 1;
                stats['errors'] += 1;
                model_results[model_name].append(
                    {"prompt_index": i, "user": user, "prompt_preview": current_prompt_text[:100] + "...",
                     "true_label_preview": "ERROR", "prediction": "N/A", "best_sbert_score": "N/A",
                     "best_rougeL_score": "N/A", "best_rouge1_score": "N/A", "best_bleu_score": "N/A",
                     "best_meteor_score": "N/A", "api_error": f"Skipped due to True Label Error: {error_msg}"})
                prompt_results_for_combined_json["model_outputs"][model_name] = {"best_candidate_text": None,
                                                                                 "best_sbert_score": None,
                                                                                 "most_similar_true_label": None,
                                                                                 "similarity_to_best_true_label": None,
                                                                                 "api_error": f"Skipped due to True Label Error: {error_msg}"}
            if combined_output_file:
                try:
                    combined_output_file.write(json.dumps(prompt_results_for_combined_json, ensure_ascii=False) + '\n')
                except Exception as write_err:
                    logger.error(
                        f"Error writing combined results (true label error) for prompt {i + 1} to {combined_output_filename}: {write_err}")
            # +++ NEW: Periodic domain metrics output (also after error handling for a prompt) +++
            if (i + 1) % 1000 == 0 and (i + 1) < len(prompts_to_process):
                logger.info(f"\n{'=' * 10} Domain Metrics Snapshot at Prompt {i + 1} (after error handling) {'=' * 10}")
                log_domain_metrics(domain_stats, logger, meteor_available)  # Pass global meteor_available
            continue

        current_prompt_scores_sbert = {}
        for model_name in active_models:
            config = MODEL_CONFIGS[model_name];
            stats = model_stats[model_name];
            stats['total'] += 1

            # +++ NEW: Increment domain total model attempts for this specific model on this prompt +++
            domain_stats[current_subreddit]['total_model_attempts_on_prompts'] = domain_stats[current_subreddit].get(
                'total_model_attempts_on_prompts', 0) + 1

            logger.info(f"  Model: {model_name[:30]:<30} | Requesting {NUM_CANDIDATES} candidates...")
            api_client = initialized_clients.get(model_name);
            api_error = None;
            candidate_responses = []
            best_sbert_candidate_idx = -1;
            max_similarity_score = -1.0;
            best_sbert_candidate_preview = "Error"
            max_rougeL_score = -1.0;
            max_rouge1_score = -1.0;
            max_bleu_score = -1.0;
            max_meteor_score = -1.0
            best_sbert_candidate_raw_string = ""
            most_similar_true_label_content = None;
            similarity_to_best_true_label = None

            if api_client is None:
                if config["api_type"] == "azure":
                    try:
                        api_client = AzureOpenAI(api_key=config["api_key"], azure_endpoint=config["azure_endpoint"],
                                                 api_version=config["api_version"]);
                        initialized_clients[
                            model_name] = api_client
                    except Exception as e:
                        api_error = f"Azure Client Init Error: {e}";
                        logger.error(
                            f"Client Init Error for {model_name}! ({api_error})")
                elif config["api_type"] == "google_vertex":
                    api_error = "Google Client Init Not Implemented";
                    logger.error(
                        f"Client Init Error for {model_name}! ({api_error})")
                else:
                    api_error = f"Unknown api_type '{config['api_type']}'";
                    logger.error(
                        f"Client Init Error for {model_name}! ({api_error})")
                if api_error:
                    stats['errors'] += 1;
                    domain_stats[current_subreddit]['api_errors_count'] = domain_stats[current_subreddit].get(
                        'api_errors_count', 0) + 1  # +++ Domain Stat
                    current_prompt_scores_sbert[model_name] = -1.0
                    model_results[model_name].append(
                        {"prompt_index": i, "user": user, "prompt_preview": current_prompt_text[:100] + "...",
                         "true_label_preview": "N/A", "prediction": f"Error: {api_error}", "best_sbert_score": "N/A",
                         "best_rougeL_score": "N/A", "best_rouge1_score": "N/A", "best_bleu_score": "N/A",
                         "best_meteor_score": "N/A", "api_error": api_error})
                    prompt_results_for_combined_json["model_outputs"][model_name] = {"best_candidate_text": None,
                                                                                     "best_sbert_score": None,
                                                                                     "most_similar_true_label": None,
                                                                                     "similarity_to_best_true_label": None,
                                                                                     "api_error": api_error}
                    continue

            task_num_candidates = NUM_CANDIDATES;
            task_max_tokens = GENERATION_MAX_TOKENS;
            task_temperature = GENERATION_TEMPERATURE
            start_time = time.time()
            if config["api_type"] == "azure":
                if not isinstance(api_client, AzureOpenAI):
                    api_error = "Azure client became invalid (unexpected)";
                    logger.error(
                        f"Unexpected error for {model_name}: {api_error}")
                else:
                    candidate_responses, api_error = call_azure_openai_with_retry(api_client, config["deployment_name"],
                                                                                  current_prompt_text, config,
                                                                                  task_num_candidates, task_max_tokens,
                                                                                  task_temperature)
            elif config["api_type"] == "google_vertex":
                candidate_responses, api_error = call_google_vertex_with_retry(api_client, config.get("model_id"),
                                                                               # Use .get for model_id
                                                                               current_prompt_text, task_num_candidates,
                                                                               task_max_tokens, task_temperature)
            else:
                api_error = f"API call logic not implemented for api_type '{config['api_type']}'";
                logger.error(
                    f"API Call Error for {model_name}: {api_error}")
            end_time = time.time();
            duration = end_time - start_time

            # Reset scores for this model's attempt on this prompt
            max_similarity_score = -1.0;
            max_rougeL_score = -1.0;
            max_rouge1_score = -1.0;
            max_bleu_score = -1.0;
            max_meteor_score = -1.0
            best_sbert_candidate_idx = -1;
            best_sbert_candidate_raw_string = "";
            best_sbert_candidate_preview = "Error"
            most_similar_true_label_content = None;
            similarity_to_best_true_label = None

            if api_error:
                logger.warning(f"Model: {model_name[:30]:<30} | API Error! ({str(api_error)[:60]}...)")
                stats['errors'] += 1;
                domain_stats[current_subreddit]['api_errors_count'] = domain_stats[current_subreddit].get(
                    'api_errors_count', 0) + 1  # +++ Domain Stat
                best_sbert_candidate_preview = f"Error: {api_error}";
                current_prompt_scores_sbert[model_name] = -1.0
            elif not candidate_responses:
                no_cand_msg = "No candidates received (API call successful)"
                logger.warning(f"Model: {model_name[:30]:<30} | Pred Error! {no_cand_msg}.")
                stats['errors'] += 1;
                domain_stats[current_subreddit]['post_api_eval_errors_count'] = domain_stats[current_subreddit].get(
                    'post_api_eval_errors_count', 0) + 1  # +++ Domain Stat
                best_sbert_candidate_preview = f"Error: {no_cand_msg}";
                current_prompt_scores_sbert[model_name] = -1.0;
                api_error = no_cand_msg
            else:
                valid_candidates_data = []
                try:
                    num_received = len(candidate_responses)
                    logger.debug(f"    Processing {num_received} candidates received from {model_name}...")
                    with torch.no_grad():
                        for idx, cand_str in enumerate(candidate_responses):
                            if not isinstance(cand_str, str) or not cand_str.strip():
                                logger.warning(
                                    f"    Cand #{idx + 1}/{num_received} for {model_name} is not a valid string or is empty. Type: {type(cand_str).__name__}. Content (start): '{str(cand_str)[:200]}...'. Skipping.")
                                continue
                            embedding = None;
                            struct = None;
                            try:
                                if FLAG_34:
                                    try:
                                        struct = json.loads(cand_str)
                                        if not isinstance(struct, list): logger.warning(
                                            f"    Cand #{idx + 1}/{num_received} for {model_name} (Task 3.4) JSON parsed but is not a list ({type(struct)}). Content (start): '{cand_str}...'. Skipping."); continue
                                        embedding = get_thread_embedding(struct, sbert_model, device)
                                        if embedding is None: logger.warning(
                                            f"    Cand #{idx + 1}/{num_received} for {model_name} (Task 3.4) valid JSON list but failed to embed. Content (start): '{cand_str[:200]}...'. Skipping."); continue
                                    except json.JSONDecodeError as json_e:
                                        logger.warning(
                                            f"    Cand #{idx + 1}/{num_received} for {model_name} (Task 3.4) failed JSON parsing: {json_e}. Content (start): '{cand_str[:200]}...'. Skipping.");
                                        continue
                                elif FLAG_33:
                                    embedding = sbert_model.encode(cand_str, convert_to_tensor=True, device=device)
                                    if not torch.is_tensor(embedding): logger.warning(
                                        f"    Cand #{idx + 1}/{num_received} for {model_name} (Task 3.3) failed to generate a valid embedding tensor. Content (start): '{cand_str[:200]}...'. Skipping."); continue

                                if embedding is not None:  # Check if embedding was successfully created
                                    valid_candidates_data.append({"idx": idx, "raw_str": cand_str, "struct": struct,
                                                                  "embedding": embedding.to(device) if torch.is_tensor(
                                                                      embedding) else embedding})  # Ensure tensor embeddings are on device
                                else:  # Should not happen if checks above are thorough, but as a safeguard
                                    logger.warning(
                                        f"    Cand #{idx + 1}/{num_received} for {model_name} did not produce a valid embedding. Skipping.")
                                    continue

                            except Exception as e:  # Catch errors during individual candidate processing
                                logger.error(
                                    f"    Unexpected error processing candidate #{idx + 1}/{num_received} for {model_name}: {e}. Content (start): '{cand_str[:200]}...'. Skipping.");
                                continue

                    if valid_candidates_data:
                        logger.debug(
                            f"    Successfully processed {len(valid_candidates_data)}/{num_received} candidates for {model_name}.")
                        # Ensure all embeddings are tensors before stacking for SBERT
                        candidate_embeddings_list = [d['embedding'] for d in valid_candidates_data if
                                                     torch.is_tensor(d['embedding'])]
                        if not candidate_embeddings_list:
                            raise ValueError("No valid tensor embeddings found among processed candidates for SBERT.")

                        candidate_embeddings_tensor = torch.stack(candidate_embeddings_list).to(device)

                        sbert_sim_matrix = util.cos_sim(candidate_embeddings_tensor, true_label_embeddings.to(device))
                        max_sbert_scores_per_candidate, _ = torch.max(sbert_sim_matrix, dim=1)
                        best_sbert_candidate_local_idx_in_valid = torch.argmax(max_sbert_scores_per_candidate).item()

                        # Find the original index from valid_candidates_data
                        # This requires mapping local_idx (from candidate_embeddings_tensor) back to valid_candidates_data idx
                        # This assumes candidate_embeddings_list was formed in the same order as valid_candidates_data entries that had tensor embeddings

                        # Simpler: iterate valid_candidates_data to find max_sbert_score if the above mapping is tricky
                        # For now, assume best_sbert_candidate_local_idx_in_valid maps directly to valid_candidates_data if all were tensors
                        # Let's refine this to be safer:

                        sbert_scores_for_valid_candidates = []
                        for cand_data_item in valid_candidates_data:
                            if torch.is_tensor(cand_data_item['embedding']):
                                sim_to_true_labels = util.cos_sim(cand_data_item['embedding'].to(device),
                                                                  true_label_embeddings.to(device))
                                sbert_scores_for_valid_candidates.append(torch.max(sim_to_true_labels).item())
                            else:  # Should not happen if filtered earlier
                                sbert_scores_for_valid_candidates.append(-1.0)  # Placeholder for non-tensor error

                        if not sbert_scores_for_valid_candidates:
                            raise ValueError("No SBERT scores could be calculated for valid candidates.")

                        max_similarity_score = max(sbert_scores_for_valid_candidates)
                        best_sbert_candidate_local_idx = sbert_scores_for_valid_candidates.index(max_similarity_score)

                        best_sbert_data = valid_candidates_data[best_sbert_candidate_local_idx]
                        best_sbert_candidate_idx = best_sbert_data['idx']  # Original index from API response
                        best_sbert_candidate_raw_string = best_sbert_data['raw_str']

                        # Determine which true label was most similar to this best candidate
                        best_cand_embedding_on_device = best_sbert_data['embedding'].to(device) if torch.is_tensor(
                            best_sbert_data['embedding']) else sbert_model.encode(best_sbert_data['raw_str'],
                                                                                  convert_to_tensor=True, device=device)

                        sim_scores_vs_all_true = util.cos_sim(best_cand_embedding_on_device,
                                                              true_label_embeddings.to(device))
                        best_true_label_local_idx = torch.argmax(
                            sim_scores_vs_all_true[0]).item()  # sim_scores_vs_all_true will be 1xN

                        most_similar_true_label_data = valid_true_labels_data[best_true_label_local_idx]
                        most_similar_true_label_content = most_similar_true_label_data['thread']
                        similarity_to_best_true_label = max_similarity_score

                        stats['successful_preds'] += 1;
                        stats['total_similarity'] += max_similarity_score
                        current_prompt_scores_sbert[model_name] = max_similarity_score
                        # +++ NEW: Update domain SBERT stats +++
                        if max_similarity_score != -1.0:
                            domain_stats[current_subreddit]['successful_sbert_evals'] = domain_stats[
                                                                                            current_subreddit].get(
                                'successful_sbert_evals', 0) + 1
                            domain_stats[current_subreddit]['sum_sbert_scores'] = domain_stats[current_subreddit].get(
                                'sum_sbert_scores', 0.0) + max_similarity_score

                        if FLAG_33 and MULTI_EVAL:
                            max_rougeL_score_cand, max_rouge1_score_cand, max_bleu_score_cand, max_meteor_score_cand = -1.0, -1.0, -1.0, -1.0
                            # Use the best_sbert_candidate_raw_string for ROUGE/BLEU/METEOR against true_label_text_single
                            # Or iterate all valid candidates to find the one that maximizes R/B/M, not necessarily the SBERT best one.
                            # The current code finds the R/B/M scores for the SBERT-best candidate.
                            # To match original intent: iterate all valid candidates again to find best for R/B/M.

                            # Resetting scores for this section, as we iterate candidates for these metrics
                            temp_max_rougeL, temp_max_rouge1, temp_max_bleu, temp_max_meteor = 0.0, 0.0, 0.0, 0.0

                            for cand_data_item in valid_candidates_data:  # Iterate through all valid candidates
                                cand_text_for_eval = cand_data_item['raw_str']
                                if not cand_text_for_eval.strip(): continue

                                if rouge_scorer_instance:
                                    try:
                                        rouge_scores = rouge_scorer_instance.score(true_label_text_single,
                                                                                   cand_text_for_eval)
                                        current_rougeL = rouge_scores['rougeL'].fmeasure
                                        current_rouge1 = rouge_scores['rouge1'].fmeasure
                                        if current_rougeL > temp_max_rougeL: temp_max_rougeL = current_rougeL
                                        if current_rouge1 > temp_max_rouge1: temp_max_rouge1 = current_rouge1
                                    except Exception:
                                        pass  # Ignore calc error for one candidate

                                candidate_tokens_for_eval = nltk.word_tokenize(cand_text_for_eval.lower())
                                if not candidate_tokens_for_eval: continue

                                try:  # BLEU
                                    current_bleu = sentence_bleu(true_label_tokenized_list, candidate_tokens_for_eval,
                                                                 smoothing_function=SmoothingFunction().method1)
                                    if current_bleu > temp_max_bleu: temp_max_bleu = current_bleu
                                except Exception:
                                    pass

                                if meteor_available and true_label_tokenized_for_meteor:  # METEOR
                                    try:
                                        current_meteor = meteor_score(true_label_tokenized_for_meteor,
                                                                      candidate_tokens_for_eval)
                                        if current_meteor > temp_max_meteor: temp_max_meteor = current_meteor
                                    except Exception:
                                        pass

                            max_rougeL_score = temp_max_rougeL if temp_max_rougeL > 0 else -1.0  # Keep -1 if no score
                            max_rouge1_score = temp_max_rouge1 if temp_max_rouge1 > 0 else -1.0
                            max_bleu_score = temp_max_bleu if temp_max_bleu > 0 else -1.0
                            max_meteor_score = temp_max_meteor if temp_max_meteor > 0 else -1.0

                            if max_rougeL_score >= 0.0:
                                stats['successful_rougeL'] += 1;
                                stats['total_rougeL'] += max_rougeL_score
                                domain_stats[current_subreddit]['successful_rougeL_evals'] = domain_stats[
                                                                                                 current_subreddit].get(
                                    'successful_rougeL_evals', 0) + 1
                                domain_stats[current_subreddit]['sum_rougeL_scores'] = domain_stats[
                                                                                           current_subreddit].get(
                                    'sum_rougeL_scores', 0.0) + max_rougeL_score
                            if max_rouge1_score >= 0.0:
                                stats['successful_rouge1'] += 1;
                                stats['total_rouge1'] += max_rouge1_score
                                domain_stats[current_subreddit]['successful_rouge1_evals'] = domain_stats[
                                                                                                 current_subreddit].get(
                                    'successful_rouge1_evals', 0) + 1
                                domain_stats[current_subreddit]['sum_rouge1_scores'] = domain_stats[
                                                                                           current_subreddit].get(
                                    'sum_rouge1_scores', 0.0) + max_rouge1_score
                            if max_bleu_score >= 0.0:
                                stats['successful_bleu'] += 1;
                                stats['total_bleu'] += max_bleu_score
                                domain_stats[current_subreddit]['successful_bleu_evals'] = domain_stats[
                                                                                               current_subreddit].get(
                                    'successful_bleu_evals', 0) + 1
                                domain_stats[current_subreddit]['sum_bleu_scores'] = domain_stats[
                                                                                         current_subreddit].get(
                                    'sum_bleu_scores', 0.0) + max_bleu_score
                            if meteor_available and max_meteor_score >= 0.0:
                                stats['successful_meteor'] += 1;
                                stats['total_meteor'] += max_meteor_score
                                domain_stats[current_subreddit]['successful_meteor_evals'] = domain_stats[
                                                                                                 current_subreddit].get(
                                    'successful_meteor_evals', 0) + 1
                                domain_stats[current_subreddit]['sum_meteor_scores'] = domain_stats[
                                                                                           current_subreddit].get(
                                    'sum_meteor_scores', 0.0) + max_meteor_score

                        if FLAG_33:
                            best_sbert_candidate_preview = best_sbert_candidate_raw_string[:100] + "..."
                        else:
                            first_comment_body = "(Invalid Struct)"
                            try:
                                if isinstance(best_sbert_data['struct'], list) and best_sbert_data['struct']:
                                    first_node = best_sbert_data['struct'][0]
                                    if isinstance(first_node, dict): first_comment_body = first_node.get('body',
                                                                                                         '(Body Missing)')[
                                                                                          :80]
                            except Exception as preview_e:
                                logger.warning(
                                    f"    Error generating preview for {model_name}: {preview_e}"); first_comment_body = "(Preview Error)"
                            best_sbert_candidate_preview = f"BestCandFirstComment: {first_comment_body}..."

                        log_info = f"Best SBERT Sim: {max_similarity_score:.4f} (Cand Original Idx #{best_sbert_candidate_idx + 1} vs TrueLabel Idx {most_similar_true_label_data['original_index']})"
                        if FLAG_33 and MULTI_EVAL:
                            log_info += f" | Best R-L: {max_rougeL_score:.4f} | Best R-1: {max_rouge1_score:.4f} | Best BLEU: {max_bleu_score:.4f}"
                            if meteor_available: log_info += f" | Best METEOR: {max_meteor_score:.4f}"
                        log_info += f" | Time(API+Eval): {duration:.2f}s"
                        logger.info(f"Model: {model_name[:30]:<30} | {log_info}")

                        if WRITE_OUTPUT and best_candidate_files.get(model_name) and best_sbert_candidate_idx != -1:
                            try:
                                output_data_single_model = {"prompt_index": i, "user": user,
                                                            "target_id": target_post_id if FLAG_34 else None,
                                                            "true_label": true_label_input,
                                                            "subreddit": current_subreddit,
                                                            "best_sbert_candidate_text": best_sbert_candidate_raw_string,
                                                            "best_sbert_score": max_similarity_score}
                                best_candidate_files[model_name].write(
                                    json.dumps(output_data_single_model, ensure_ascii=False) + '\n')
                            except Exception as write_err:
                                logger.error(
                                    f"Error writing best candidate file for {model_name}, prompt {i + 1}: {write_err}")
                    else:  # No valid_candidates_data
                        no_valid_cand_msg = f"No valid candidates processed out of {num_received if 'num_received' in locals() else 'unknown'} received."
                        logger.warning(f"Model: {model_name[:30]:<30} | Pred Error! {no_valid_cand_msg}")
                        stats['errors'] += 1;
                        domain_stats[current_subreddit]['post_api_eval_errors_count'] = domain_stats[
                                                                                            current_subreddit].get(
                            'post_api_eval_errors_count', 0) + 1  # +++ Domain Stat
                        best_sbert_candidate_preview = f"Error: {no_valid_cand_msg}";
                        max_similarity_score = -1.0;
                        max_rougeL_score = -1.0;
                        max_rouge1_score = -1.0;
                        max_bleu_score = -1.0;
                        max_meteor_score = -1.0
                        api_error = no_valid_cand_msg;
                        current_prompt_scores_sbert[model_name] = -1.0

                except ValueError as ve:  # Errors during similarity calculation or candidate processing logic
                    eval_err_msg = f"Eval Processing Error (e.g. Sim Calc): {ve}";
                    logger.error(f"{eval_err_msg} for {model_name}!");
                    stats['errors'] += 1;
                    domain_stats[current_subreddit]['post_api_eval_errors_count'] = domain_stats[current_subreddit].get(
                        'post_api_eval_errors_count', 0) + 1  # +++ Domain Stat
                    best_sbert_candidate_preview = eval_err_msg;
                    max_similarity_score = -1.0;
                    max_rougeL_score = -1.0;
                    max_rouge1_score = -1.0;
                    max_bleu_score = -1.0;
                    max_meteor_score = -1.0;
                    api_error = eval_err_msg;
                    current_prompt_scores_sbert[model_name] = -1.0
                except RuntimeError as rte:  # CUDA/MPS OOM
                    oom_err_msg = f"CUDA/MPS Runtime Error (Sim Calc / potential OOM): {rte}";
                    logger.error(f"Eval Runtime Error for {model_name}! {oom_err_msg}");
                    stats['errors'] += 1;
                    domain_stats[current_subreddit]['post_api_eval_errors_count'] = domain_stats[current_subreddit].get(
                        'post_api_eval_errors_count', 0) + 1  # +++ Domain Stat
                    best_sbert_candidate_preview = oom_err_msg;
                    max_similarity_score = -1.0;
                    max_rougeL_score = -1.0;
                    max_rouge1_score = -1.0;
                    max_bleu_score = -1.0;
                    max_meteor_score = -1.0;
                    api_error = oom_err_msg;
                    current_prompt_scores_sbert[model_name] = -1.0
                except Exception as e:  # Other unexpected errors
                    unexp_err_msg = f"Unexpected Eval Error (Sim Calc): {type(e).__name__}: {str(e)[:100]}...";
                    logger.error(f"Unexpected Eval Error for {model_name}! {unexp_err_msg}");
                    stats['errors'] += 1;
                    domain_stats[current_subreddit]['post_api_eval_errors_count'] = domain_stats[current_subreddit].get(
                        'post_api_eval_errors_count', 0) + 1  # +++ Domain Stat
                    best_sbert_candidate_preview = unexp_err_msg;
                    max_similarity_score = -1.0;
                    max_rougeL_score = -1.0;
                    max_rouge1_score = -1.0;
                    max_bleu_score = -1.0;
                    max_meteor_score = -1.0;
                    api_error = unexp_err_msg;
                    current_prompt_scores_sbert[model_name] = -1.0

            # Log running stats for model
            successful_sbert_calcs = stats['successful_preds']
            running_avg_sim = (
                        stats['total_similarity'] / successful_sbert_calcs) if successful_sbert_calcs > 0 else 0.0
            log_msg = f"  Model: {model_name[:30]:<30} | Running Avg SBERT: {running_avg_sim:.4f} ({successful_sbert_calcs} valid out of {stats['total']})"
            if FLAG_33 and MULTI_EVAL:
                running_avg_rougeL = (stats['total_rougeL'] / stats['successful_rougeL']) if stats[
                                                                                                 'successful_rougeL'] > 0 else 0.0
                running_avg_rouge1 = (stats['total_rouge1'] / stats['successful_rouge1']) if stats[
                                                                                                 'successful_rouge1'] > 0 else 0.0
                running_avg_bleu = (stats['total_bleu'] / stats['successful_bleu']) if stats[
                                                                                           'successful_bleu'] > 0 else 0.0
                running_avg_meteor = (stats['total_meteor'] / stats['successful_meteor']) if meteor_available and stats[
                    'successful_meteor'] > 0 else 0.0
                log_msg += f" | Avg R-L: {running_avg_rougeL:.4f} ({stats['successful_rougeL']}) | Avg R-1: {running_avg_rouge1:.4f} ({stats['successful_rouge1']}) | Avg BLEU: {running_avg_bleu:.4f} ({stats['successful_bleu']})"
                if meteor_available: log_msg += f" | Avg METEOR: {running_avg_meteor:.4f} ({stats['successful_meteor']})"
            log_msg += f" | Errors: {stats['errors']}"
            logger.info(log_msg)

            true_label_display = true_label_input[:80] + "..." if isinstance(true_label_input, str) else (
                f"{len(true_label_input)} threads" if isinstance(true_label_input, list) else "N/A")

            model_results[model_name].append({
                "prompt_index": i, "user": user, "prompt_preview": current_prompt_text[:100] + "...",
                "true_label_preview": true_label_display, "prediction": best_sbert_candidate_preview,
                "best_sbert_score": f"{max_similarity_score:.6f}" if max_similarity_score != -1.0 else "N/A",
                "best_rougeL_score": f"{max_rougeL_score:.6f}" if FLAG_33 and MULTI_EVAL and max_rougeL_score >= 0.0 else "N/A",
                "best_rouge1_score": f"{max_rouge1_score:.6f}" if FLAG_33 and MULTI_EVAL and max_rouge1_score >= 0.0 else "N/A",
                "best_bleu_score": f"{max_bleu_score:.6f}" if FLAG_33 and MULTI_EVAL and max_bleu_score >= 0.0 else "N/A",
                "best_meteor_score": f"{max_meteor_score:.6f}" if FLAG_33 and MULTI_EVAL and meteor_available and max_meteor_score >= 0.0 else "N/A",
                "api_error": str(api_error) if api_error else "",
            })
            prompt_results_for_combined_json["model_outputs"][model_name] = {
                "best_candidate_text": best_sbert_candidate_raw_string if best_sbert_candidate_idx != -1 else None,
                "best_sbert_score": max_similarity_score if max_similarity_score != -1.0 else None,
                "most_similar_true_label": most_similar_true_label_content,
                "similarity_to_best_true_label": similarity_to_best_true_label,
                "api_error": str(api_error) if api_error else None
            }
            if device != 'cpu':
                try:
                    if 'candidate_embeddings_tensor' in locals(): del candidate_embeddings_tensor
                    if 'sbert_sim_matrix' in locals(): del sbert_sim_matrix
                    if 'max_sbert_scores_per_candidate' in locals(): del max_sbert_scores_per_candidate
                    torch.cuda.empty_cache()
                except NameError:
                    pass
                except Exception as e:
                    logger.warning(f"Error during GPU memory cleanup: {e}")
            if API_DELAY_SECONDS > 0: time.sleep(API_DELAY_SECONDS)

        if combined_output_file:
            try:
                combined_output_file.write(json.dumps(prompt_results_for_combined_json, ensure_ascii=False) + '\n')
            except TypeError as te:
                logger.error(
                    f"Error serializing combined results for prompt {i + 1} to JSON: {te}. Data (types): { {k: type(v).__name__ for k, v in prompt_results_for_combined_json.items()} }")
            except Exception as write_err:
                logger.error(
                    f"Error writing combined results for prompt {i + 1} to {combined_output_filename}: {write_err}")

        if len(current_prompt_scores_sbert) == len(active_models):
            valid_scores = [s for s in current_prompt_scores_sbert.values() if s != -1.0]
            if len(valid_scores) == len(active_models):  # All models must have produced a valid score
                all_low = all(s < LOW_SIM_THRESHOLD for s in valid_scores);
                all_high = all(s > HIGH_SIM_THRESHOLD for s in valid_scores)
                if all_low or all_high: confusing_prompt_indices.add(i); logger.warning(
                    f"--- Prompt {i + 1} flagged as CONFUSING (SBERT: {'ALL < ' + str(LOW_SIM_THRESHOLD) if all_low else 'ALL > ' + str(HIGH_SIM_THRESHOLD)}) ---")

        if PROMPT_DELAY_SECONDS > 0 and i < final_prompt_index: logger.info(
            f"--- Delaying {PROMPT_DELAY_SECONDS}s ---"); time.sleep(PROMPT_DELAY_SECONDS)
        if device != 'cpu':
            try:
                if 'true_label_embeddings' in locals() and true_label_embeddings is not None: del true_label_embeddings; torch.cuda.empty_cache()
            except NameError:
                pass
            except Exception as e:
                logger.warning(f"Error during GPU memory cleanup after prompt: {e}")

        # +++ NEW: Periodic domain metrics output +++
        if (i + 1) % 1000 == 0 and (i + 1) < len(
                prompts_to_process):  # Ensure not to print at the very end if it aligns with final print
            logger.info(f"\n{'=' * 10} Domain Metrics Snapshot at Prompt {i + 1} {'=' * 10}")
            log_domain_metrics(domain_stats, logger, meteor_available)  # Pass global meteor_available

    if WRITE_OUTPUT:
        for model_name, file_handle in best_candidate_files.items():
            if file_handle:
                try:
                    file_handle.close()
                except Exception as e:
                    logger.warning(f"Ignoring error while closing best candidate file for {model_name}: {e}")
    if combined_output_file:
        try:
            combined_output_file.close(); logger.info(f"Closed combined input/output file: {combined_output_filename}")
        except Exception as e:
            logger.error(f"Error closing combined input/output file {combined_output_filename}: {e}")

    # +++ NEW: Final domain metrics output +++
    logger.info(f"\n{'=' * 10} Final Domain Metrics ({run_mode}) {'=' * 10}")
    log_domain_metrics(domain_stats, logger, meteor_available)  # Pass global meteor_available

    logger.info(
        f"\n{'=' * 10} Evaluation Finished ({run_mode}) - Calculating Final Metrics & Writing Results {'=' * 10}")
    if confusing_prompt_indices:
        logger.info(
            f"Identified {len(confusing_prompt_indices)} confusing prompts (Indices based on 0-start): {sorted(list(confusing_prompt_indices))}")
    else:
        logger.info("No confusing prompts identified based on SBERT scores.")
    overall_summary = []

    summary_fieldnames = [
        "model_name", "task", "total_prompts_processed", "successful_predictions", "api_pred_eval_errors",
        "avg_best_sbert_score", "overall_success_rate",
        "avg_best_rougeL_score", "avg_best_rouge1_score",
        "avg_best_bleu_score", "avg_best_meteor_score",
        "confusing_samples_excluded", "avg_best_sbert_score_filtered",
        "overall_success_rate_filtered",
        "avg_best_rougeL_score_filtered", "avg_best_rouge1_score_filtered",
        "avg_best_bleu_score_filtered", "avg_best_meteor_score_filtered",
        "output_file"
    ]

    for model_name in active_models:
        results = model_results[model_name];
        stats = model_stats[model_name]
        safe_model_name = re.sub(r'[\\/*?:"<>|]', '_', model_name);
        eval_output_filename = os.path.join(OUTPUT_DIR, f"eval_{safe_model_name}_{TASK_NAME}{run_identifier}.csv")
        logger.info(f"\n--- Processing Final Results for: {model_name} ---")

        if results:
            logger.info(f"  Writing {len(results)} results summary to {eval_output_filename}...")
            try:
                with open(eval_output_filename, 'w', newline='', encoding='utf-8-sig') as f:
                    result_fieldnames = ["prompt_index", "user", "prompt_preview", "true_label_preview", "prediction",
                                         "best_sbert_score", "best_rougeL_score", "best_rouge1_score",
                                         "best_bleu_score", "best_meteor_score", "api_error"]
                    writer = csv.DictWriter(f, fieldnames=result_fieldnames, extrasaction='ignore')
                    writer.writeheader();
                    writer.writerows(results)
            except IOError as e:
                logger.error(f"Error writing output file {eval_output_filename}: {e}")
        else:
            logger.warning(f"  No results recorded for {model_name}. Skipping CSV write.")

        total_processed = stats['total'];
        api_or_pred_errors = stats['errors'];
        successful_sbert_preds = stats['successful_preds']
        avg_best_sbert_score = (
                    stats['total_similarity'] / successful_sbert_preds) if successful_sbert_preds > 0 else 0.0
        success_rate_sbert = ((
                                          total_processed - api_or_pred_errors) / total_processed) * 100 if total_processed > 0 else 0.0

        avg_best_rougeL_score = (stats['total_rougeL'] / stats['successful_rougeL']) if stats.get('successful_rougeL',
                                                                                                  0) > 0 else 0.0
        avg_best_rouge1_score = (stats['total_rouge1'] / stats['successful_rouge1']) if stats.get('successful_rouge1',
                                                                                                  0) > 0 else 0.0
        avg_best_bleu_score = (stats['total_bleu'] / stats['successful_bleu']) if stats.get('successful_bleu',
                                                                                            0) > 0 else 0.0
        avg_best_meteor_score = (stats['total_meteor'] / stats['successful_meteor']) if meteor_available and stats.get(
            'successful_meteor', 0) > 0 else 0.0

        filtered_results = [r for r in results if r.get('prompt_index', -1) not in confusing_prompt_indices]
        total_processed_filtered = len(filtered_results);
        errors_filtered = sum(1 for r in filtered_results if r['prediction'].startswith("Error:") or r.get('api_error'))
        successful_sbert_preds_filtered = sum(1 for r in filtered_results if
                                              not r['prediction'].startswith("Error:") and not r.get(
                                                  'api_error') and r.get('best_sbert_score', 'N/A') != 'N/A')
        total_similarity_filtered = sum(
            float(r['best_sbert_score']) for r in filtered_results if r.get('best_sbert_score', 'N/A') != 'N/A')
        avg_best_sbert_score_filtered = (
                    total_similarity_filtered / successful_sbert_preds_filtered) if successful_sbert_preds_filtered > 0 else 0.0
        success_rate_sbert_filtered = ((
                                                   total_processed_filtered - errors_filtered) / total_processed_filtered) * 100 if total_processed_filtered > 0 else 0.0

        avg_best_rougeL_score_filtered, avg_best_rouge1_score_filtered, avg_best_bleu_score_filtered, avg_best_meteor_score_filtered = 0.0, 0.0, 0.0, 0.0
        if FLAG_33 and MULTI_EVAL:
            successful_rougeL_filtered = sum(1 for r in filtered_results if r.get('best_rougeL_score', 'N/A') != 'N/A');
            total_rougeL_filtered = sum(
                float(r['best_rougeL_score']) for r in filtered_results if r.get('best_rougeL_score', 'N/A') != 'N/A')
            avg_best_rougeL_score_filtered = (
                        total_rougeL_filtered / successful_rougeL_filtered) if successful_rougeL_filtered > 0 else 0.0
            successful_rouge1_filtered = sum(1 for r in filtered_results if r.get('best_rouge1_score', 'N/A') != 'N/A');
            total_rouge1_filtered = sum(
                float(r['best_rouge1_score']) for r in filtered_results if r.get('best_rouge1_score', 'N/A') != 'N/A')
            avg_best_rouge1_score_filtered = (
                        total_rouge1_filtered / successful_rouge1_filtered) if successful_rouge1_filtered > 0 else 0.0
            successful_bleu_filtered = sum(1 for r in filtered_results if r.get('best_bleu_score', 'N/A') != 'N/A');
            total_bleu_filtered = sum(
                float(r['best_bleu_score']) for r in filtered_results if r.get('best_bleu_score', 'N/A') != 'N/A')
            avg_best_bleu_score_filtered = (
                        total_bleu_filtered / successful_bleu_filtered) if successful_bleu_filtered > 0 else 0.0
            if meteor_available:
                successful_meteor_filtered = sum(
                    1 for r in filtered_results if r.get('best_meteor_score', 'N/A') != 'N/A');
                total_meteor_filtered = sum(float(r['best_meteor_score']) for r in filtered_results if
                                            r.get('best_meteor_score', 'N/A') != 'N/A')
                avg_best_meteor_score_filtered = (
                            total_meteor_filtered / successful_meteor_filtered) if successful_meteor_filtered > 0 else 0.0

        summary = {
            "model_name": model_name, "task": TASK_NAME, "total_prompts_processed": total_processed,
            "successful_predictions": successful_sbert_preds, "api_pred_eval_errors": api_or_pred_errors,
            "avg_best_sbert_score": f"{avg_best_sbert_score:.6f}", "overall_success_rate": f"{success_rate_sbert:.2f}%",
            "avg_best_rougeL_score": f"{avg_best_rougeL_score:.6f}" if FLAG_33 and MULTI_EVAL else "N/A",
            "avg_best_rouge1_score": f"{avg_best_rouge1_score:.6f}" if FLAG_33 and MULTI_EVAL else "N/A",
            "avg_best_bleu_score": f"{avg_best_bleu_score:.6f}" if FLAG_33 and MULTI_EVAL else "N/A",
            "avg_best_meteor_score": f"{avg_best_meteor_score:.6f}" if FLAG_33 and MULTI_EVAL and meteor_available else "N/A",
            "confusing_samples_excluded": len(confusing_prompt_indices),
            "avg_best_sbert_score_filtered": f"{avg_best_sbert_score_filtered:.6f}",
            "overall_success_rate_filtered": f"{success_rate_sbert_filtered:.2f}%",
            "avg_best_rougeL_score_filtered": f"{avg_best_rougeL_score_filtered:.6f}" if FLAG_33 and MULTI_EVAL else "N/A",
            "avg_best_rouge1_score_filtered": f"{avg_best_rouge1_score_filtered:.6f}" if FLAG_33 and MULTI_EVAL else "N/A",
            "avg_best_bleu_score_filtered": f"{avg_best_bleu_score_filtered:.6f}" if FLAG_33 and MULTI_EVAL else "N/A",
            "avg_best_meteor_score_filtered": f"{avg_best_meteor_score_filtered:.6f}" if FLAG_33 and MULTI_EVAL and meteor_available else "N/A",
            "output_file": eval_output_filename
        }
        overall_summary.append(summary)

        logger.info(f"  Final Summary for {model_name} ({TASK_NAME}):")
        logger.info(f"    Total Prompts Attempted: {summary['total_prompts_processed']}")
        logger.info(f"    Successful Evals (API+Parse+SBERT OK): {summary['successful_predictions']}")
        logger.info(f"    API/Parse/Eval Errors: {summary['api_pred_eval_errors']}")
        logger.info(f"    Overall Success Rate (Non-Error): {summary['overall_success_rate']}")
        logger.info(f"    --- SBERT Similarity (Avg Best) ---");
        logger.info(f"    Overall: {summary['avg_best_sbert_score']}");
        logger.info(
            f"    Filtered: {summary['avg_best_sbert_score_filtered']} (excluding {len(confusing_prompt_indices)} confusing)")
        if FLAG_33 and MULTI_EVAL:
            logger.info(f"    --- ROUGE-L F1 (Avg Best) ---");
            logger.info(f"    Overall: {summary['avg_best_rougeL_score']}");
            logger.info(f"    Filtered: {summary['avg_best_rougeL_score_filtered']}")
            logger.info(f"    --- ROUGE-1 F1 (Avg Best) ---");
            logger.info(f"    Overall: {summary['avg_best_rouge1_score']}");
            logger.info(f"    Filtered: {summary['avg_best_rouge1_score_filtered']}")
            logger.info(f"    --- BLEU Score (Avg Best) ---");
            logger.info(f"    Overall: {summary['avg_best_bleu_score']}");
            logger.info(f"    Filtered: {summary['avg_best_bleu_score_filtered']}")
            if meteor_available: logger.info(f"    --- METEOR Score (Avg Best) ---"); logger.info(
                f"    Overall: {summary['avg_best_meteor_score']}"); logger.info(
                f"    Filtered: {summary['avg_best_meteor_score_filtered']}")
        logger.info(f"    Per-Model Results File: {summary['output_file']}")
        logger.info("-" * 30)

    logger.info(f"\n{'=' * 20} Final Evaluation Summary (All Models){' [' + run_mode + ']'} {'=' * 20}")
    for summary_item in overall_summary:
        log_str = f"Model: {summary_item['model_name']:<30} | Task: {summary_item['task']:<25} | Success: {summary_item['overall_success_rate']:>7s} | Avg SBERT: {summary_item['avg_best_sbert_score']}"
        if FLAG_33 and MULTI_EVAL:
            log_str += f" | Avg R-L: {summary_item['avg_best_rougeL_score']} | Avg R-1: {summary_item['avg_best_rouge1_score']}"
            log_str += f" | Avg BLEU: {summary_item['avg_best_bleu_score']}"
            if meteor_available: log_str += f" | Avg METEOR: {summary_item['avg_best_meteor_score']}"
        logger.info(log_str)

    summary_filename = os.path.join(OUTPUT_DIR, f"evaluation_summary_{TASK_NAME}{run_identifier}.csv")
    logger.info(f"\nSaving overall summary to {summary_filename}...")
    try:
        with open(summary_filename, 'w', newline='', encoding='utf-8-sig') as f:
            if overall_summary:
                writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
                writer.writeheader()
                final_summary_rows = []
                for idx, s_item in enumerate(overall_summary):
                    s_copy = s_item.copy()
                    if idx > 0: s_copy['confusing_samples_excluded'] = ''  # Show only once for clarity
                    row_to_write = {field: s_copy.get(field, "N/A") for field in summary_fieldnames}
                    final_summary_rows.append(row_to_write)
                writer.writerows(final_summary_rows)
            else:
                f.write("No summary data generated.\n")
        logger.info(f"Overall summary saved successfully.")
    except IOError as e:
        logger.error(f"Error writing summary file {summary_filename}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving the summary CSV: {e}")

    logger.info(f"\nInterleaved evaluation ({run_mode}) complete.")
    logger.info(
        f"Combined input/output/best_match JSONL saved to: {combined_output_filename if combined_output_file and not combined_output_file.closed else ('SAVE FAILED' if not combined_output_file else combined_output_filename)}")
    logger.info(f"Log file: {log_filename}")


if __name__ == "__main__":
    nltk_punkt_available = False
    nltk_wordnet_omw_available = False  # Combined flag for wordnet and omw-1.4

    if FLAG_33 and MULTI_EVAL_ENABLED:  # Only check if potentially needed
        try:
            nltk.data.find('tokenizers/punkt');
            nltk_punkt_available = True
            logger.info("NLTK 'punkt' tokenizer found.")
        except LookupError:
            print("NLTK 'punkt' tokenizer not found. Attempting download...");
            logger.info("NLTK 'punkt' tokenizer not found. Attempting download...")
            try:
                nltk.download('punkt', quiet=True); print("'punkt' downloaded successfully."); logger.info(
                    "'punkt' downloaded successfully."); nltk_punkt_available = True
            except Exception as nltk_e:
                logger.error(f"Failed NLTK 'punkt' download: {nltk_e}. Some metrics will be disabled.")
        except Exception as e:
            logger.error(f"Error checking NLTK 'punkt' data: {e}. Some metrics will be disabled.")

        try:  # Wordnet and OMW-1.4
            nltk.data.find('corpora/wordnet');
            nltk.data.find('corpora/omw-1.4');
            nltk_wordnet_omw_available = True
            logger.info("NLTK 'wordnet' and 'omw-1.4' resources found (for METEOR).")
        except LookupError:
            print("NLTK 'wordnet' or 'omw-1.4' not found. Attempting download...");
            logger.info("NLTK 'wordnet' or 'omw-1.4' not found. Attempting download...")
            try:
                nltk.download('wordnet', quiet=True);
                nltk.download('omw-1.4', quiet=True);
                print("'wordnet' and 'omw-1.4' downloaded successfully.");
                logger.info("'wordnet' and 'omw-1.4' downloaded successfully.")
                nltk_wordnet_omw_available = True
            except Exception as nltk_e:
                logger.warning(
                    f"Failed NLTK 'wordnet'/'omw-1.4' download: {nltk_e}. METEOR might be disabled or less effective.")
        except Exception as e:
            logger.warning(f"Error checking NLTK 'wordnet'/'omw-1.4' data: {e}. METEOR might be disabled.")

        if not nltk_punkt_available:
            MULTI_EVAL = False  # Affects global MULTI_EVAL used in main
            meteor_available = False  # Affects global meteor_available used in main
            logger.warning("MULTI_EVAL (ROUGE/BLEU/METEOR) is disabled as 'punkt' tokenizer is unavailable.")
        elif not nltk_wordnet_omw_available:
            meteor_available = False  # Disable only METEOR if punkt is OK but wordnet/omw failed
            logger.warning(
                "METEOR scoring is disabled/may be limited as 'wordnet'/'omw-1.4' resources are unavailable.")
        else:  # Both punkt and wordnet/omw are available
            meteor_available = True  # Ensure it's true if all checks passed
    else:  # MULTI_EVAL_ENABLED is False or not FLAG_33
        MULTI_EVAL = False
        meteor_available = False
        if FLAG_33: logger.info("MULTI_EVAL was initially disabled or pre-requisites not met.")

    main()