import json
import os
import time
import datetime
import logging
from collections import defaultdict
import threading # 引入多线程模块
import math # 用于向上取整
import numpy as np # For RMSE calculation
import re
import vertexai
import warnings
from google.protobuf.json_format import MessageToDict, MessageToJson
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 10
warnings.filterwarnings(
    action='ignore',
    message=r'Your application has authenticated using end user credentials.*without a quota project',
    category=UserWarning,
    module=r'google\.auth\._default'
)
from vertexai.generative_models import GenerativeModel, Part
from sklearn.metrics import mean_squared_error, mean_absolute_error # For MSE, MAE

# --- Model API Libraries ---
from openai import AzureOpenAI
# import google.generativeai as genai # Not directly used in this version, but kept if needed elsewhere
from anthropic import AnthropicVertex

# --- Configuration Constants ---
# !! IMPORTANT: Replace placeholders with your actual credentials and configurations !!

# GPT-4o Configuration
AZURE_GPT4O_API_KEY = "YOUR_AZURE_OPENAI_API_KEY"
AZURE_GPT4O_ENDPOINT = "YOUR_AZURE_OPENAI_ENDPOINT"
AZURE_GPT4O_API_VERSION = ""
AZURE_GPT4O_DEPLOYMENT_NAME = "YOUR_GPT4O_DEPLOYMENT_NAME"

# Gemini Configuration
GEMINI_PROJECT_ID = 'YOUR_VERTEX_PROJECT_ID'
GEMINI_LOCATION = ''
GEMINI_MODEL_NAME = "" 

# Claude Configuration
CLAUDE_PROJECT_ID = ''
CLAUDE_LOCATION = ''
CLAUDE_MODEL_NAME = ""

# General Task Name for Logging
TASK_NAME = "3.2task_score_prediction_multithread" # Updated task name
OUTPUT_FOLDER = "MultiThread_ScorePrediction" # Potentially new output folder
NUM_THREADS = 6 

# --- Logger Setup ---
def setup_logger(log_filename, logger_name, output_folder=OUTPUT_FOLDER):
    """Sets up a logger that writes to a file and console."""
    os.makedirs(output_folder, exist_ok=True)
    full_log_path = os.path.join(output_folder, log_filename)

    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(full_log_path)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    # Console handler can be added if per-thread console output is desired beyond main logger
    # ch = logging.StreamHandler()
    # ch.setFormatter(logging.Formatter('%(threadName)s: %(message)s'))
    # logger.addHandler(ch)

    return logger
def is_rate_limit_error(exception):
    # For openai
    if "openai" in str(type(exception)).lower() and hasattr(exception, 'status_code') and exception.status_code == 429:
        return True
    # For anthropic
    if "anthropic" in str(type(exception)).lower():
        if hasattr(exception, 'status_code') and exception.status_code == 429:
            return True
        if "RateLimitError" in str(type(exception)): # More general check for Anthropic
            return True
    # For google-cloud / vertexai (Gemini)
    if "google.api_core.exceptions" in str(type(exception)).lower():
        if "ResourceExhausted" in str(type(exception)): # Typically 429 for quota
            return True
        if hasattr(exception, 'code') and exception.code == 429: # Less common but possible
             return True
        if hasattr(exception, 'status_code') and exception.status_code == 429: # For http-based errors
            return True
    # General check for HTTP errors from underlying libraries like 'requests'
    if hasattr(exception, 'response') and hasattr(exception.response, 'status_code') and exception.response.status_code == 429:
        return True
    return False
# --- Regression Metrics Calculation Function ---
def calculate_regression_metrics(y_true, y_pred):
    """Calculates MSE, RMSE, MAE."""
    if not y_true or not y_pred or len(y_true) != len(y_pred) or not y_true: # Ensure lists are not empty and have same length
        return {"mse": float('nan'), "rmse": float('nan'), "mae": float('nan'), "samples": 0}

    try:
        # Ensure all elements are numbers; handle potential conversion errors if not already floats
        y_true_numeric = [float(i) for i in y_true]
        y_pred_numeric = [float(i) for i in y_pred]
    except (ValueError, TypeError) as e:
        # This case should ideally be prevented by upstream parsing
        # logging.error(f"Error converting labels/predictions to float for metrics: {e}") # Consider where to log this
        return {"mse": float('nan'), "rmse": float('nan'), "mae": float('nan'), "samples": 0, "error": str(e)}


    mse = mean_squared_error(y_true_numeric, y_pred_numeric)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_numeric, y_pred_numeric)
    return {"mse": mse, "rmse": rmse, "mae": mae, "samples": len(y_true_numeric)}

# --- Model API Call Functions (Modified for Score Prediction) ---
def get_gpt4o_prediction(prompt_text, logger_instance, post_id_for_logging="N/A"): # 添加 post_id_for_logging
    """Gets a numerical score prediction from Azure GPT-4o with retry logic for 429 errors."""
    for attempt in range(MAX_RETRIES):
        try:
            client = AzureOpenAI(
                api_key=AZURE_GPT4O_API_KEY,
                api_version=AZURE_GPT4O_API_VERSION,
                azure_endpoint=AZURE_GPT4O_ENDPOINT
            )
            response = client.chat.completions.create(
                model=AZURE_GPT4O_DEPLOYMENT_NAME,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=25,
                temperature=0.0
            )
            prediction_str = response.choices[0].message.content.strip()
            try:
                prediction_score = float(prediction_str)
                return prediction_score, None # prediction, error_type
            except ValueError:
                logger_instance.warning(f"Post {post_id_for_logging}: GPT-4o returned non-numerical: '{prediction_str}'")
                return None, "non_numerical_response"
        except Exception as e:
            if is_rate_limit_error(e) and attempt < MAX_RETRIES - 1:
                logger_instance.warning(f"Post {post_id_for_logging}: GPT-4o API 429 error (Attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {RETRY_DELAY_SECONDS}s... Error: {e}")
                time.sleep(RETRY_DELAY_SECONDS)
            elif is_rate_limit_error(e) and attempt == MAX_RETRIES - 1:
                logger_instance.error(f"Post {post_id_for_logging}: GPT-4o API 429 error. Max retries ({MAX_RETRIES}) reached. Skipping post. Error: {e}")
                return None, "rate_limit_max_retries"
            else:
                logger_instance.error(f"Post {post_id_for_logging}: Error calling GPT-4o API (Attempt {attempt + 1}): {e}")
                return None, "other_api_error" # Or a more specific error if needed
    return None, "unknown_failure_after_retries" # Should be caught by specific returns above

def get_gemini_prediction(prompt_text, logger_instance, post_id_for_logging="N/A"): # 添加 post_id_for_logging
    """Gets a numerical score prediction from Gemini with retry logic for 429 errors."""
    for attempt in range(MAX_RETRIES):
        try:
            model_instance = GenerativeModel(model_name=GEMINI_MODEL_NAME)
            generation_config = {
                "max_output_tokens": 15,
                "temperature": 0.0,
            }
            # logger_instance.info(f"Calling Gemini with prompt (first 50 chars): {prompt_text[:50]}...")
            response = model_instance.generate_content(
                prompt_text,
                generation_config=generation_config
            )

            if response.candidates and response.candidates[0].content.parts:
                prediction_str = response.candidates[0].content.parts[0].text.strip()
                try:
                    prediction_score = float(prediction_str)
                    # logger_instance.info(f"Gemini prediction: {prediction_score}")
                    return prediction_score, None
                except ValueError:
                    logger_instance.warning(f"Post {post_id_for_logging}: Gemini returned non-numerical: '{prediction_str}'. Full response: {response}")
                    return None, "non_numerical_response"
            else:
                logger_instance.warning(f"Post {post_id_for_logging}: Gemini no valid candidates. Full response: {response}")
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                     logger_instance.warning(f"Post {post_id_for_logging}: Gemini prompt blocked. Reason: {response.prompt_feedback.block_reason}")
                     return None, "blocked_prompt" # Specific error for blocked prompt
                return None, "no_valid_candidates"
        except Exception as e:
            if is_rate_limit_error(e) and attempt < MAX_RETRIES - 1:
                logger_instance.warning(f"Post {post_id_for_logging}: Gemini API 429/ResourceExhausted error (Attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {RETRY_DELAY_SECONDS}s... Error: {e}")
                time.sleep(RETRY_DELAY_SECONDS)
            elif is_rate_limit_error(e) and attempt == MAX_RETRIES - 1:
                logger_instance.error(f"Post {post_id_for_logging}: Gemini API 429/ResourceExhausted error. Max retries ({MAX_RETRIES}) reached. Skipping post. Error: {e}")
                return None, "rate_limit_max_retries"
            else:
                logger_instance.error(f"Post {post_id_for_logging}: Error calling Gemini API (Attempt {attempt + 1}): {e}", exc_info=False) # exc_info=False to reduce log noise unless debugging
                return None, "other_api_error"
    return None, "unknown_failure_after_retries"


def get_claude_prediction(prompt_text, logger_instance, post_id_for_logging="N/A"): # 添加 post_id_for_logging
    """Gets a numerical score prediction from Claude via Vertex AI with retry for 429 and extraction."""
    current_max_tokens = 7
    for attempt in range(MAX_RETRIES):
        try:
            client = AnthropicVertex(region=CLAUDE_LOCATION, project_id=CLAUDE_PROJECT_ID)
            message = client.messages.create(
                max_tokens=current_max_tokens,
                messages=[{"role": "user", "content": prompt_text}],
                model=CLAUDE_MODEL_NAME,
                temperature=0.0
            )
            if message.content and message.content[0].text:
                prediction_str = message.content[0].text.strip()
                # logger_instance.info(f"Claude raw response: '{prediction_str}'")
                try:
                    prediction_score = float(prediction_str)
                    return prediction_score, None
                except ValueError:
                    matches = re.findall(r"[-+]?\b\d+(?:\.\d+)?\b", prediction_str)
                    if matches:
                        try:
                            extracted_score_str = matches[-1]
                            prediction_score = float(extracted_score_str)
                            return prediction_score, None
                        except ValueError:
                            logger_instance.warning(f"Post {post_id_for_logging}: Claude regex found '{matches}', but last '{extracted_score_str}' not float. Original: '{prediction_str}'")
                            return None, "non_numerical_regex_match"
                    else:
                        logger_instance.warning(f"Post {post_id_for_logging}: Claude non-numerical and no regex match: '{prediction_str}'")
                        return None, "non_numerical_response"
            else:
                logger_instance.warning(f"Post {post_id_for_logging}: Claude no content: {message}")
                return None, "no_content_in_response"
        except Exception as e:
            if is_rate_limit_error(e) and attempt < MAX_RETRIES - 1:
                logger_instance.warning(f"Post {post_id_for_logging}: Claude API 429 error (Attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {RETRY_DELAY_SECONDS}s... Error: {e}")
                time.sleep(RETRY_DELAY_SECONDS)
            elif is_rate_limit_error(e) and attempt == MAX_RETRIES - 1:
                logger_instance.error(f"Post {post_id_for_logging}: Claude API 429 error. Max retries ({MAX_RETRIES}) reached. Skipping post. Error: {e}")
                return None, "rate_limit_max_retries"
            else:
                logger_instance.error(f"Post {post_id_for_logging}: Error calling Claude API (Attempt {attempt + 1}): {e}", exc_info=False)
                return None, "other_api_error"
    return None, "unknown_failure_after_retries"

# --- Thread Worker Function ---
# --- Thread Worker Function ---
def process_data_chunk(data_chunk, model_name_to_eval, thread_id, output_folder=OUTPUT_FOLDER):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f"{model_name_to_eval}-{TASK_NAME}-thread{thread_id}-{timestamp}.log"
    logger_instance = setup_logger(log_filename, f"logger_thread_{thread_id}", output_folder)
    
    results_filename = os.path.join(output_folder, f"results_thread{thread_id}.jsonl")

    logger_instance.info(f"Thread {thread_id}: Starting processing for model: {model_name_to_eval}")
    logger_instance.info(f"Thread {thread_id}: Processing {len(data_chunk)} items.")

    model_function_map = {
        "GPT4o": get_gpt4o_prediction,
        "Gemini": get_gemini_prediction,
        "Claude": get_claude_prediction,
    }
    model_function = model_function_map.get(model_name_to_eval)

    if not model_function:
        logger_instance.error(f"Thread {thread_id}: Unsupported model: {model_name_to_eval}")
        # Return structure for aggregation
        return {
            "processed_count": 0, "skipped_count": len(data_chunk),
            "true_scores": [], "predicted_scores": [],
            "skipped_due_to_429_count": 0, "skipped_post_ids_due_to_429": []
        }

    processed_posts_count = 0 # Counts successfully processed posts with valid scores
    skipped_posts_general_count = 0 # Counts skips due to reasons other than 429 max retries
    
    # New counters and lists for 429 skips
    skipped_due_to_429_retries_count = 0
    skipped_post_ids_due_to_429 = [] # Store post identifiers (e.g., original index or a specific ID from data)

    thread_results_data = [] 
    current_thread_true_scores = []
    current_thread_predicted_scores = []

    for i, post_data_str in enumerate(data_chunk):
        post_identifier = f"ChunkItem_{i+1}" # Default identifier, can be improved if data has unique IDs
        original_post_index = -1 # Placeholder for a more global index if available

        try:
            post_data = json.loads(post_data_str)
            # Try to get a more meaningful identifier if available, e.g., a post_id field
            # For example: post_identifier = post_data.get("post_id", f"ChunkItem_{i+1}")
            # If you have a global index from before chunking, you'd pass that or calculate it

            task_info = post_data.get("prediction_task", {})
            prompt = task_info.get("prompt_text")
            true_label_raw = task_info.get("true_label")
            domain = task_info.get("subreddit", "unknown_domain")
            # Attempt to get a unique ID if present in your data, otherwise use loop index 'i'
            # For example, if your JSON lines have a "post_id" or "unique_id" field:
            # post_unique_id_for_tracking = post_data.get("id_field", f"thread_{thread_id}_item_{i}")
            # For now, we'll use a composite ID for skipped_post_ids_due_to_429
            post_tracking_id = f"Thread{thread_id}_Line{i}" # Adjust if you have a better ID

            if not prompt or true_label_raw is None:
                logger_instance.warning(f"Thread {thread_id}, Post {post_tracking_id}: Missing prompt/true_label. Skipping.")
                skipped_posts_general_count += 1
                thread_results_data.append({
                    "original_data": post_data, "true_label": true_label_raw, "predicted_label": None,
                    "domain": domain, "status": "skipped_missing_data", "post_id": post_tracking_id
                })
                continue
            
            try:
                true_score = float(true_label_raw)
            except (ValueError, TypeError):
                logger_instance.warning(f"Thread {thread_id}, Post {post_tracking_id}: Invalid true_label '{true_label_raw}'. Skipping.")
                skipped_posts_general_count += 1
                thread_results_data.append({
                    "original_data": post_data, "true_label": true_label_raw, "predicted_label": None,
                    "domain": domain, "status": "skipped_invalid_true_label", "post_id": post_tracking_id
                })
                continue

            # logger_instance.info(f"Thread {thread_id}, Processing Post {post_tracking_id} (Domain: {domain}) ---")
            
            # Pass post_tracking_id for better logging in API functions
            predicted_score, error_type = model_function(prompt, logger_instance, post_tracking_id)

            if error_type == "rate_limit_max_retries":
                logger_instance.error(f"Thread {thread_id}, Post {post_tracking_id}: Skipped due to max retries on 429 error.")
                skipped_due_to_429_retries_count += 1
                skipped_post_ids_due_to_429.append(post_tracking_id)
                thread_results_data.append({
                    "original_data": post_data, "true_label": true_score, "predicted_label": None,
                    "domain": domain, "status": "skipped_429_max_retries", "post_id": post_tracking_id
                })
                continue # Move to the next post
            
            if predicted_score is None: # Handle other types of prediction failures
                logger_instance.warning(f"Thread {thread_id}, Post {post_tracking_id}: Failed to get prediction (error: {error_type}). Skipping.")
                skipped_posts_general_count += 1
                thread_results_data.append({
                    "original_data": post_data, "true_label": true_score, "predicted_label": None,
                    "domain": domain, "status": f"skipped_prediction_failure_{error_type}", "post_id": post_tracking_id
                })
                continue
            
            # logger_instance.info(f"Thread {thread_id}: True Score: {true_score}, Predicted Score: {predicted_score:.4f}")

            current_thread_true_scores.append(true_score)
            current_thread_predicted_scores.append(predicted_score)

            thread_results_data.append({
                "prompt_substring": prompt[:100] + "..." if prompt else "N/A",
                "true_label": true_score,
                "predicted_label": predicted_score,
                "domain": domain,
                "status": "processed",
                "post_id": post_tracking_id
            })
            processed_posts_count += 1

            if processed_posts_count % 50 == 0: # Log cumulative less frequently
                if current_thread_true_scores:
                    cumulative_thread_metrics = calculate_regression_metrics(current_thread_true_scores, current_thread_predicted_scores)
                    logger_instance.info(
                        f"Thread {thread_id}, After {processed_posts_count} processed items in this thread - Cumulative Stats (Samples: {cumulative_thread_metrics['samples']}): "
                        f"MSE: {cumulative_thread_metrics['mse']:.4f}, RMSE: {cumulative_thread_metrics['rmse']:.4f}, MAE: {cumulative_thread_metrics['mae']:.4f}"
                    )

        except json.JSONDecodeError:
            logger_instance.error(f"Thread {thread_id}, Item {post_tracking_id}: Failed to decode JSON. Skipping. Line: '{post_data_str[:200]}...'")
            skipped_posts_general_count += 1
            thread_results_data.append({
                "original_data_string": post_data_str, "true_label": None, "predicted_label": None,
                "domain": "unknown_json_error", "status": "skipped_json_error", "post_id": post_tracking_id
            })
        except Exception as e:
            logger_instance.error(f"Thread {thread_id}, Post {post_tracking_id} processing error: {e}. Skipping.", exc_info=True)
            skipped_posts_general_count += 1
            domain_for_error = "unknown_error_domain"
            loaded_post_data_for_error = post_data_str
            try:
                temp_post_data = json.loads(post_data_str)
                domain_for_error = temp_post_data.get("prediction_task", {}).get("subreddit", "unknown_error_domain_in_except")
                loaded_post_data_for_error = temp_post_data
            except: pass
            thread_results_data.append({
                "original_data": loaded_post_data_for_error, "true_label": None, "predicted_label": None,
                "domain": domain_for_error, "status": f"skipped_runtime_error: {str(e)}", "post_id": post_tracking_id
            })

    with open(results_filename, 'w', encoding='utf-8') as f_out:
        for res_item in thread_results_data:
            f_out.write(json.dumps(res_item) + "\n")

    logger_instance.info(f"Thread {thread_id}: Finished. Processed: {processed_posts_count}, Skipped (General): {skipped_posts_general_count}, Skipped (429 Retries): {skipped_due_to_429_retries_count}")
    if skipped_post_ids_due_to_429:
        logger_instance.info(f"Thread {thread_id}: Post IDs skipped due to 429 retries: {skipped_post_ids_due_to_429}")
    
    # Final metrics for this thread
    # ... (logging for final thread metrics remains the same) ...
    
    for handler in list(logger_instance.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
        logger_instance.removeHandler(handler)

    return {
        "processed_count": processed_posts_count,
        "skipped_general_count": skipped_posts_general_count,
        "true_scores": current_thread_true_scores,
        "predicted_scores": current_thread_predicted_scores,
        "skipped_due_to_429_count": skipped_due_to_429_retries_count,
        "skipped_post_ids_due_to_429": skipped_post_ids_due_to_429,
        "results_file_path": results_filename # For easier aggregation if needed, though we primarily use returned lists
    }

# --- Main Evaluation Function (Modified for Multithreading and Regression) ---
def evaluate_model_multithreaded(model_name_to_eval, input_file_path, num_threads=NUM_THREADS, output_folder=OUTPUT_FOLDER):
    main_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    main_log_filename = f"MAIN-{model_name_to_eval}-{TASK_NAME}-{main_timestamp}.log"
    main_logger = setup_logger(main_log_filename, "main_logger", output_folder)

    main_logger.info(f"Starting multithreaded regression evaluation for model: {model_name_to_eval}")
    main_logger.info(f"Input file: {input_file_path}")
    main_logger.info(f"Number of threads: {num_threads}")
    main_logger.info(f"Output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        main_logger.info(f"Successfully read {len(all_lines)} lines from {input_file_path}.")
    except FileNotFoundError:
        main_logger.error(f"Input file not found: {input_file_path}")
        return
    except Exception as e:
        main_logger.error(f"Error reading input file {input_file_path}: {e}")
        return

    if not all_lines:
        main_logger.warning("Input file is empty. Nothing to process.")
        return

    total_lines = len(all_lines)
    chunk_size = math.ceil(total_lines / num_threads)
    data_chunks = [all_lines[i:i + chunk_size] for i in range(0, total_lines, chunk_size)]
    main_logger.info(f"Data split into {len(data_chunks)} chunks of (up to) {chunk_size} lines each.")

    threads = []
    active_thread_ids = []
    for i, chunk in enumerate(data_chunks):
        if not chunk:
            main_logger.info(f"Skipping creation of thread {i+1} as its data chunk is empty.")
            continue
        thread_id = i + 1
        active_thread_ids.append(thread_id)
        thread = threading.Thread(
            target=process_data_chunk,
            args=(chunk, model_name_to_eval, thread_id, output_folder),
            name=f"Thread-{thread_id}"
        )
        threads.append(thread)
        main_logger.info(f"Starting thread {thread_id} with {len(chunk)} items.")
        thread.start()

    for thread in threads:
        thread.join()
    main_logger.info("All threads have completed.")

    main_logger.info("--- STARTING FINAL AGGREGATION AND STATISTICS (REGRESSION) ---")
    all_true_scores_combined = []
    all_predicted_scores_combined = []
    domain_stats_combined = defaultdict(lambda: {"true_scores": [], "predicted_scores": []})
    total_processed_combined = 0
    total_skipped_combined = 0

    for thread_id in active_thread_ids: # Iterate based on threads that were actually started
        results_file = os.path.join(output_folder, f"results_thread{thread_id}.jsonl")
        try:
            with open(results_file, 'r', encoding='utf-8') as f_res:
                for line_num, line in enumerate(f_res):
                    try:
                        res_item = json.loads(line)
                        if res_item.get("status") == "processed":
                            true_score = res_item.get("true_label") # Expecting number
                            predicted_score = res_item.get("predicted_label") # Expecting number
                            domain = res_item.get("domain", "unknown_domain_in_result")

                            if true_score is not None and predicted_score is not None:
                                try: # Ensure they are float for aggregation
                                    all_true_scores_combined.append(float(true_score))
                                    all_predicted_scores_combined.append(float(predicted_score))
                                    domain_stats_combined[domain]["true_scores"].append(float(true_score))
                                    domain_stats_combined[domain]["predicted_scores"].append(float(predicted_score))
                                    total_processed_combined +=1
                                except (ValueError, TypeError):
                                    total_skipped_combined += 1
                                    main_logger.warning(f"Result item from {results_file} line {line_num+1} had non-numeric score after 'processed' status: {res_item}")
                            else:
                                total_skipped_combined += 1
                                main_logger.warning(f"Result item from {results_file} line {line_num+1} marked processed but lacks scores: {res_item}")
                        else:
                            total_skipped_combined += 1
                    except json.JSONDecodeError:
                        main_logger.error(f"Failed to decode JSON from result file {results_file}, line {line_num+1}. Skipping line.")
                        total_skipped_combined += 1
        except FileNotFoundError:
            main_logger.warning(f"Result file {results_file} not found. Skipping aggregation for this thread.")
        except Exception as e:
            main_logger.error(f"Error reading result file {results_file}: {e}. Skipping aggregation for this thread.")

    summary_output_path = os.path.join(output_folder, f"FINAL_SUMMARY_REGRESSION_{model_name_to_eval}-{main_timestamp}.txt")
    summary_lines = []

    def log_and_store(message):
        main_logger.info(message)
        summary_lines.append(message)

    log_and_store("\n--- FINAL COMBINED REGRESSION EVALUATION RESULTS ---")
    if all_true_scores_combined:
        final_overall_metrics = calculate_regression_metrics(all_true_scores_combined, all_predicted_scores_combined)
        log_and_store(f"Overall Final: MSE: {final_overall_metrics['mse']:.4f}, RMSE: {final_overall_metrics['rmse']:.4f}, MAE: {final_overall_metrics['mae']:.4f}")
        log_and_store(f"Total Samples for Overall Metrics: {final_overall_metrics['samples']}")
    else:
        log_and_store("No posts were successfully processed with valid scores across all threads to calculate overall final metrics.")

    log_and_store("\n--- Final Combined Domain-Specific Regression Metrics ---")
    if domain_stats_combined:
        for d, data in sorted(domain_stats_combined.items()):
            if data["true_scores"]:
                domain_metrics = calculate_regression_metrics(data["true_scores"], data["predicted_scores"])
                log_and_store(f"Domain '{d}': MSE: {domain_metrics['mse']:.4f}, RMSE: {domain_metrics['rmse']:.4f}, MAE: {domain_metrics['mae']:.4f} (Samples: {domain_metrics['samples']})")
            else:
                log_and_store(f"Domain '{d}': No data with valid scores processed for metrics.")
    else:
        log_and_store("No domain-specific data with valid scores collected from any thread.")

    log_and_store(f"\nTotal posts processed successfully (combined): {total_processed_combined}")
    log_and_store(f"Total posts skipped (combined, due to errors, missing/invalid data, or prediction failures): {total_skipped_combined}")
    log_and_store(f"Regression evaluation finished for model: {model_name_to_eval}")
    
    main_log_file_path = "N/A"
    if main_logger.handlers and isinstance(main_logger.handlers[0], logging.FileHandler):
        main_log_file_path = main_logger.handlers[0].baseFilename
    log_and_store(f"Main log file: {main_log_file_path}")
    log_and_store(f"Individual thread logs and results are in: {output_folder}")
    log_and_store(f"This summary is saved to: {summary_output_path}")

    with open(summary_output_path, 'w', encoding='utf-8') as f_summary:
        for line in summary_lines:
            f_summary.write(line + "\n")

    print(f"\n>>> Multithreaded regression evaluation complete. Summary saved to {summary_output_path} <<<")
    print(f">>> Main log: {os.path.join(output_folder, main_log_filename)} <<<")

    for handler in list(main_logger.handlers): # Iterate over a copy
        if isinstance(handler, logging.FileHandler):
            handler.close()
        main_logger.removeHandler(handler)


# --- Main script execution block ---
if __name__ == "__main__":
    MODEL_TO_RUN = "Claude"  # Or "GPT4o", "Gemini"
    # INPUT_JSONL_FILE = "WithoutConversationPrompts_ExactScorePrediction.jsonl" # Ensure this file has numerical true_label
    INPUT_JSONL_FILE = "WithConversationPrompts_ExactScorePrediction.jsonl" # Ensure this file has numerical true_label
    # Ensure OUTPUT_FOLDER exists early for init_check.log
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    try:
        temp_init_logger = setup_logger("init_check.log", "init_logger_temp", OUTPUT_FOLDER)
        if MODEL_TO_RUN == "Gemini":
            temp_init_logger.info(f"Attempting to initialize Vertex AI for Gemini (Project: {GEMINI_PROJECT_ID}, Location: {GEMINI_LOCATION})")
            vertexai.init(project=GEMINI_PROJECT_ID, location=GEMINI_LOCATION)
            temp_init_logger.info("Vertex AI initialized for Gemini.")
        elif MODEL_TO_RUN == "Claude":
             temp_init_logger.info(f"Attempting to initialize Vertex AI for Claude (Project: {CLAUDE_PROJECT_ID}, Location: {CLAUDE_LOCATION})")
             vertexai.init(project=CLAUDE_PROJECT_ID, location=CLAUDE_LOCATION)
             temp_init_logger.info("Vertex AI initialized for Claude (or confirmed).")
        
        for handler in list(temp_init_logger.handlers): # Clean up temp logger
            if isinstance(handler, logging.FileHandler):
                handler.close()
            temp_init_logger.removeHandler(handler)

    except Exception as e:
        print(f"CRITICAL: Failed to initialize Vertex AI. Error: {e}. Check project ID and location, and authentication.")
        # Consider exiting if Vertex AI is essential for the chosen model and fails to init
        # exit(1)

    if not os.path.exists(INPUT_JSONL_FILE):
        print(f"Error: Input file '{INPUT_JSONL_FILE}' not found. Please check path and permissions.")
        # Create a dummy file for demonstration if it doesn't exist (optional, for testing)
        # print(f"Creating a dummy input file '{INPUT_JSONL_FILE}' for testing purposes.")
        # with open(INPUT_JSONL_FILE, 'w') as f_dummy:
        #     f_dummy.write('{"prediction_task": {"prompt_text": "Predict score for this text.", "true_label": "2.5", "subreddit": "testdomain1"}}\n')
        #     f_dummy.write('{"prediction_task": {"prompt_text": "Another text for score prediction.", "true_label": "-1.0", "subreddit": "testdomain2"}}\n')
        #     f_dummy.write('{"prediction_task": {"prompt_text": "Text with invalid label.", "true_label": "not_a_score", "subreddit": "testdomain1"}}\n')
        #     f_dummy.write('{"prediction_task": {"prompt_text": "Text with missing label.", "subreddit": "testdomain1"}}\n')

    if os.path.exists(INPUT_JSONL_FILE): # Re-check after potential dummy creation
        evaluate_model_multithreaded(
            MODEL_TO_RUN, 
            INPUT_JSONL_FILE, 
            num_threads=NUM_THREADS, 
            output_folder=OUTPUT_FOLDER
        )
    else:
        # This else block will now only be reached if the dummy file creation logic is removed/commented out
        # and the file truly doesn't exist.
        print(f"Error: Input file '{INPUT_JSONL_FILE}' still not found. Aborting.")