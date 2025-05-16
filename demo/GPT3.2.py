import json
import os
import time
import datetime
import logging
from collections import defaultdict
import math # 用于向上取整
import numpy as np # For RMSE calculation
import re
# import warnings # No longer using vertexai, so UserWarning for google.auth might not be relevant

# --- Model API Libraries ---
from openai import AzureOpenAI
from sklearn.metrics import mean_squared_error, mean_absolute_error # For MSE, MAE

# --- Configuration Constants ---
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 10

# Use the DEFAULT_API_VERSION that is common or specified for these models.
# If each model in MODEL_CONFIGS has its own specific api_version,
# ensure it's correctly set there.
DEFAULT_API_VERSION = ""
MODEL_CONFIGS = {
    "gpt": {"api_type": "azure", "api_key": "",
                           "azure_endpoint": "https://eastus2instancefranck.openai.azure.com/",
                           "api_version": DEFAULT_API_VERSION, "deployment_name": "gpt"},
}

# General Task Name for Logging
TASK_NAME_PREFIX = "score_prediction_single" # Updated task name prefix
OUTPUT_FOLDER = "SingleGPT_ScorePrediction"

# Dataset file paths (relative to script location or absolute)
DATASET_PATHS = {
    "With": "WithConversationPrompts_ExactScorePrediction.jsonl",
    "Without": "WithoutConversationPrompts_ExactScorePrediction.jsonl"
}

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
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(ch)

    return logger

def is_rate_limit_error(exception):
    # For openai
    if "openai" in str(type(exception)).lower() and hasattr(exception, 'status_code') and exception.status_code == 429:
        return True
    # General check for HTTP errors from underlying libraries like 'requests'
    if hasattr(exception, 'response') and hasattr(exception.response, 'status_code') and exception.response.status_code == 429:
        return True
    return False

# --- Regression Metrics Calculation Function ---
def calculate_regression_metrics(y_true, y_pred):
    """Calculates MSE, RMSE, MAE."""
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {"mse": float('nan'), "rmse": float('nan'), "mae": float('nan'), "samples": 0}

    try:
        y_true_numeric = [float(i) for i in y_true]
        y_pred_numeric = [float(i) for i in y_pred]
    except (ValueError, TypeError) as e:
        return {"mse": float('nan'), "rmse": float('nan'), "mae": float('nan'), "samples": 0, "error": str(e)}

    mse = mean_squared_error(y_true_numeric, y_pred_numeric)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_numeric, y_pred_numeric)
    return {"mse": mse, "rmse": rmse, "mae": mae, "samples": len(y_true_numeric)}

# --- Model API Call Function (Generic Azure GPT) ---
def get_azure_gpt_prediction(prompt_text, model_config, logger_instance, post_id_for_logging="N/A"):
    """Gets a numerical score prediction from a configured Azure GPT model with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            client = AzureOpenAI(
                api_key=model_config["api_key"],
                api_version=model_config["api_version"],
                azure_endpoint=model_config["azure_endpoint"]
            )
            response = client.chat.completions.create(
                model=model_config["deployment_name"],
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=25, # Assuming score is a short response
                temperature=0.0
            )
            prediction_str = response.choices[0].message.content.strip()
            try:
                prediction_score = float(prediction_str)
                return prediction_score, None # prediction, error_type
            except ValueError:
                # Attempt to extract a number if the response contains other text
                matches = re.findall(r"[-+]?\b\d+(?:\.\d+)?\b", prediction_str)
                if matches:
                    try:
                        extracted_score_str = matches[-1] # Take the last found number
                        prediction_score = float(extracted_score_str)
                        logger_instance.warning(f"Post {post_id_for_logging}: GPT returned '{prediction_str}', extracted score: {prediction_score}")
                        return prediction_score, None
                    except ValueError:
                        logger_instance.warning(f"Post {post_id_for_logging}: GPT returned non-numerical after regex: '{prediction_str}' (extracted: '{matches}').")
                        return None, "non_numerical_response_after_regex"
                else:
                    logger_instance.warning(f"Post {post_id_for_logging}: GPT returned non-numerical: '{prediction_str}'")
                    return None, "non_numerical_response"
        except Exception as e:
            if is_rate_limit_error(e) and attempt < MAX_RETRIES - 1:
                logger_instance.warning(f"Post {post_id_for_logging}: GPT API 429 error (Attempt {attempt + 1}/{MAX_RETRIES}) for model {model_config['deployment_name']}. Retrying in {RETRY_DELAY_SECONDS}s... Error: {e}")
                time.sleep(RETRY_DELAY_SECONDS)
            elif is_rate_limit_error(e) and attempt == MAX_RETRIES - 1:
                logger_instance.error(f"Post {post_id_for_logging}: GPT API 429 error for model {model_config['deployment_name']}. Max retries ({MAX_RETRIES}) reached. Skipping post. Error: {e}")
                return None, "rate_limit_max_retries"
            else:
                logger_instance.error(f"Post {post_id_for_logging}: Error calling GPT API for model {model_config['deployment_name']} (Attempt {attempt + 1}): {e}")
                return None, "other_api_error"
    return None, "unknown_failure_after_retries"


# --- Main Evaluation Function (Single-threaded) ---
def evaluate_model(model_to_evaluate_name, dataset_type_to_use, output_folder=OUTPUT_FOLDER):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    task_name = f"{TASK_NAME_PREFIX}_{model_to_evaluate_name}_{dataset_type_to_use}"
    log_filename = f"{task_name}-{timestamp}.log"
    logger_instance = setup_logger(log_filename, "eval_logger", output_folder)

    logger_instance.info(f"Starting evaluation for model: {model_to_evaluate_name} using dataset: {dataset_type_to_use}")

    if model_to_evaluate_name not in MODEL_CONFIGS:
        logger_instance.error(f"Model '{model_to_evaluate_name}' not found in MODEL_CONFIGS. Aborting.")
        return

    model_config_to_use = MODEL_CONFIGS[model_to_evaluate_name]
    logger_instance.info(f"Using model configuration: Deployment Name='{model_config_to_use['deployment_name']}', Endpoint='{model_config_to_use['azure_endpoint']}'")


    input_file_path = DATASET_PATHS.get(dataset_type_to_use)
    if not input_file_path:
        logger_instance.error(f"Dataset type '{dataset_type_to_use}' not found in DATASET_PATHS. Aborting.")
        return
    if not os.path.exists(input_file_path):
        logger_instance.error(f"Input file not found: {input_file_path}. Aborting.")
        # You might want to create a dummy file here if it's for testing and it doesn't exist,
        # similar to the original script's main block.
        # For now, we'll just abort.
        return

    logger_instance.info(f"Input file: {input_file_path}")
    logger_instance.info(f"Output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    results_filename = os.path.join(output_folder, f"results_{task_name}-{timestamp}.jsonl")

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        logger_instance.info(f"Successfully read {len(all_lines)} lines from {input_file_path}.")
    except Exception as e:
        logger_instance.error(f"Error reading input file {input_file_path}: {e}")
        return

    if not all_lines:
        logger_instance.warning("Input file is empty. Nothing to process.")
        return

    processed_posts_count = 0
    skipped_posts_general_count = 0
    skipped_due_to_429_retries_count = 0
    skipped_post_ids_due_to_429 = []

    all_results_data = []
    all_true_scores = []
    all_predicted_scores = []
    domain_stats = defaultdict(lambda: {"true_scores": [], "predicted_scores": []})

    # Debug: Counter for the first 20 outputs
    debug_output_count = 0
    MAX_DEBUG_OUTPUTS = 20

    for i, post_data_str in enumerate(all_lines):
        post_identifier = f"Item_{i+1}" # Unique identifier for this run

        try:
            post_data = json.loads(post_data_str)
            task_info = post_data.get("prediction_task", {})
            prompt = task_info.get("prompt_text")
            true_label_raw = task_info.get("true_label")
            domain = task_info.get("subreddit", "unknown_domain")

            if not prompt or true_label_raw is None:
                logger_instance.warning(f"Post {post_identifier}: Missing prompt/true_label. Skipping.")
                skipped_posts_general_count += 1
                all_results_data.append({
                    "original_data": post_data, "true_label": true_label_raw, "predicted_label": None,
                    "domain": domain, "status": "skipped_missing_data", "post_id": post_identifier
                })
                continue

            try:
                true_score = float(true_label_raw)
            except (ValueError, TypeError):
                logger_instance.warning(f"Post {post_identifier}: Invalid true_label '{true_label_raw}'. Skipping.")
                skipped_posts_general_count += 1
                all_results_data.append({
                    "original_data": post_data, "true_label": true_label_raw, "predicted_label": None,
                    "domain": domain, "status": "skipped_invalid_true_label", "post_id": post_identifier
                })
                continue

            predicted_score, error_type = get_azure_gpt_prediction(prompt, model_config_to_use, logger_instance, post_identifier)

            if error_type == "rate_limit_max_retries":
                logger_instance.error(f"Post {post_identifier}: Skipped due to max retries on 429 error.")
                skipped_due_to_429_retries_count += 1
                skipped_post_ids_due_to_429.append(post_identifier)
                all_results_data.append({
                    "original_data": post_data, "true_label": true_score, "predicted_label": None,
                    "domain": domain, "status": "skipped_429_max_retries", "post_id": post_identifier
                })
                continue

            if predicted_score is None:
                logger_instance.warning(f"Post {post_identifier}: Failed to get prediction (error: {error_type}). Skipping.")
                skipped_posts_general_count += 1
                all_results_data.append({
                    "original_data": post_data, "true_label": true_score, "predicted_label": None,
                    "domain": domain, "status": f"skipped_prediction_failure_{error_type}", "post_id": post_identifier
                })
                continue

            all_true_scores.append(true_score)
            all_predicted_scores.append(predicted_score)
            domain_stats[domain]["true_scores"].append(true_score)
            domain_stats[domain]["predicted_scores"].append(predicted_score)

            result_item = {
                "prompt_substring": prompt[:100] + "..." if prompt else "N/A",
                "true_label": true_score,
                "predicted_label": predicted_score,
                "domain": domain,
                "status": "processed",
                "post_id": post_identifier
            }
            all_results_data.append(result_item)
            processed_posts_count += 1

            # Debug output for the first 20 processed items
            if debug_output_count < MAX_DEBUG_OUTPUTS:
                print(f"DEBUG ({debug_output_count+1}/{MAX_DEBUG_OUTPUTS}): Post {post_identifier} - True: {true_score}, Predicted: {predicted_score:.4f}, Domain: {domain}")
                debug_output_count += 1


            if processed_posts_count % 50 == 0:
                if all_true_scores:
                    current_metrics = calculate_regression_metrics(all_true_scores, all_predicted_scores)
                    logger_instance.info(
                        f"After {processed_posts_count} processed items - Cumulative Stats (Samples: {current_metrics['samples']}): "
                        f"MSE: {current_metrics['mse']:.4f}, RMSE: {current_metrics['rmse']:.4f}, MAE: {current_metrics['mae']:.4f}"
                    )

        except json.JSONDecodeError:
            logger_instance.error(f"Item {post_identifier}: Failed to decode JSON. Skipping. Line: '{post_data_str[:200]}...'")
            skipped_posts_general_count += 1
            all_results_data.append({
                "original_data_string": post_data_str, "true_label": None, "predicted_label": None,
                "domain": "unknown_json_error", "status": "skipped_json_error", "post_id": post_identifier
            })
        except Exception as e:
            logger_instance.error(f"Post {post_identifier} processing error: {e}. Skipping.", exc_info=True)
            skipped_posts_general_count += 1
            domain_for_error = "unknown_error_domain"
            try: # Try to get domain even from malformed data for context
                temp_post_data = json.loads(post_data_str)
                domain_for_error = temp_post_data.get("prediction_task", {}).get("subreddit", "unknown_error_domain_in_except")
            except: pass
            all_results_data.append({
                "original_data_string": post_data_str, "true_label": None, "predicted_label": None,
                "domain": domain_for_error, "status": f"skipped_runtime_error: {str(e)}", "post_id": post_identifier
            })

    with open(results_filename, 'w', encoding='utf-8') as f_out:
        for res_item in all_results_data:
            f_out.write(json.dumps(res_item) + "\n")
    logger_instance.info(f"All item-level results saved to: {results_filename}")

    logger_instance.info(f"Finished processing. Processed: {processed_posts_count}, Skipped (General): {skipped_posts_general_count}, Skipped (429 Retries): {skipped_due_to_429_retries_count}")
    if skipped_post_ids_due_to_429:
        logger_instance.info(f"Post IDs skipped due to 429 retries: {skipped_post_ids_due_to_429}")

    # --- FINAL AGGREGATION AND STATISTICS ---
    summary_output_path = os.path.join(output_folder, f"FINAL_SUMMARY_{task_name}-{timestamp}.txt")
    summary_lines = []

    def log_and_store(message):
        logger_instance.info(message)
        summary_lines.append(message)

    log_and_store(f"\n--- FINAL EVALUATION RESULTS for {model_to_evaluate_name} on {dataset_type_to_use} ---")
    if all_true_scores:
        final_overall_metrics = calculate_regression_metrics(all_true_scores, all_predicted_scores)
        log_and_store(f"Overall Final: MSE: {final_overall_metrics['mse']:.4f}, RMSE: {final_overall_metrics['rmse']:.4f}, MAE: {final_overall_metrics['mae']:.4f}")
        log_and_store(f"Total Samples for Overall Metrics: {final_overall_metrics['samples']}")
    else:
        log_and_store("No posts were successfully processed with valid scores to calculate overall final metrics.")

    log_and_store("\n--- Final Domain-Specific Regression Metrics ---")
    if domain_stats:
        for d, data in sorted(domain_stats.items()):
            if data["true_scores"]:
                domain_metrics = calculate_regression_metrics(data["true_scores"], data["predicted_scores"])
                log_and_store(f"Domain '{d}': MSE: {domain_metrics['mse']:.4f}, RMSE: {domain_metrics['rmse']:.4f}, MAE: {domain_metrics['mae']:.4f} (Samples: {domain_metrics['samples']})")
            else:
                log_and_store(f"Domain '{d}': No data with valid scores processed for metrics.")
    else:
        log_and_store("No domain-specific data with valid scores collected.")

    log_and_store(f"\nTotal posts processed successfully: {processed_posts_count}")
    log_and_store(f"Total posts skipped (general errors, missing data, etc.): {skipped_posts_general_count}")
    log_and_store(f"Total posts skipped (due to 429 max retries): {skipped_due_to_429_retries_count}")
    log_and_store(f"Evaluation finished for model: {model_to_evaluate_name}, dataset: {dataset_type_to_use}")

    log_file_path = "N/A"
    for handler in logger_instance.handlers:
        if isinstance(handler, logging.FileHandler):
            log_file_path = handler.baseFilename
            break
    log_and_store(f"Log file: {log_file_path}")
    log_and_store(f"Item-level results JSONL: {results_filename}")
    log_and_store(f"This summary is saved to: {summary_output_path}")

    with open(summary_output_path, 'w', encoding='utf-8') as f_summary:
        for line in summary_lines:
            f_summary.write(line + "\n")

    print(f"\n>>> Evaluation complete. Summary saved to {summary_output_path} <<<")
    print(f">>> Log file: {log_file_path} <<<")

    for handler in list(logger_instance.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
        logger_instance.removeHandler(handler)


# --- Main script execution block ---
if __name__ == "__main__":
    # ----- USER CONFIGURATION -----
    # Choose one of the model names defined in MODEL_CONFIGS
    MODEL_CHOICE = "gpt-4.1"
    # MODEL_CHOICE = "gpt-4o-mini"

    # Choose "With" or "Without" for the dataset type
    DATASET_TYPE_CHOICE = "With"
    # DATASET_TYPE_CHOICE = "Without"
    # -----------------------------

    # Ensure OUTPUT_FOLDER exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    temp_init_logger = setup_logger("init_check.log", "init_logger_temp", OUTPUT_FOLDER) # For pre-run checks

    # Validate model choice
    if MODEL_CHOICE not in MODEL_CONFIGS:
        temp_init_logger.error(f"Error: MODEL_CHOICE '{MODEL_CHOICE}' is not defined in MODEL_CONFIGS. Available models: {list(MODEL_CONFIGS.keys())}")
        print(f"Error: MODEL_CHOICE '{MODEL_CHOICE}' is not defined in MODEL_CONFIGS. Aborting.")
        exit(1)
    else:
        temp_init_logger.info(f"Selected model for evaluation: {MODEL_CHOICE}")


    # Validate dataset type and file existence
    selected_input_file = DATASET_PATHS.get(DATASET_TYPE_CHOICE)
    if not selected_input_file:
        temp_init_logger.error(f"Error: DATASET_TYPE_CHOICE '{DATASET_TYPE_CHOICE}' is not a valid key in DATASET_PATHS. Available types: {list(DATASET_PATHS.keys())}")
        print(f"Error: DATASET_TYPE_CHOICE '{DATASET_TYPE_CHOICE}' is not valid. Aborting.")
        exit(1)

    if not os.path.exists(selected_input_file):
        temp_init_logger.error(f"Error: Input file '{selected_input_file}' for dataset type '{DATASET_TYPE_CHOICE}' not found. Please check path and permissions.")
        # You can add logic here to create a dummy file for testing if needed:
        # print(f"Attempting to create a dummy input file '{selected_input_file}' for testing...")
        # try:
        #     with open(selected_input_file, 'w') as f_dummy:
        #         f_dummy.write('{"prediction_task": {"prompt_text": "Dummy prompt for score 1.5", "true_label": "1.5", "subreddit": "dummy_with"}}\n')
        #         f_dummy.write('{"prediction_task": {"prompt_text": "Another dummy prompt for score -0.5", "true_label": "-0.5", "subreddit": "dummy_with"}}\n')
        #     temp_init_logger.info(f"Successfully created dummy file: {selected_input_file}")
        # except Exception as e_dummy:
        #     temp_init_logger.error(f"Failed to create dummy file {selected_input_file}: {e_dummy}")
        #     print(f"Error: Failed to create dummy input file '{selected_input_file}'. Aborting.")
        #     exit(1)
        # For production, it's better to abort if the file isn't found.
        print(f"Error: Input file '{selected_input_file}' not found. Aborting.")
        exit(1)
    else:
        temp_init_logger.info(f"Using dataset file: {selected_input_file} for type '{DATASET_TYPE_CHOICE}'")


    # Clean up temp logger before main evaluation
    for handler in list(temp_init_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
        temp_init_logger.removeHandler(handler)


    # Proceed with evaluation
    evaluate_model(
        MODEL_CHOICE,
        DATASET_TYPE_CHOICE,
        output_folder=OUTPUT_FOLDER
    )