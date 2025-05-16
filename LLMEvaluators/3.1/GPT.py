import json
import logging
import os
from openai import AzureOpenAI, APIError, APITimeoutError, APIConnectionError, RateLimitError
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from datetime import datetime
from collections import defaultdict
import time
import argparse

# --- Model Definitions (As provided by user) ---
DEFAULT_API_VERSION = ""
# REQUIRED_NEW_API_VERSION = "2024-12-01-preview" # Not directly used unless a model config specifies it

MODEL_CONFIGS = {
    "gpt-4.1": {"api_type": "azure", "api_key": "",
                           "azure_endpoint": "https://eastus2instancefranck.openai.azure.com/",
                           "api_version": DEFAULT_API_VERSION, "deployment_name": "gpt-4.1"},
    "gpt-4o": {"api_type": "azure", "api_key": "",
                          "azure_endpoint": "https://vietgpt.openai.azure.com/", "api_version": DEFAULT_API_VERSION,
                          "deployment_name": "gpt-4o"},
    "gpt-4o-mini": {"api_type": "azure", "api_key": "",
                             "azure_endpoint": "https://vietgpt.openai.azure.com/", "api_version": DEFAULT_API_VERSION,
                             "deployment_name": "gpt-4o-mini"},
}

TASK_NAME = "-3.1" # Using "3.1task" as part of the task name

# Global logger instance
logger = logging.getLogger(__name__)

def setup_logger(model_name_key):
    """Configures the logger to output to console and a file."""
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f"{model_name_key}-{TASK_NAME}-{now}.log"

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    fh = logging.FileHandler(log_filename, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return log_filename

def get_model_prediction(model_name_key, prompt_text):
    """
    Gets a prediction from the specified Azure OpenAI model.
    """
    if model_name_key not in MODEL_CONFIGS:
        logger.error(f"Model configuration for '{model_name_key}' not found.")
        return None

    config = MODEL_CONFIGS[model_name_key]

    try:
        client = AzureOpenAI(
            api_key=config["api_key"],
            api_version=config["api_version"],
            azure_endpoint=config["azure_endpoint"]
        )

        # The prompt already ends with "Predicted score sentiment:",
        # so we expect the model to complete this.
        # We might need to adjust max_tokens if the prompt is very long
        # or if we only expect "positive" or "negative".
        # A small number of tokens should be sufficient for "positive" or "negative".
        response = client.chat.completions.create(
            model=config["deployment_name"], # Corresponds to deployment_name
            messages=[
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=10, # Should be enough for "positive" or "negative"
            temperature=0.0 # For deterministic output
        )
        
        prediction = response.choices[0].message.content.strip().lower()
        
        # Validate prediction
        if prediction in ["positive", "negative"]:
            return prediction
        else:
            logger.warning(f"Model '{model_name_key}' returned an unexpected prediction: '{prediction}'. Expected 'positive' or 'negative'.")
            return None # Or treat as incorrect by returning a value that won't match

    except RateLimitError:
        logger.error(f"Rate limit exceeded for model {model_name_key}. Waiting and retrying...")
        time.sleep(60) # Wait for 60 seconds before retrying
        return get_model_prediction(model_name_key, prompt_text) # Retry
    except APITimeoutError:
        logger.error(f"API request timed out for model {model_name_key}.")
    except APIConnectionError:
        logger.error(f"API connection error for model {model_name_key}.")
    except APIError as e:
        logger.error(f"Azure OpenAI API error for model {model_name_key}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while calling model {model_name_key}: {e}")
    return None

def calculate_metrics(y_true, y_pred, label_map={"positive": 1, "negative": 0}):
    """Calculates accuracy, F1 score, and MCC."""
    if not y_true or not y_pred:
        return 0.0, 0.0, 0.0

    y_true_mapped = [label_map.get(label) for label in y_true if label in label_map]
    y_pred_mapped = [label_map.get(label) for label in y_pred if label in label_map]

    # Ensure we only use successfully mapped labels for metrics
    # This handles cases where a prediction might have been None or invalid
    # and wasn't added to y_pred, or was added as None.
    # We should only have valid 'positive'/'negative' strings in y_true/y_pred lists
    # if the calling logic filters correctly.

    if len(y_true_mapped) != len(y_pred_mapped) or not y_true_mapped:
         # This might happen if predictions are consistently invalid and filtered out
        logger.warning("Mismatch in mapped true/pred list lengths or empty lists for metric calculation.")
        return 0.0, 0.0, 0.0
    
    # Handle cases for F1 and MCC where there might be only one class present
    # For binary classification, pos_label=1 is standard if positive is mapped to 1.
    # 'macro' average for F1 is suitable for multi-class, but for binary, can specify pos_label
    # or let it infer. For clarity, since it's binary, specifying labels can be good.
    # However, if one class is not present in predictions, some metrics might be ill-defined.
    
    try:
        acc = accuracy_score(y_true_mapped, y_pred_mapped)
    except Exception as e:
        logger.error(f"Error calculating accuracy: {e}")
        acc = 0.0

    try:
        # For binary, f1_score with default average='binary' assumes pos_label=1
        # If only one class is present in both y_true and y_pred, F1 might behave unexpectedly
        # depending on the class. Using labels=[0,1] and average='macro' is safer if imbalance or missing classes.
        # Let's stick to 'binary' and handle potential warnings/errors.
        # We can map our labels to 0 and 1. "positive":1, "negative":0.
        f1 = f1_score(y_true_mapped, y_pred_mapped, pos_label=label_map["positive"], zero_division=0.0)
        # If you want macro F1 (e.g. if you consider it two classes always)
        # f1 = f1_score(y_true_mapped, y_pred_mapped, average='macro', zero_division=0.0)
    except Exception as e:
        logger.error(f"Error calculating F1 score: {e}")
        f1 = 0.0
        
    try:
        mcc = matthews_corrcoef(y_true_mapped, y_pred_mapped)
    except Exception as e:
        # MCC can also warn if ill-defined (e.g. one class only in true/pred)
        logger.warning(f"Could not calculate MCC (may be ill-defined): {e}")
        mcc = 0.0
        
    return acc, f1, mcc

def evaluate_model(model_name_key, jsonl_file_path):
    """
    Main function to evaluate a model against a dataset in a .jsonl file.
    """
    log_filepath = setup_logger(model_name_key)
    logger.info(f"Starting evaluation for model: {model_name_key}")
    logger.info(f"Input file: {jsonl_file_path}")
    logger.info(f"Log file will be saved to: {log_filepath}")

    skipped_posts_count = 0
    processed_posts_count = 0

    y_true_overall = []
    y_pred_overall = []
    
    # For per-domain metrics
    # defaultdict stores {'domain_name': {'true': [...], 'pred': [...]}}
    domain_metrics_data = defaultdict(lambda: {"true": [], "pred": []})
    
    label_map = {"positive": 1, "negative": 0}

    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                post_index = i + 1
                try:
                    data = json.loads(line)
                    prediction_task = data.get("prediction_task", {})
                    prompt_text = prediction_task.get("prompt_text")
                    true_label_str = prediction_task.get("true_label", "").strip().lower()
                    domain = prediction_task.get("subreddit", "unknown_domain").lower()
                    target_reply_id = prediction_task.get("target_reply_id", "N/A")

                    if not prompt_text or not true_label_str or true_label_str not in ["positive", "negative"]:
                        logger.warning(f"Post {post_index}: Skipping due to missing prompt, true_label, or invalid true_label ('{true_label_str}'). Reply ID: {target_reply_id}")
                        skipped_posts_count += 1
                        continue

                    predicted_label_str = get_model_prediction(model_name_key, prompt_text)

                    if predicted_label_str is None: # Model call failed or returned invalid
                        logger.warning(f"Post {post_index} (Reply ID: {target_reply_id}): Skipping due to model prediction failure or invalid output.")
                        skipped_posts_count += 1
                        # Optionally, count this as incorrect instead of skipping fully from metrics
                        # For now, strictly skipping if no valid "positive"/"negative" came back.
                        continue 
                    
                    processed_posts_count += 1

                    is_correct = (predicted_label_str == true_label_str)
                    result_str = "Correct" if is_correct else "Incorrect"

                    y_true_overall.append(true_label_str)
                    y_pred_overall.append(predicted_label_str)
                    domain_metrics_data[domain]["true"].append(true_label_str)
                    domain_metrics_data[domain]["pred"].append(predicted_label_str)

                    # Log individual post result
                    logger.info(f"Post {processed_posts_count} (Source Index: {post_index}, Reply ID: {target_reply_id}) - Domain: {domain} - Pred: {predicted_label_str}, True: {true_label_str} - Result: {result_str}")

                    # Calculate and log current overall metrics
                    if y_true_overall and y_pred_overall: # Ensure there's data
                        acc_overall, f1_overall, mcc_overall = calculate_metrics(y_true_overall, y_pred_overall, label_map)
                        logger.info(f"Overall Metrics (after {processed_posts_count} processed posts): Accuracy: {acc_overall:.4f}, F1: {f1_overall:.4f}, MCC: {mcc_overall:.4f}")

                    # Periodic per-domain metrics output
                    if processed_posts_count > 0 and processed_posts_count % 1000 == 0:
                        logger.info(f"--- Per-Domain Metrics at post {processed_posts_count} ---")
                        for d, data_lists in domain_metrics_data.items():
                            if data_lists["true"] and data_lists["pred"]:
                                acc_dom, f1_dom, mcc_dom = calculate_metrics(data_lists["true"], data_lists["pred"], label_map)
                                logger.info(f"Domain: {d} - Accuracy: {acc_dom:.4f}, F1: {f1_dom:.4f}, MCC: {mcc_dom:.4f} (Samples: {len(data_lists['true'])})")
                        logger.info(f"--- End of Per-Domain Metrics at post {processed_posts_count} ---")
                        
                except json.JSONDecodeError:
                    logger.error(f"Post {post_index}: Skipping due to JSON decoding error in line: {line.strip()}")
                    skipped_posts_count += 1
                except Exception as e:
                    logger.error(f"Post {post_index}: An unexpected error occurred during processing: {e}")
                    skipped_posts_count += 1
                    import traceback
                    logger.error(traceback.format_exc())


    except FileNotFoundError:
        logger.error(f"Input file not found: {jsonl_file_path}")
        return
    except Exception as e:
        logger.error(f"An error occurred while reading the file or during the main loop: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    logger.info("--- Evaluation Finished ---")
    logger.info(f"Total posts processed: {processed_posts_count}")
    logger.info(f"Total posts skipped: {skipped_posts_count}")

    if y_true_overall and y_pred_overall:
        acc_overall, f1_overall, mcc_overall = calculate_metrics(y_true_overall, y_pred_overall, label_map)
        logger.info(f"Final Overall Metrics: Accuracy: {acc_overall:.4f}, F1: {f1_overall:.4f}, MCC: {mcc_overall:.4f}")
    else:
        logger.info("No data processed to calculate final overall metrics.")

    logger.info("--- Final Per-Domain Metrics ---")
    if not domain_metrics_data:
        logger.info("No domain-specific data collected.")
    for domain, data_lists in domain_metrics_data.items():
        if data_lists["true"] and data_lists["pred"]:
            acc_dom, f1_dom, mcc_dom = calculate_metrics(data_lists["true"], data_lists["pred"], label_map)
            logger.info(f"Domain: {domain} - Accuracy: {acc_dom:.4f}, F1: {f1_dom:.4f}, MCC: {mcc_dom:.4f} (Samples: {len(data_lists['true'])})")
        else:
            logger.info(f"Domain: {domain} - No valid prediction pairs to calculate metrics.")
    logger.info("--- End of Final Per-Domain Metrics ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM for score sentiment prediction.")
    parser.add_argument("model_key", type=str, choices=MODEL_CONFIGS.keys(),
                        help="The key of the model to evaluate (e.g., 'gpt-4o-mini').")
    parser.add_argument("jsonl_filepath", type=str,
                        help="Path to the .jsonl file containing evaluation prompts.")
    
    args = parser.parse_args()

    # Basic check for API keys if they are placeholders
    if MODEL_CONFIGS[args.model_key].get("api_key", "").startswith("YOUR_AZURE_OPENAI_KEY"):
        print(f"ERROR: API key for model {args.model_key} seems to be a placeholder. Please update MODEL_CONFIGS.")
        exit(1)
    if MODEL_CONFIGS[args.model_key].get("azure_endpoint", "").startswith("YOUR_AZURE_OPENAI_ENDPOINT"):
        print(f"ERROR: Azure endpoint for model {args.model_key} seems to be a placeholder. Please update MODEL_CONFIGS.")
        exit(1)

    evaluate_model(args.model_key, args.jsonl_filepath)

    print(f"\nEvaluation complete. Check the log file in the script's directory for detailed results.")