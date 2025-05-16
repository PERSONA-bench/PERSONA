import json
import os
import time
import datetime
import logging
from collections import defaultdict
import threading # 引入多线程模块
import math # 用于向上取整

import vertexai
import warnings
from google.protobuf.json_format import MessageToDict, MessageToJson

warnings.filterwarnings(
    action='ignore',
    message=r'Your application has authenticated using end user credentials.*without a quota project',
    category=UserWarning,
    module=r'google\.auth\._default'
)
from vertexai.generative_models import GenerativeModel, Part
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef

# --- Model API Libraries ---
from openai import AzureOpenAI
import google.generativeai as genai
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
TASK_NAME = "3.1task_multithread"
OUTPUT_FOLDER = "MultiThread" 
NUM_THREADS = 1 

# --- Logger Setup ---
def setup_logger(log_filename, logger_name, output_folder=OUTPUT_FOLDER):
    """Sets up a logger that writes to a file and console."""
    os.makedirs(output_folder, exist_ok=True)
    full_log_path = os.path.join(output_folder, log_filename)

    logger = logging.getLogger(logger_name) 
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    # File Handler
    fh = logging.FileHandler(full_log_path)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    return logger


def get_gpt4o_prediction(prompt_text, logger_instance):
    """Gets a prediction from Azure GPT-4o."""
    try:
        client = AzureOpenAI(
            api_key=AZURE_GPT4O_API_KEY,
            api_version=AZURE_GPT4O_API_VERSION,
            azure_endpoint=AZURE_GPT4O_ENDPOINT
        )
        response = client.chat.completions.create(
            model=AZURE_GPT4O_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=10,
            temperature=0.0
        )
        prediction = response.choices[0].message.content.strip().lower()
        if prediction in ["positive", "negative"]:
            return prediction
        else:
            logger_instance.warning(f"GPT-4o returned an unexpected format: {prediction}")
            return None
    except Exception as e:
        logger_instance.error(f"Error calling GPT-4o API: {e}")
        return None

def get_gemini_prediction(prompt_text, logger_instance):
    try:

        # vertexai.init(project=GEMINI_PROJECT_ID, location=GEMINI_LOCATION)
        model_instance = GenerativeModel(model_name=GEMINI_MODEL_NAME)
        generation_config = {
            "max_output_tokens": 10, # "positive" or "negative"
            "temperature": 0.0,
        }

        logger_instance.info(f"Calling Gemini with prompt (first 50 chars): {prompt_text[:50]}...")
        response = model_instance.generate_content(
            prompt_text,
            generation_config=generation_config
        )

        if response.candidates and response.candidates[0].content.parts:
            prediction_text = response.candidates[0].content.parts[0].text.strip().lower()
            if prediction_text in ["positive", "negative"]:
                logger_instance.info(f"Gemini prediction: {prediction_text}")
                return prediction_text
            else:
                logger_instance.warning(f"Gemini returned an unexpected format: {prediction_text}. Full response: {response}")
                return None
        else:
            logger_instance.warning(f"Gemini no valid candidates or parts. Full response: {response}")
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 logger_instance.warning(f"Gemini prompt blocked. Reason: {response.prompt_feedback.block_reason}")
            return None
    except Exception as e:
        logger_instance.error(f"Error calling Gemini API: {e}", exc_info=True)
        return None

def get_claude_prediction(prompt_text, logger_instance):
    """Gets a prediction from Claude via Vertex AI."""
    try:

        client = AnthropicVertex(region=CLAUDE_LOCATION, project_id=CLAUDE_PROJECT_ID)
        message = client.messages.create(
            max_tokens=10, # "positive" or "negative"
            messages=[{"role": "user", "content": prompt_text}],
            model=CLAUDE_MODEL_NAME,
            temperature=0.0
        )
        if message.content and message.content[0].text:
            prediction = message.content[0].text.strip().lower()
            if prediction in ["positive", "negative"]:
                return prediction
            else:
                logger_instance.warning(f"Claude returned an unexpected format: {prediction}")
                return None
        else:
            logger_instance.warning(f"Claude returned no content or text in the first part: {message}")
            return None
    except Exception as e:
        logger_instance.error(f"Error calling Claude API: {e}")
        return None

# --- Metrics Calculation Function---
def calculate_metrics_sklearn(y_true, y_pred, pos_label="positive"):
    """Calculates accuracy, F1, precision, recall, MCC using sklearn."""
    if not y_true or not y_pred:
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "mcc": 0.0, "samples": 0}

    try:
        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall, "mcc": mcc, "samples": len(y_true)}
    except ValueError as e:
        # print(f"Metric calculation warning (sklearn): {e}")
        acc = accuracy_score(y_true, y_pred)
        return {"accuracy": acc, "f1": 0.0, "precision": 0.0, "recall": 0.0, "mcc": 0.0, "samples": len(y_true)}


# --- Thread Worker Function ---
def process_data_chunk(data_chunk, model_name_to_eval, thread_id, output_folder=OUTPUT_FOLDER):
    """
    Processes a chunk of data in a separate thread.
    Logs to its own file and saves results to its own file.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f"{model_name_to_eval}-{TASK_NAME}-thread{thread_id}-{timestamp}.log"
    logger_instance = setup_logger(log_filename, f"logger_thread_{thread_id}", output_folder)
    
    results_filename = os.path.join(output_folder, f"results_thread{thread_id}.jsonl")

    logger_instance.info(f"Thread {thread_id}: Starting processing for model: {model_name_to_eval}")
    logger_instance.info(f"Thread {thread_id}: Processing {len(data_chunk)} items.")
    logger_instance.info(f"Thread {thread_id}: Results will be saved to: {results_filename}")

    model_function = None
    if model_name_to_eval == "GPT4o":
        model_function = get_gpt4o_prediction
    elif model_name_to_eval == "Gemini":

        model_function = get_gemini_prediction
    elif model_name_to_eval == "Claude":

        model_function = get_claude_prediction
    else:
        logger_instance.error(f"Thread {thread_id}: Unsupported model: {model_name_to_eval}")
        return

    processed_posts = 0
    skipped_posts = 0
    thread_results = []

    for i, post_data_str in enumerate(data_chunk):
        try:
            post_data = json.loads(post_data_str)
            task_info = post_data.get("prediction_task", {})
            prompt = task_info.get("prompt_text")
            true_label = task_info.get("true_label", "").lower()
            domain = task_info.get("subreddit", "unknown_domain")

            if not prompt or not true_label or true_label not in ["positive", "negative"]:
                logger_instance.warning(f"Thread {thread_id}, Post {i+1}: Invalid data or missing fields. Skipping.")
                skipped_posts += 1
                continue

            logger_instance.info(f"Thread {thread_id}, Post {processed_posts + 1} (Domain: {domain}) ---")
            
            predicted_label = model_function(prompt, logger_instance)

            if predicted_label is None:
                logger_instance.warning(f"Thread {thread_id}, Post {processed_posts + 1}: Failed to get prediction. Skipping.")
                skipped_posts += 1
                thread_results.append({
                    "original_data": post_data,
                    "true_label": true_label,
                    "predicted_label": None,
                    "domain": domain,
                    "status": "skipped_prediction_failure"
                })
                continue
            
            is_correct = (predicted_label == true_label)
            logger_instance.info(f"Thread {thread_id}: True: {true_label}, Predicted: {predicted_label} -> {'CORRECT' if is_correct else 'INCORRECT'}")

            thread_results.append({
                "original_data": post_data,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "domain": domain,
                "status": "processed"
            })
            processed_posts += 1

        except json.JSONDecodeError:
            logger_instance.error(f"Thread {thread_id}, Item {i+1}: Failed to decode JSON. Skipping.")
            skipped_posts += 1
            thread_results.append({
                "original_data_string": post_data_str,
                "true_label": None,
                "predicted_label": None,
                "domain": "unknown_json_error",
                "status": "skipped_json_error"
            })
        except Exception as e:
            logger_instance.error(f"Thread {thread_id}, Post {processed_posts + 1}: Unexpected error: {e}. Skipping.", exc_info=True)
            skipped_posts += 1

            try:
                loaded_post_data = json.loads(post_data_str)
                domain_for_error = loaded_post_data.get("prediction_task", {}).get("subreddit", "unknown_error_domain")
            except:
                loaded_post_data = post_data_str
                domain_for_error = "unknown_parsing_error_domain"

            thread_results.append({
                "original_data": loaded_post_data,
                "true_label": None,
                "predicted_label": None,
                "domain": domain_for_error,
                "status": f"skipped_runtime_error: {str(e)}"
            })

    with open(results_filename, 'w', encoding='utf-8') as f_out:
        for res in thread_results:
            f_out.write(json.dumps(res) + "\n")

    logger_instance.info(f"Thread {thread_id}: Finished processing. Processed: {processed_posts}, Skipped: {skipped_posts}")
    logger_instance.info(f"Thread {thread_id}: Results saved to {results_filename}")
    for handler in logger_instance.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
        logger_instance.removeHandler(handler)


# --- Main Evaluation Function (Modified for Multithreading) ---
def evaluate_model_multithreaded(model_name_to_eval, input_file_path, num_threads=NUM_THREADS, output_folder=OUTPUT_FOLDER):
    """
    Evaluates the specified model using multiple threads.
    """
    main_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    main_log_filename = f"MAIN-{model_name_to_eval}-{TASK_NAME}-{main_timestamp}.log"
    main_logger = setup_logger(main_log_filename, "main_logger", output_folder)

    main_logger.info(f"Starting multithreaded evaluation for model: {model_name_to_eval}")
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
    for i, chunk in enumerate(data_chunks):
        if not chunk:
            main_logger.info(f"Skipping creation of thread {i+1} as its data chunk is empty.")
            continue
        thread = threading.Thread(
            target=process_data_chunk,
            args=(chunk, model_name_to_eval, i + 1, output_folder),
            name=f"Thread-{i+1}"
        )
        threads.append(thread)
        main_logger.info(f"Starting thread {i+1} with {len(chunk)} items.")
        thread.start()

    for thread in threads:
        thread.join()
    main_logger.info("All threads have completed.")

    main_logger.info("--- STARTING FINAL AGGREGATION AND STATISTICS ---")
    all_true_labels_combined = []
    all_predictions_combined = []
    domain_stats_combined = defaultdict(lambda: {"true_labels": [], "predictions": []})
    total_processed_combined = 0
    total_skipped_combined = 0

    for i in range(len(threads)):
        thread_id = i + 1
        results_file = os.path.join(output_folder, f"results_thread{thread_id}.jsonl")
        try:
            with open(results_file, 'r', encoding='utf-8') as f_res:
                for line_num, line in enumerate(f_res):
                    try:
                        res_item = json.loads(line)
                        if res_item.get("status") == "processed":
                            true_label = res_item.get("true_label")
                            predicted_label = res_item.get("predicted_label")
                            domain = res_item.get("domain", "unknown_domain_in_result")

                            if true_label and predicted_label:
                                all_true_labels_combined.append(true_label)
                                all_predictions_combined.append(predicted_label)
                                domain_stats_combined[domain]["true_labels"].append(true_label)
                                domain_stats_combined[domain]["predictions"].append(predicted_label)
                                total_processed_combined +=1
                            else:
                                total_skipped_combined += 1
                                main_logger.warning(f"Result item from {results_file} line {line_num+1} marked processed but lacks labels: {res_item}")
                        else:
                            total_skipped_combined += 1

                    except json.JSONDecodeError:
                        main_logger.error(f"Failed to decode JSON from result file {results_file}, line {line_num+1}. Skipping line.")
                        total_skipped_combined += 1
        except FileNotFoundError:
            main_logger.warning(f"Result file {results_file} not found. Skipping aggregation for this thread.")
        except Exception as e:
            main_logger.error(f"Error reading result file {results_file}: {e}. Skipping aggregation for this thread.")

    summary_output_path = os.path.join(output_folder, f"FINAL_SUMMARY_{model_name_to_eval}-{main_timestamp}.txt")
    summary_lines = []

    def log_and_store(message):
        main_logger.info(message)
        summary_lines.append(message)

    log_and_store("\n--- FINAL COMBINED EVALUATION RESULTS ---")
    if all_true_labels_combined:
        final_overall_metrics = calculate_metrics_sklearn(all_true_labels_combined, all_predictions_combined)
        log_and_store(f"Overall Final: Accuracy: {final_overall_metrics['accuracy']:.4f}, F1: {final_overall_metrics['f1']:.4f}, Precision: {final_overall_metrics['precision']:.4f}, Recall: {final_overall_metrics['recall']:.4f}, MCC: {final_overall_metrics['mcc']:.4f}")
        log_and_store(f"Total Samples for Overall Metrics: {final_overall_metrics['samples']}")
    else:
        log_and_store("No posts were successfully processed across all threads to calculate overall final metrics.")

    log_and_store("\n--- Final Combined Domain-Specific Metrics ---")
    if domain_stats_combined:
        for d, data in sorted(domain_stats_combined.items()): # 
            if data["true_labels"]:
                domain_metrics = calculate_metrics_sklearn(data["true_labels"], data["predictions"])
                log_and_store(f"Domain '{d}': Accuracy: {domain_metrics['accuracy']:.4f}, F1: {domain_metrics['f1']:.4f}, Precision: {domain_metrics['precision']:.4f}, Recall: {domain_metrics['recall']:.4f}, MCC: {domain_metrics['mcc']:.4f} (Samples: {domain_metrics['samples']})")
            else:
                log_and_store(f"Domain '{d}': No data processed for metrics.")
    else:
        log_and_store("No domain-specific data collected from any thread.")

    log_and_store(f"\nTotal posts processed successfully (combined): {total_processed_combined}")
    log_and_store(f"Total posts skipped (combined, due to errors or missing data): {total_skipped_combined}")
    log_and_store(f"Evaluation finished for model: {model_name_to_eval}")
    log_and_store(f"Main log file: {main_logger.handlers[0].baseFilename}")
    log_and_store(f"Individual thread logs and results are in: {output_folder}")
    log_and_store(f"This summary is saved to: {summary_output_path}")

    with open(summary_output_path, 'w', encoding='utf-8') as f_summary:
        for line in summary_lines:
            f_summary.write(line + "\n")

    print(f"\n>>> Multithreaded evaluation complete. Summary saved to {summary_output_path} <<<")
    print(f">>> Main log: {os.path.join(output_folder, main_log_filename)} <<<")
    for handler in main_logger.handlers: #
        if isinstance(handler, logging.FileHandler):
            handler.close()
        main_logger.removeHandler(handler)


# --- Main script execution block ---
if __name__ == "__main__":
    # --- Configuration for the run ---
    MODEL_TO_RUN = "Claude"  # "GPT4o", "Gemini"
    INPUT_JSONL_FILE = "WithoutConversationPrompts_ScorePrediction_Refactored.jsonl"

    try:
        if MODEL_TO_RUN == "Gemini":
            main_logger_for_init = setup_logger("init_check.log", "init_logger_temp", OUTPUT_FOLDER)
            main_logger_for_init.info(f"Attempting to initialize Vertex AI for Gemini (Project: {GEMINI_PROJECT_ID}, Location: {GEMINI_LOCATION})")
            vertexai.init(project=GEMINI_PROJECT_ID, location=GEMINI_LOCATION)
            main_logger_for_init.info("Vertex AI initialized for Gemini.")
            for handler in main_logger_for_init.handlers: handler.close(); main_logger_for_init.removeHandler(handler)
        elif MODEL_TO_RUN == "Claude":
             main_logger_for_init = setup_logger("init_check.log", "init_logger_temp", OUTPUT_FOLDER)
             main_logger_for_init.info(f"Attempting to initialize Vertex AI for Claude (Project: {CLAUDE_PROJECT_ID}, Location: {CLAUDE_LOCATION})")
             vertexai.init(project=CLAUDE_PROJECT_ID, location=CLAUDE_LOCATION)
             main_logger_for_init.info("Vertex AI initialized for Claude (or confirmed).")
             for handler in main_logger_for_init.handlers: handler.close(); main_logger_for_init.removeHandler(handler)

    except Exception as e:

        print(f"CRITICAL: Failed to initialize Vertex AI. Error: {e}. Check project ID and location, and authentication.")


    if not os.path.exists(INPUT_JSONL_FILE):
        print(f"Error: Input file '{INPUT_JSONL_FILE}' still not found after attempting to create dummy. Please check path and permissions.")
    else:
        evaluate_model_multithreaded(MODEL_TO_RUN, INPUT_JSONL_FILE, num_threads=NUM_THREADS, output_folder=OUTPUT_FOLDER)