import json
import os
import sys
import logging
from datetime import datetime as dt
from collections import defaultdict
import csv
import re
# --- Evaluation Library Imports (similar to 3.3_Eval_Claude_v3.py) ---
MULTI_EVAL_ENABLED = True
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
except ImportError:
    print("ERROR: 'sentence-transformers' or 'torch' not found. Please install them: pip install sentence-transformers torch")
    MULTI_EVAL_ENABLED = False
    # sys.exit(1) # Allow script to run without SBERT if other metrics are fine

try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk

    nltk_packages = {'punkt': False, 'wordnet': False, 'omw-1.4': False}
    for pkg_id in nltk_packages.keys():
        try:
            nltk.data.find(f'tokenizers/{pkg_id}' if pkg_id == 'punkt' else f'corpora/{pkg_id}')
            nltk_packages[pkg_id] = True
        except LookupError:
            print(f"NLTK '{pkg_id}' resource not found. Attempting to download...")
            try:
                nltk.download(pkg_id, quiet=True)
                nltk_packages[pkg_id] = True
                print(f"NLTK '{pkg_id}' downloaded successfully.")
            except Exception as nltk_e:
                print(f"ERROR: Failed to download NLTK '{pkg_id}': {nltk_e}.")
                if pkg_id == 'punkt': # Critical for ROUGE/BLEU
                    MULTI_EVAL_ENABLED = False
                    print("ROUGE/BLEU will be disabled due to missing 'punkt'.")

except ImportError:
    print("ERROR: 'rouge-score' or 'nltk' not found. Install them (`pip install rouge-score nltk`) for ROUGE/BLEU/METEOR evaluation.")
    MULTI_EVAL_ENABLED = False

if not MULTI_EVAL_ENABLED:
    print("INFO: Some or all text evaluation metrics (SBERT, ROUGE, BLEU, METEOR) will be disabled due to missing libraries or NLTK resources.")

try:
    import numpy as np
except ImportError:
    print("ERROR: 'numpy' not found. Please install it: pip install numpy")
    sys.exit(1)
# --- End Imports ---


# --- Configuration ---
INPUT_JSONL_FILE = "DS_3.3_Pse_results.jsonl"  # From the previous script
OUTPUT_METRICS_DIR = "Evaluation_Metrics"
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2' # Consistent with 3.3_Eval_Claude_v3.py

os.makedirs(OUTPUT_METRICS_DIR, exist_ok=True)

# --- Logging Setup ---
log_filename = os.path.join(OUTPUT_METRICS_DIR, f"Metrics_Report-{dt.now():%Y%m%d_%H%M%S}.log")
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("MetricsCalculator")
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)
# --- End Logging ---


# --- Global Resources (initialized in main) ---
sbert_model = None
rouge_scorer_instance = None
meteor_available = False
device = 'cpu'


def initialize_evaluation_resources():
    global sbert_model, rouge_scorer_instance, meteor_available, device

    if 'torch' in sys.modules and 'sentence_transformers' in sys.modules:
        try:
            logger.info(f"Loading SBERT model: {SBERT_MODEL_NAME}...")
            device = 'cuda' if torch.cuda.is_available() else \
                     ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
            sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=device)
            logger.info(f"SBERT model loaded successfully on device: {device}.")
        except Exception as e:
            logger.error(f"Error loading SBERT model: {e}. SBERT evaluation will be disabled.")
            # sbert_model will remain None
    else:
        logger.warning("SBERT libraries (torch, sentence-transformers) not fully available. SBERT evaluation disabled.")


    if MULTI_EVAL_ENABLED and 'rouge_score' in sys.modules and 'nltk' in sys.modules and nltk_packages['punkt']:
        try:
            rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL', 'rouge1'], use_stemmer=True)
            logger.info("ROUGE (L,1) scorer initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize ROUGE scorer: {e}. ROUGE evaluation disabled.")
            # rouge_scorer_instance will remain None

        # Check for METEOR specific NLTK resources
        if nltk_packages['wordnet'] and nltk_packages['omw-1.4']:
            meteor_available = True
            logger.info("METEOR scoring enabled (WordNet and OMW-1.4 found).")
        else:
            meteor_available = False
            logger.warning("METEOR scoring disabled due to missing NLTK 'wordnet' or 'omw-1.4'.")
    elif not nltk_packages['punkt']:
         logger.warning("ROUGE, BLEU, METEOR disabled as NLTK 'punkt' is unavailable.")
    else:
        logger.info("ROUGE/BLEU/METEOR dependent libraries not fully available or MULTI_EVAL_ENABLED is False.")


def calculate_sbert_similarity(text1, text2):
    if sbert_model is None or not text1 or not text2:
        return 0.0  # Or None, depending on how you want to handle missing data/model
    try:
        with torch.no_grad():
            emb1 = sbert_model.encode(text1, convert_to_tensor=True, device=device)
            emb2 = sbert_model.encode(text2, convert_to_tensor=True, device=device)
            cos_sim = util.pytorch_cos_sim(emb1, emb2)
        return cos_sim.item()
    except Exception as e:
        logger.error(f"Error in SBERT calculation for texts ('{str(text1)[:50]}...', '{str(text2)[:50]}...'): {e}")
        return 0.0 # Or handle error appropriately

def calculate_rouge_scores(reference, candidate):
    if rouge_scorer_instance is None or not reference or not candidate:
        return {"rougeL_fmeasure": 0.0, "rouge1_fmeasure": 0.0}
    try:
        scores = rouge_scorer_instance.score(reference, candidate)
        return {
            "rougeL_fmeasure": scores['rougeL'].fmeasure,
            "rouge1_fmeasure": scores['rouge1'].fmeasure
        }
    except Exception as e:
        logger.error(f"Error in ROUGE calculation for candidate '{str(candidate)[:50]}...': {e}")
        return {"rougeL_fmeasure": 0.0, "rouge1_fmeasure": 0.0}


def calculate_bleu_score(reference, candidate):
    if not MULTI_EVAL_ENABLED or not nltk_packages['punkt'] or not reference or not candidate:
        return 0.0
    try:
        ref_tokens = [nltk.word_tokenize(reference.lower())]
        can_tokens = nltk.word_tokenize(candidate.lower())
        if not can_tokens: return 0.0 # Avoid error with empty candidate
        # Using SmoothingFunction().method1 to avoid issues with short sentences or no n-gram overlap
        return sentence_bleu(ref_tokens, can_tokens, smoothing_function=SmoothingFunction().method1)
    except Exception as e:
        logger.error(f"Error in BLEU calculation for candidate '{str(candidate)[:50]}...': {e}")
        return 0.0

def calculate_meteor_score(reference, candidate):
    if not MULTI_EVAL_ENABLED or not meteor_available or not reference or not candidate: # meteor_available checks for wordnet and omw-1.4
        return 0.0
    try:
        ref_tokens = [nltk.word_tokenize(reference.lower())] # METEOR expects a list of reference token lists
        can_tokens = nltk.word_tokenize(candidate.lower())
        if not can_tokens: return 0.0
        return meteor_score(ref_tokens, can_tokens)
    except Exception as e:
        logger.error(f"Error in METEOR calculation for candidate '{str(candidate)[:50]}...': {e}")
        return 0.0

def process_results_file(filepath):
    domain_metrics = defaultdict(lambda: {
        "count": 0,
        "sbert_sum": 0.0, "sbert_count": 0,
        "rougeL_sum": 0.0, "rougeL_count": 0,
        "rouge1_sum": 0.0, "rouge1_count": 0,
        "bleu_sum": 0.0, "bleu_count": 0,
        "meteor_sum": 0.0, "meteor_count": 0,
        "output_errors": 0 # Count entries where output might be missing or an error string
    })
    all_data_metrics = {
        "count": 0,
        "sbert_sum": 0.0, "sbert_count": 0,
        "rougeL_sum": 0.0, "rougeL_count": 0,
        "rouge1_sum": 0.0, "rouge1_count": 0,
        "bleu_sum": 0.0, "bleu_count": 0,
        "meteor_sum": 0.0, "meteor_count": 0,
        "output_errors": 0
    }
    
    individual_results_for_csv = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON line {line_num+1} in {filepath}: {line.strip()}")
                    continue

                true_label = data.get("true_label")
                # The 'output' field from results.jsonl might be a single string or potentially a list of candidates.
                # For now, assuming it's a single string.
                # If it can be a list (e.g., from `repeat=10`), the logic needs to change here
                # to iterate through the list and pick the best.
                # Current assumption: data["output"] is the single model output string.
                llm_output_raw = data.get("output")
                subreddit = data.get("subreddit", "Unknown_Domain")
                original_prompt = data.get("original_prompt", "") # Get original prompt if available

                if not true_label or not llm_output_raw:
                    logger.warning(f"Skipping entry due to missing true_label or output. Line {line_num+1}, Subreddit: {subreddit}")
                    if subreddit not in domain_metrics : domain_metrics[subreddit] # Ensure domain is initialized
                    domain_metrics[subreddit]["output_errors"] += 1
                    all_data_metrics["output_errors"] += 1
                    domain_metrics[subreddit]["count"] += 1 # Count it as an attempt for the domain
                    all_data_metrics["count"] += 1
                    continue

                # If llm_output_raw could be a list of candidates:
                candidate_outputs = []
                if isinstance(llm_output_raw, list):
                    candidate_outputs = [str(c) for c in llm_output_raw if isinstance(c, str) and c.strip()]
                    if not candidate_outputs: # If list is empty or contains no valid strings
                         logger.warning(f"Output field was a list but contained no valid string candidates. Line {line_num+1}")
                         llm_output_best = "" # Treat as empty output
                    # If candidate_outputs is not empty, we'd need to evaluate each and pick the best.
                    # For now, if it's a list, we'll just take the first valid one for simplicity,
                    # as the "pick best of 10" logic isn't fully defined by input structure.
                    # To implement "pick best of 10": iterate candidate_outputs, calc all metrics for each, store the set of metrics for the one that maximizes SBERT (or other primary metric).
                    llm_output_best = candidate_outputs[0] if candidate_outputs else ""

                elif isinstance(llm_output_raw, str):
                    candidate_outputs = [llm_output_raw] # Treat as a single candidate
                    llm_output_best = llm_output_raw
                    llm_output_best = re.sub(r"<think>.*?</think>", "", llm_output_best, flags=re.DOTALL).strip()
                else:
                    logger.warning(f"Unexpected type for 'output' field: {type(llm_output_raw)}. Skipping metrics. Line {line_num+1}")
                    llm_output_best = "" # Treat as empty

                if not llm_output_best.strip() or "Error:" in llm_output_best : # Check if it's an error message or empty
                    logger.info(f"LLM output is empty or indicates an error for prompt (Subreddit: {subreddit}): '{str(llm_output_best)[:100]}...' - Skipping metrics calculation for this entry.")
                    domain_metrics[subreddit]["output_errors"] += 1
                    all_data_metrics["output_errors"] += 1
                    # Still count this as an attempt for the domain and overall
                    domain_metrics[subreddit]["count"] += 1
                    all_data_metrics["count"] += 1
                    
                    # Store minimal info for CSV for errored/empty outputs
                    result_row = {
                        "subreddit": subreddit, "prompt_preview": original_prompt[:100] + "...",
                        "true_label": true_label, "llm_output": llm_output_best,
                        "sbert_score": "N/A", "rougeL_f1": "N/A", "rouge1_f1": "N/A",
                        "bleu_score": "N/A", "meteor_score": "N/A", "error": "Empty or Error Output"
                    }
                    individual_results_for_csv.append(result_row)
                    continue


                # Calculate metrics for the (best) candidate
                sbert_sim = calculate_sbert_similarity(true_label, llm_output_best)
                rouge_scores = calculate_rouge_scores(true_label, llm_output_best)
                bleu = calculate_bleu_score(true_label, llm_output_best)
                meteor = calculate_meteor_score(true_label, llm_output_best)

                current_metrics = {
                    "sbert_score": sbert_sim,
                    "rougeL_f1": rouge_scores["rougeL_fmeasure"],
                    "rouge1_f1": rouge_scores["rouge1_fmeasure"],
                    "bleu_score": bleu,
                    "meteor_score": meteor
                }
                
                # Store for CSV
                individual_results_for_csv.append({
                    "subreddit": subreddit, "prompt_preview": original_prompt[:100] + "...",
                    "true_label": true_label, "llm_output": llm_output_best,
                    **current_metrics
                })


                # Aggregate for domain
                domain_metrics[subreddit]["count"] += 1
                if sbert_model:
                    domain_metrics[subreddit]["sbert_sum"] += sbert_sim
                    domain_metrics[subreddit]["sbert_count"] += 1
                if rouge_scorer_instance:
                    domain_metrics[subreddit]["rougeL_sum"] += rouge_scores["rougeL_fmeasure"]
                    domain_metrics[subreddit]["rougeL_count"] += 1
                    domain_metrics[subreddit]["rouge1_sum"] += rouge_scores["rouge1_fmeasure"]
                    domain_metrics[subreddit]["rouge1_count"] += 1
                if MULTI_EVAL_ENABLED and nltk_packages['punkt']:
                    domain_metrics[subreddit]["bleu_sum"] += bleu
                    domain_metrics[subreddit]["bleu_count"] += 1
                if MULTI_EVAL_ENABLED and meteor_available:
                    domain_metrics[subreddit]["meteor_sum"] += meteor
                    domain_metrics[subreddit]["meteor_count"] += 1

                # Aggregate for all data
                all_data_metrics["count"] += 1
                if sbert_model:
                    all_data_metrics["sbert_sum"] += sbert_sim
                    all_data_metrics["sbert_count"] += 1
                if rouge_scorer_instance:
                    all_data_metrics["rougeL_sum"] += rouge_scores["rougeL_fmeasure"]
                    all_data_metrics["rougeL_count"] += 1
                    all_data_metrics["rouge1_sum"] += rouge_scores["rouge1_fmeasure"]
                    all_data_metrics["rouge1_count"] += 1
                if MULTI_EVAL_ENABLED and nltk_packages['punkt']:
                    all_data_metrics["bleu_sum"] += bleu
                    all_data_metrics["bleu_count"] += 1
                if MULTI_EVAL_ENABLED and meteor_available:
                    all_data_metrics["meteor_sum"] += meteor
                    all_data_metrics["meteor_count"] += 1
                    
    except FileNotFoundError:
        logger.error(f"Input file {filepath} not found.")
        return None, None, None
    except Exception as e:
        logger.error(f"An error occurred while processing {filepath}: {e}", exc_info=True)
        return None, None, None
        
    # Save individual results to CSV
    if individual_results_for_csv:
        csv_filename = os.path.join(OUTPUT_METRICS_DIR, f"Individual_Metrics_Results-{dt.now():%Y%m%d_%H%M%S}.csv")
        try:
            with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
                fieldnames = ["subreddit", "prompt_preview", "true_label", "llm_output", "sbert_score", "rougeL_f1", "rouge1_f1", "bleu_score", "meteor_score", "error"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(individual_results_for_csv)
            logger.info(f"Saved individual item metrics to {csv_filename}")
        except IOError as e:
            logger.error(f"Could not write individual metrics CSV to {csv_filename}: {e}")


    return domain_metrics, all_data_metrics


def calculate_average_metrics(metrics_summary):
    if not metrics_summary or metrics_summary["count"] == 0:
        return {
            "avg_sbert": 0.0, "avg_rougeL": 0.0, "avg_rouge1": 0.0,
            "avg_bleu": 0.0, "avg_meteor": 0.0, "count": 0, "output_errors": metrics_summary.get("output_errors", 0)
        }

    return {
        "avg_sbert": (metrics_summary["sbert_sum"] / metrics_summary["sbert_count"]) if metrics_summary["sbert_count"] > 0 else 0.0,
        "avg_rougeL": (metrics_summary["rougeL_sum"] / metrics_summary["rougeL_count"]) if metrics_summary["rougeL_count"] > 0 else 0.0,
        "avg_rouge1": (metrics_summary["rouge1_sum"] / metrics_summary["rouge1_count"]) if metrics_summary["rouge1_count"] > 0 else 0.0,
        "avg_bleu": (metrics_summary["bleu_sum"] / metrics_summary["bleu_count"]) if metrics_summary["bleu_count"] > 0 else 0.0,
        "avg_meteor": (metrics_summary["meteor_sum"] / metrics_summary["meteor_count"]) if metrics_summary["meteor_count"] > 0 else 0.0,
        "count": metrics_summary["count"],
        "valid_evals_sbert": metrics_summary["sbert_count"],
        "valid_evals_rouge": metrics_summary["rougeL_count"], # Assuming rougeL and rouge1 counts are same
        "valid_evals_bleu": metrics_summary["bleu_count"],
        "valid_evals_meteor": metrics_summary["meteor_count"],
        "output_errors": metrics_summary.get("output_errors", 0)
    }

def log_and_save_metrics(domain_metrics_aggregated, overall_metrics_aggregated):
    logger.info("\n--- Evaluation Metrics Summary ---")
    
    summary_data_for_file = []

    # Log and prepare per-domain metrics
    logger.info("\n--- Metrics per Domain (Subreddit) ---")
    for domain, metrics_sum in sorted(domain_metrics_aggregated.items()):
        avg_metrics = calculate_average_metrics(metrics_sum)
        logger.info(f"Domain: {domain} (Processed: {avg_metrics['count']}, Output Errors: {avg_metrics['output_errors']})")
        if sbert_model: logger.info(f"  Avg SBERT Similarity: {avg_metrics['avg_sbert']:.4f} (over {avg_metrics['valid_evals_sbert']} valid evals)")
        if rouge_scorer_instance:
            logger.info(f"  Avg ROUGE-L F1: {avg_metrics['avg_rougeL']:.4f} (over {avg_metrics['valid_evals_rouge']} valid evals)")
            logger.info(f"  Avg ROUGE-1 F1: {avg_metrics['avg_rouge1']:.4f} (over {avg_metrics['valid_evals_rouge']} valid evals)")
        if MULTI_EVAL_ENABLED and nltk_packages['punkt']:
            logger.info(f"  Avg BLEU Score: {avg_metrics['avg_bleu']:.4f} (over {avg_metrics['valid_evals_bleu']} valid evals)")
        if MULTI_EVAL_ENABLED and meteor_available:
            logger.info(f"  Avg METEOR Score: {avg_metrics['avg_meteor']:.4f} (over {avg_metrics['valid_evals_meteor']} valid evals)")
        
        summary_data_for_file.append({
            "Scope": "Domain", "Identifier": domain, **avg_metrics
        })

    # Log and prepare overall metrics
    logger.info("\n--- Overall Metrics (All Data) ---")
    overall_avg_metrics = calculate_average_metrics(overall_metrics_aggregated)
    logger.info(f"Total Processed: {overall_avg_metrics['count']}, Total Output Errors: {overall_avg_metrics['output_errors']}")
    if sbert_model: logger.info(f"  Overall Avg SBERT Similarity: {overall_avg_metrics['avg_sbert']:.4f} (over {overall_avg_metrics['valid_evals_sbert']} valid evals)")
    if rouge_scorer_instance:
        logger.info(f"  Overall Avg ROUGE-L F1: {overall_avg_metrics['avg_rougeL']:.4f} (over {overall_avg_metrics['valid_evals_rouge']} valid evals)")
        logger.info(f"  Overall Avg ROUGE-1 F1: {overall_avg_metrics['avg_rouge1']:.4f} (over {overall_avg_metrics['valid_evals_rouge']} valid evals)")
    if MULTI_EVAL_ENABLED and nltk_packages['punkt']:
        logger.info(f"  Overall Avg BLEU Score: {overall_avg_metrics['avg_bleu']:.4f} (over {overall_avg_metrics['valid_evals_bleu']} valid evals)")
    if MULTI_EVAL_ENABLED and meteor_available:
        logger.info(f"  Overall Avg METEOR Score: {overall_avg_metrics['avg_meteor']:.4f} (over {overall_avg_metrics['valid_evals_meteor']} valid evals)")

    summary_data_for_file.append({
        "Scope": "Overall", "Identifier": "All Data", **overall_avg_metrics
    })

    # Save summary to a CSV file
    summary_filename = os.path.join(OUTPUT_METRICS_DIR, f"Metrics_Summary_Report-{dt.now():%Y%m%d_%H%M%S}.csv")
    try:
        with open(summary_filename, 'w', newline='', encoding='utf-8-sig') as f:
            if summary_data_for_file:
                # Dynamically create fieldnames based on what's available
                fieldnames = ["Scope", "Identifier", "count", "output_errors"]
                if sbert_model: fieldnames.extend(["avg_sbert", "valid_evals_sbert"])
                if rouge_scorer_instance: fieldnames.extend(["avg_rougeL", "avg_rouge1", "valid_evals_rouge"])
                if MULTI_EVAL_ENABLED and nltk_packages['punkt']: fieldnames.extend(["avg_bleu", "valid_evals_bleu"])
                if MULTI_EVAL_ENABLED and meteor_available: fieldnames.extend(["avg_meteor", "valid_evals_meteor"])
                
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for row_data in summary_data_for_file:
                     # Ensure all keys in fieldnames exist in row_data, provide default if not
                    row_to_write = {field: row_data.get(field, "N/A") for field in fieldnames}
                    writer.writerow(row_to_write)

            else:
                f.write("No summary data generated.\n")
        logger.info(f"Metrics summary report saved to: {summary_filename}")
    except IOError as e:
        logger.error(f"Could not write metrics summary CSV to {summary_filename}: {e}")


def main():
    logger.info("Starting metrics calculation process...")
    initialize_evaluation_resources()

    if not os.path.exists(INPUT_JSONL_FILE):
        logger.error(f"Cannot find input file: {INPUT_JSONL_FILE}. Please make sure it exists.")
        return

    domain_metrics_aggregated, overall_metrics_aggregated = process_results_file(INPUT_JSONL_FILE)

    if domain_metrics_aggregated is not None and overall_metrics_aggregated is not None:
        log_and_save_metrics(domain_metrics_aggregated, overall_metrics_aggregated)
    else:
        logger.error("Metrics calculation failed. No summary will be generated.")

    logger.info("Metrics calculation process finished.")
    logger.info(f"Log file saved to: {log_filename}")


if __name__ == "__main__":
    # Ensure NLTK resources are available or downloaded before evaluation starts
    # This logic is simplified here; more robust checks are in initialize_evaluation_resources
    if 'nltk' in sys.modules:
        for pkg_id in ['punkt', 'wordnet', 'omw-1.4']:
            try:
                if pkg_id == 'punkt': nltk.data.find(f'tokenizers/{pkg_id}')
                else: nltk.data.find(f'corpora/{pkg_id}')
            except LookupError:
                print(f"Attempting to download NLTK resource: {pkg_id}")
                try:
                    nltk.download(pkg_id, quiet=True)
                except Exception:
                    print(f"Failed to download {pkg_id}. Some metrics might be affected.")
    main()