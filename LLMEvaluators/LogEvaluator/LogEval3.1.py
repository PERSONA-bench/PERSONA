import json
import re
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import warnings


warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.metrics')

def extract_prediction(output_text):

    cleaned_output = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL)
    match = re.findall(r'\b(positive|negative)\b', cleaned_output, re.IGNORECASE)
    if match:
        return match[-1].lower()
    return None

def compute_classification_metrics(input_path, output_csv_path):

    records = []
    ignored_records = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                output = rec.get('output', '')
                
                pred_label_str = extract_prediction(output)
                true_label_str = rec.get('true_label', '').lower()
                if pred_label_str not in ['positive', 'negative'] or \
                   true_label_str not in ['positive', 'negative']:
                    ignored_records += 1
                    continue
                    
                domain = rec.get('subreddit', rec.get('domain', 'UNKNOWN'))
                records.append({'domain': domain, 'pred_label': pred_label_str, 'true_label': true_label_str})
            except json.JSONDecodeError:
                ignored_records +=1
                print(f"Warning: Skipping invalid JSON line: {line.strip()}")
            except Exception as e:
                ignored_records += 1
                print(f"Warning: Error processing line: {line.strip()} - {e}")


    if not records:
        print("No valid records found to process.")
        return

    df = pd.DataFrame(records)
    
    label_map = {'positive': 1, 'negative': 0}
    df['pred'] = df['pred_label'].map(label_map)
    df['true'] = df['true_label'].map(label_map)

    metrics_list = []


    for domain, group in df.groupby('domain'):
        if len(group['true'].unique()) < 2 and len(group) > 0 : # MCC F1 对于单类别预测没有意义或会报错
            acc = accuracy_score(group['true'], group['pred'])
            f1 = f1_score(group['true'], group['pred'], pos_label=group['true'].unique()[0], average='binary' if len(group['true'].unique()) == 1 else 'weighted', zero_division=0)
            mcc = 0 
            print(f"Warning: Domain '{domain}' has only one class in true labels. MCC set to 0, F1 calculated for the present class.")
        elif len(group) == 0:
            acc, f1, mcc = 0,0,0
        else:
            acc = accuracy_score(group['true'], group['pred'])
            f1 = f1_score(group['true'], group['pred'], average='weighted', zero_division=0) #使用 weighted 避免二分类问题中的 pos_label 问题
            mcc = matthews_corrcoef(group['true'], group['pred'])
            
        metrics_list.append({
            'domain': domain,
            'accuracy': acc,
            'f1_score': f1,
            'mcc': mcc,
            'count': len(group)
        })


    if len(df['true'].unique()) < 2 and len(df) > 0:
        overall_acc = accuracy_score(df['true'], df['pred'])
        overall_f1 = f1_score(df['true'], df['pred'], pos_label=df['true'].unique()[0], average='binary' if len(df['true'].unique()) == 1 else 'weighted', zero_division=0)
        overall_mcc = 0 
        print("Warning: Overall data has only one class in true labels. MCC set to 0, F1 calculated for the present class.")
    elif len(df) == 0:
        overall_acc, overall_f1, overall_mcc = 0,0,0
    else:
        overall_acc = accuracy_score(df['true'], df['pred'])
        overall_f1 = f1_score(df['true'], df['pred'], average='weighted', zero_division=0)
        overall_mcc = matthews_corrcoef(df['true'], df['pred'])
        
    metrics_list.append({
        'domain': 'ALL',
        'accuracy': overall_acc,
        'f1_score': overall_f1,
        'mcc': overall_mcc,
        'count': len(df)
    })
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(output_csv_path, index=False)

    print(f"\nProcessed {len(df)} records.")
    if ignored_records > 0:
        print(f"Ignored {ignored_records} records due to missing/invalid labels or JSON errors.")
    print(f"\nMetrics saved to {output_csv_path}")
    print("\n--- Metrics per Domain ---")

    domain_metrics_df = metrics_df[metrics_df['domain'] != 'ALL']
    if not domain_metrics_df.empty:
        print(domain_metrics_df.to_string(index=False))
    
    overall_metrics_df = metrics_df[metrics_df['domain'] == 'ALL']
    if not overall_metrics_df.empty:
        print("\n--- Overall Metrics ---")
        print(overall_metrics_df.to_string(index=False))


if __name__ == "__main__":

    input_jsonl_path = 'Pse_3.2_results_DS.jsonl' 
    output_csv_filename = 'DSclassification_metrics_results.csv'
    
    compute_classification_metrics(input_jsonl_path, output_csv_filename)