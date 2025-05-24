import json
import re
import math
import pandas as pd

def compute_metrics(input_path, output_csv_path):

    records = []
    ignored = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            output = rec.get('output', '')
        
            cleaned = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)

            nums = re.findall(r'-?\d+(?:\.\d+)?', cleaned)
            if not nums:
                ignored += 1
                continue
            pred = float(nums[-1])
            true = rec.get('true_label')
            try:
                true = float(true)
            except:
                ignored += 1
                continue
            domain = rec.get('subreddit', rec.get('domain', 'UNKNOWN'))
            records.append({'domain': domain, 'pred': pred, 'true': true})

    df = pd.DataFrame(records)


    metrics = []
    for domain, group in df.groupby('domain'):
        rmse = math.sqrt(((group['pred'] - group['true']) ** 2).mean())
        mae = (group['pred'] - group['true']).abs().mean()
        metrics.append({'domain': domain, 'rmse': rmse, 'mae': mae, 'count': len(group)})


    overall_rmse = math.sqrt(((df['pred'] - df['true']) ** 2).mean())
    overall_mae = (df['pred'] - df['true']).abs().mean()
    metrics.append({'domain': 'ALL', 'rmse': overall_rmse, 'mae': overall_mae, 'count': len(df)})


    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_csv_path, index=False)

    print(f"Metrics saved to {output_csv_path}")
    print(metrics_df)

if __name__ == "__main__":
    compute_metrics('Pse_3.2_results_DS.jsonl', 'DS_Pseu_results.csv')
