# Persona Benchmark – Quick‑Start Guide

A step‑by‑step walkthrough for reproducing the **Personalized Conversational Benchmark** experiments contained in this repository.

---

## 1 · Project layout

```
.
├── demo/                      # toy sample used in the paper
│   ├── demo.json
│   └── …
├── LLMEvaluators/             # common evaluation utilities (imported by GPT*.py)
├── PromptMakers/              # reusable prompt‑building modules
├── 3.1PromptMaker.py          # Task 3 .1 – sentiment classification prompt generator
├── 3.2PromptMaker.py          # Task 3 .2 – exact‑score regression prompt generator
├── 3.3PromptMaker.py          # Task 3 .3 – next‑reply generation prompt generator
├── GPT3.1.py                  # Azure OpenAI evaluator for Task 3 .1
├── GPT3.2.py                  # Azure OpenAI evaluator for Task 3 .2
├── GPT3.3.py                  # Azure OpenAI evaluator for Task 3 .3 / 3 .4
├── requirements.txt           # all runtime dependencies
└── README.md                  # ← **you are here**
```

---

## 2 · Prerequisites

| Requirement       | Version tested | Notes                                                                    |
| ----------------- | -------------- | ------------------------------------------------------------------------ |
| **Python**        | 3.9 – 3.11     | 3.10 recommended                                                         |
| **git + Git LFS** | ≥ 2.39         | needed to pull the dataset                                               |
| **Azure OpenAI**  | any paid tier  | for evaluation; you can swap in OpenAI Cloud by adapting `MODEL_CONFIGS` |

> **GPU support** is optional. `torch==2.6.0+cu118` (in *requirements.txt*) automatically installs CUDA‑enabled wheels if a compatible NVIDIA driver is present.

---

## 3 · Environment setup

<details>
<summary><strong>🐍 Create & activate a virtual environment</strong></summary>

### Unix / macOS – <em>venv</em>

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows – <em>PowerShell</em>

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

</details>

### Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4 · Download the dataset

```bash
git lfs install
# Downloads into ./data/Personalized_Conversational_Benchmark
git clone https://huggingface.co/datasets/ShawnLi0415/Personalized_Conversational_Benchmark \
      data/Personalized_Conversational_Benchmark
```

If you prefer the **Hugging Face Hub CLI**:

```bash
pip install --upgrade huggingface_hub
huggingface-cli login          # optional for public datasets
huggingface-cli download \
       ShawnLi0415/Personalized_Conversational_Benchmark \
       --local-dir data/Personalized_Conversational_Benchmark
```

---

## 5 · Configure LLM credentials

All GPT scripts read their keys from the `MODEL_CONFIGS` dictionary. The cleanest approach is to expose the values as environment variables and patch the scripts to reference them:

```bash
# Bash / zsh
export AZURE_OPENAI_ENDPOINT="https://<your‑resource>.openai.azure.com/"
export AZURE_OPENAI_API_KEY="sk‑REPLACE_ME"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

Then, inside *GPT*.py, replace the empty strings:

```python
MODEL_CONFIGS = {
    "gpt": {
        "api_type": "azure",
        "api_key":  os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version":    os.getenv("AZURE_OPENAI_API_VERSION"),
        "deployment_name": "gpt"
    }
}
```

---

## 6 · Generate prompts

> **Tip:** Each *PromptMaker* script hard‑codes its input JSON path. Either edit the constant or symlink the desired split to `demo/demo.json`.

```bash
## Task 3 .1 – sentiment classification
python 3.1PromptMaker.py

## Task 3 .2 – exact score prediction
python 3.2PromptMaker.py

## Task 3 .3 – next‑reply body generation
python 3.3PromptMaker.py
```

Outputs will appear in the project root:

```
WithConversationPrompts_ScorePrediction_Refactored.jsonl
WithoutConversationPrompts_ScorePrediction_Refactored.jsonl
WithConversationPrompts_ExactScorePrediction.jsonl # For Rand Experiments
… etc.
```

---

## 7 · Run evaluation

After prompts are created and the API keys are in place:

```bash
## Task 3 .1 (binary sentiment)
python GPT3.1.py --model gpt  # additional CLI flags are accepted

## Task 3 .2 (regression)
python GPT3.2.py             # reads from With/Without *.jsonl automatically

## Task 3 .3 / 3 .4 (generation & multi‑metric eval)
python GPT3.3.py             # long run; produces a detailed log & summary
```

All evaluators write timestamped logs plus metric summaries to the working directory.

---

## 8 · One‑command quick‑demo

Create `run_demo.sh` in the repo root:

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1 – Generate all prompts
for t in 3.1 3.2 3.3; do
  python "${t}PromptMaker.py"
done

# 2 – Evaluate with a single GPT deployment
for t in 3.1 3.2 3.3; do
  python "GPT${t}.py"
done
```

```bash
chmod +x run_demo.sh
./run_demo.sh
```

The script finishes with three CSV/JSONL metric reports in the current directory.

---

## 9 · Troubleshooting

| Symptom                             | Checklist                                                                                            |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError`               | Did you **activate** the virtual environment and run `pip install -r requirements.txt`?              |
| `openai.RateLimitError`             | Verify your Azure quota. Use smaller batches or add `time.sleep()` in evaluator loops.               |
| `FileNotFoundError: demo/demo.json` | Point the *PromptMaker* `INPUT_JSON_FILE` constant to an existing split or symlink the desired JSON. |

---
... 