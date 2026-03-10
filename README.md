# AI Training Copilot

A Python project that fine-tunes a GPT-2 language model on Wikipedia articles about AI topics, then generates text from a custom prompt.

## Overview

The script `train_wiki_ai.py` performs four steps:

1. **Gather Data** — Fetches Wikipedia articles on *Artificial intelligence*, *Machine learning*, *Deep learning*, and *Neural network*, saving them to `wiki_dataset.txt`.
2. **Prepare the Model** — Loads the `distilgpt2` tokenizer and model from Hugging Face and tokenizes the corpus into fixed-size chunks.
3. **Train the AI** — Fine-tunes the model for 3 epochs using the Hugging Face `Trainer` API and saves the result to `./my_wiki_ai`.
4. **Test the AI** — Loads the saved model and generates text from the prompt `"Artificial intelligence is "`.

## Requirements

- Python 3.8+
- The packages listed in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python train_wiki_ai.py
```

Training may take several minutes on a CPU. Use a machine with a GPU to speed it up.

## Output

- `wiki_dataset.txt` — Raw Wikipedia text used for training (excluded from version control).
- `my_wiki_ai/` — Saved fine-tuned model and tokenizer (excluded from version control).
