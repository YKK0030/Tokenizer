# Custom Tokenizer Training (`train.py`)

This script trains a custom tokenizer using the [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers) library with Byte Pair Encoding (BPE). It also demonstrates a simple regex-based tokenizer for comparison.

## Features

- Reads raw text data from `data.txt`
- Tokenizes text using a basic regex tokenizer
- Trains a BPE tokenizer with special tokens (`[UNK]`, `[PAD]`, `[CLS]`, `[SEP]`, `[MASK]`)
- Saves the trained tokenizer to `tokenizer.json`
- Encodes and decodes the text using the trained tokenizer
- Prints token-to-id and id-to-token mappings

## Requirements

- Python 3.6+
- [tokenizers](https://pypi.org/project/tokenizers/)

Install dependencies:
```sh
pip install tokenizers
```

## Usage

1. Place your training data in a file named `data.txt` in the same directory.
2. Run the script:
   ```sh
   python train.py
   ```

## Output

- Prints tokens, vocabulary, token-to-id mappings, encoded and decoded text.
- Saves the trained tokenizer as `tokenizer.json`.

## Customization

- Adjust `vocab_size` and `min_frequency` in the `BpeTrainer` as needed.
- Add or remove special tokens in the `special_tokens` list.

---

**Note:**  
The script uses both a simple regex tokenizer and a BPE tokenizer for demonstration. For advanced NLP tasks, prefer the BPE tokenizer.