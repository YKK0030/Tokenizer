import re
import os
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

with open("data.txt", encoding="utf-8") as f:
    data = f.read()

def basicTokenizer(text):
    tokens = re.findall(r"[\w]+|[^\s\w]", text)
    return tokens

tokens = basicTokenizer(data)
print("Tokens:", tokens)
print("Data length:", len(data))

tokenizer_path = "tokenizer.json"

if not os.path.exists(tokenizer_path):
    print("Training new tokenizer...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=1000,
        min_frequency=2,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
    )

    tokenizer.train_from_iterator([data], trainer=trainer)
    tokenizer.save(tokenizer_path)
else:
    print("Loading existing tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)

# Encode and map tokens
tokens = tokenizer.encode(data).tokens
vocab = sorted(set(tokens))
token_to_id = {token: i for i, token in enumerate(vocab)}
id_to_token = {i: token for token, i in token_to_id.items()}

print("TOKEN TO ID", token_to_id)
print("ID TO TOKEN", id_to_token)

encoded = [token_to_id[token] for token in tokens]
print("Encoded:", encoded)

decoded = ''.join([id_to_token.get(id_, '?') for id_ in encoded])
print("Decoded:", decoded)
