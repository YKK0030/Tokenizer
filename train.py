import re
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

with open("data.txt", encoding="utf-8") as f:
    data = f.read()

def basicTokenizer(text):
    tokens = re.findall(r"[\w]+|[^\s\w]", text)
    return tokens

tokens = basicTokenizer(data)
print("Tokens:", tokens)
print("Data length:", len(data))

tokenizer = Tokenizer(models.BPE())
Tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=1000,
    min_frequency=2,
    special_tokens=[
        "[UNK]",
        "[PAD]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
    ],
)
tokenizer.train_from_iterator(data, trainer=trainer)

tokenizer.save("tokenizer.json")

tokens = tokenizer.encode(data).tokens
# print("Tokens:", tokens)
# print("Tokens length:", len(tokens))

vocab = sorted(set(tokens))
token_to_id = {token: i for i, token in enumerate(vocab)}
id_to_token = {id: token for token, id in token_to_id.items()}

print("TOKEN TO ID",token_to_id)
print("ID TO TOKEN",id_to_token)

encoded = [token_to_id[token] for token in tokens]
print("Encoded:", encoded )
# print("Encoded length:", len(encoded))

decode = ''.join([id_to_token.get(id_, '?') for id_ in encoded])
print("Decoded:", decode)
# print("Decoded length:", len(decode))

