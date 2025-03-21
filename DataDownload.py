
from datasets import load_dataset
import json


def encode_tokens(example, word2idx, max_len=128):
    input_ids = [word2idx.get(token, word2idx["<PAD>"]) for token in example["tokens"]]

    # Đổi từ `ner_tags` sang `labels`
    labels = example.get("ner_tags", [0] * len(example["tokens"]))

    # Padding input_ids
    input_ids += [word2idx["<PAD>"]] * (max_len - len(input_ids))
    input_ids = input_ids[:max_len]  # Cắt nếu quá dài

    # Padding labels
    labels += [0] * (max_len - len(labels))  # 0 là nhãn mặc định cho padding
    labels = labels[:max_len]

    return {
        "input_ids": input_ids,
        "attention_mask": [1 if i < len(example["tokens"]) else 0 for i in range(max_len)],
        "labels": labels  # Đã thay thế `labels` từ `ner_tags`
    }


# Load dataset CoNLL-2003 từ Hugging Face
dataset = load_dataset("eriktks/conll2003",trust_remote_code=True)
all_tokens = set(token for example in dataset["train"] for token in example["tokens"])
word2idx = {word: idx for idx, word in enumerate(all_tokens, start=1)}
word2idx["<PAD>"] = 0  # Padding token
# Lưu train_data, valid_data, test_data vào file JSON
def save_to_json(dataset, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
train_data = [encode_tokens(example, word2idx) for example in dataset["train"]]
valid_data = [encode_tokens(example, word2idx) for example in dataset["validation"]]
test_data = [encode_tokens(example, word2idx) for example in dataset["test"]]
save_to_json(train_data, "train_data.json")
save_to_json(valid_data, "valid_data.json")
save_to_json(test_data, "test_data.json")

