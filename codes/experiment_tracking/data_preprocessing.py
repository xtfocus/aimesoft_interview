"""
Preprocessing script to create Torch dataloaders

1. Read the txt file
2. bert-base-cased tokenizing
3. Train-val-test split

"""

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

BATCH_SIZE = 32
RANDOM_SEED = 111  # For data-split
MAX_LEN = 40  # Max sequence length (units: number of tokens)
MODEL_NAME = (
    "bert-base-cased"  # Cased model is more sensitive to exclamation: bad vs BAD
)


class TextDataset(Dataset):
    """
    TextDataset for the Sentiment Classification problem
    """

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        # Encoding documentation
        # https://huggingface.co/docs/transformers/v4.38.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode_plus
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",  # Truncate to `max_length`
            return_token_type_ids=False,  # token Type IDs is only relevant when input includes 2 sequences (such as for QA problem. See https://huggingface.co/docs/transformers/v4.38.1/en/glossary#token-type-ids)
            return_attention_mask=True,
            return_tensors="pt",  # return pytorch tensors
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = TextDataset(
        texts=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=4)


df = pd.read_csv("data/sentiment.txt", sep="\t", names=["content", "label"])

df["text"] = df[
    "content"
]  # You can try injecting something here, see if that influence CLS.
del df["content"]

class_names = list(df["label"].unique())

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

df_train, df_test = train_test_split(
    df, test_size=0.3, stratify=df["label"], random_state=RANDOM_SEED
)

# Stratify split by `label`. Which is necessary for skewed labels.
# The dataset at hand however doesn't have skewness in label.
df_dev, df_test = train_test_split(
    df_test, test_size=(1 / 3), stratify=df_test["label"], random_state=RANDOM_SEED
)


train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
dev_data_loader = create_data_loader(df_dev, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

train_size = len(df_train)
dev_size = len(df_dev)
