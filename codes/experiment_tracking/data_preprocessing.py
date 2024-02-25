"""
transformers.v3 code
Preprocessing text file --> Torch dataloaders

1. Read the text file
2. Bert-base-cased tokenizing
3. Train-val-test split

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import  BertTokenizer
import torch


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer,max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self,item):
        text = str(self.texts[item])
        label = self.labels[item]

        # https://huggingface.co/docs/transformers/v4.38.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode_plus
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True, # True means truncate to `max_length`
            padding="max_length",
            return_token_type_ids=False, # token Type IDs is relevant when input includes 2 sequences (such as QA. see https://huggingface.co/docs/transformers/v4.38.1/en/glossary#token-type-ids)
            return_attention_mask=True,
            return_tensors='pt',
            )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
            }
    
    
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = TextDataset(
        texts = df.text.to_numpy(), 
        labels = df.label.to_numpy(),
        tokenizer = tokenizer,
        max_len=max_len)

    return DataLoader(ds,
                      batch_size=batch_size,
                      num_workers=4)



RANDOM_SEED = 111

MODEL_NAME = 'bert-base-cased' # Cased model is more sensitive to exclamation

MAX_LEN = 40 # Max sequence length (units: number of tokens)

df = pd.read_csv("data/sentiment.txt", sep="\t", names=["content", "label"])
df['text'] =  df['content'] # You can try injecting something here, see if that influence CLS.
del(df['content'])

class_names = list(df['label'].unique())

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

df_train, df_test = train_test_split(df,
                                     test_size=0.3,
                                     stratify=df['label'],
                                     random_state=RANDOM_SEED)

df_dev, df_test = train_test_split(df_test,
                                   test_size=(1 / 3),
                                   stratify=df_test['label'],
                                   random_state=RANDOM_SEED)


BATCH_SIZE = 32

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
dev_data_loader = create_data_loader(df_dev, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

train_size = len(df_train)
dev_size = len(df_dev)