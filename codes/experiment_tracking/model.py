"""
model classes for sentiment classification
"""

import torch
from torch import nn
from transformers import BertModel

from data_preprocessing import MODEL_NAME


class SentimentClassifier(nn.Module):
    """
    BERT + nn.Linear using [CLS] token (pooled_output)
    """

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(MODEL_NAME)

        self.drop = nn.Dropout(p=0.3)

        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,  # https://stackoverflow.com/questions/65082243/dropout-argument-input-position-1-must-be-tensor-not-str-when-using-bert
        )

        dropped = self.drop(pooled_output)

        return self.out(dropped)


class SentimentClassifierMultiLinear(nn.Module):
    """
    BERT + nn.Linear using [CLS] token (pooled_output)
         + another nn.Linear for classification
    """

    def __init__(self, n_classes):
        super(SentimentClassifierMultiLinear, self).__init__()

        self.bert = BertModel.from_pretrained(MODEL_NAME)

        self.drop1 = nn.Dropout(p=0.3)

        self.condense = nn.Linear(self.bert.config.hidden_size, 32)

        self.drop2 = nn.Dropout(p=0.3)

        self.out = nn.Linear(32, n_classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )

        dropped_1 = self.drop1(pooled_output)

        condensed = self.condense(dropped_1)

        dropped_2 = self.drop2(condensed)

        return self.out(dropped_2)


class SentimentClassifierUntrainedCLS(nn.Module):
    """
    BERT + nn.Linear using last_hidden_state[:,0,...], which is CLS before further
        processing through the layers used for the next sentence prediction training task
    """

    def __init__(self, n_classes):
        super(SentimentClassifierUntrainedCLS, self).__init__()

        self.bert = BertModel.from_pretrained(MODEL_NAME)

        self.drop = nn.Dropout(p=0.3)

        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )

        dropped = self.drop(last_hidden_state[:, 0, ...])  # Use the untrained CLS

        return self.out(dropped)


class SentimentClassifierUntrainedCLSMultiLastLayers(nn.Module):
    """
    BERT + nn.Linear using the concatenation of last_hidden_state[:,0,...], second_last..., third ...
         + another nn.Linear
    """

    def __init__(self, n_classes, n_lasts=4):
        assert (
            n_lasts > 1
        ), "Just use `SentimentClassifier` if you don't want to use multiple layers' outputs"
        super(SentimentClassifierUntrainedCLSMultiLastLayers, self).__init__()

        self.n_lasts = n_lasts  # Number of last layers to use

        self.bert = BertModel.from_pretrained(MODEL_NAME)

        self.drop1 = nn.Dropout(p=0.3)

        self.condense = nn.Linear(self.bert.config.hidden_size * self.n_lasts, 32)

        self.drop2 = nn.Dropout(p=0.3)

        self.out = nn.Linear(32, n_classes)

    def forward(self, input_ids, attention_mask):
        _, _, hidden_states = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
            output_hidden_states=True,
        )

        cls_list = [hidden_states[-i][:, 0, ...] for i in range(1, self.n_lasts + 1)]

        cat_cls = torch.cat(cls_list, -1)

        dropped_1 = self.drop1(cat_cls)

        condensed = self.condense(dropped)

        dropped_2 = self.drop2(cat_cls)

        return self.out(dropped_2)


class SentimentClassifierUntrainedCLSMultiLastLayers(nn.Module):
    """
    BERT + nn.Linear using the concatenation of last_hidden_state[:,0,...], second_last..., third ...
    """

    def __init__(self, n_classes, n_lasts=4):
        assert (
            n_lasts > 1
        ), "Just use `SentimentClassifier` if you don't want to use multiple layers' outputs"
        super(SentimentClassifierUntrainedCLSMultiLastLayers, self).__init__()

        self.n_lasts = n_lasts  # Number of last layers to use

        self.bert = BertModel.from_pretrained(MODEL_NAME)

        self.drop = nn.Dropout(p=0.3)

        self.out = nn.Linear(self.bert.config.hidden_size * self.n_lasts, 32)

    def forward(self, input_ids, attention_mask):
        _, _, hidden_states = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
            output_hidden_states=True,
        )

        cls_list = [hidden_states[-i][:, 0, ...] for i in range(1, self.n_lasts + 1)]

        cat_cls = torch.cat(cls_list, -1)

        dropped = self.drop(cat_cls)

        return self.out(dropped)


class SentimentClassifierUntrainedCLSMultiLastLayersMultiLinear(nn.Module):
    """
    BERT + nn.Linear using the concatenation of last_hidden_state[:,0,...], second_last..., third ...
         + another nn.Linear
    """

    def __init__(self, n_classes, n_lasts=4):
        assert (
            n_lasts > 1
        ), "Just use `SentimentClassifier` if you don't want to use multiple layers' outputs"
        super(
            SentimentClassifierUntrainedCLSMultiLastLayersMultiLinear, self
        ).__init__()

        self.n_lasts = n_lasts  # Number of last layers to use

        self.bert = BertModel.from_pretrained(MODEL_NAME)

        self.drop1 = nn.Dropout(p=0.3)

        self.condense = nn.Linear(self.bert.config.hidden_size * self.n_lasts, 32)

        self.drop2 = nn.Dropout(p=0.3)

        self.out = nn.Linear(32, n_classes)

    def forward(self, input_ids, attention_mask):
        hidden_states = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
            output_hidden_states=True,
        ).hidden_states

        cls_list = [hidden_states[-i][:, 0, ...] for i in range(1, self.n_lasts + 1)]

        cat_cls = torch.cat(cls_list, -1)

        dropped_1 = self.drop1(cat_cls)

        condensed = self.condense(dropped_1)

        dropped_2 = self.drop2(condensed)

        return self.out(dropped_2)
