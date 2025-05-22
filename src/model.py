import torch
from torch import nn
from transformers import BertModel

class MBTIClassifier(nn.Module):
    def __init__(self, pretrained_model: str = 'bert-base-uncased', num_labels: int = 4):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        logits = self.classifier(self.dropout(pooled))
        return logits