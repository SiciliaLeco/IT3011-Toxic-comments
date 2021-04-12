import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BertClassifier, self).__init__()
        # TODO:
        D_in = 768
        H = 50
        D_out = 2
        self.bert = BertModel.from_pretrained('bert-base-uncased')
    def forward(self, input_ids, attention_mask):
        #TODO:
        return input_ids