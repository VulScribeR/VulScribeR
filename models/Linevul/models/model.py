import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, input_ids=None, labels=None, loss_func=None, class_weights=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))
        # print(outputs)
        logits = outputs[0]
        # print(logits)
        prob = torch.sigmoid(logits)
        if labels is not None:

            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob
