# Аналог стандартной обёртки, эксперимент по разморозке отдельных слоёв не дал результатов
from transformers import BertModel, BertPreTrainedModel, AutoTokenizer
import torch.nn as nn
from web.neural_network.BERT2 import config

class BertForMultiClassSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=6):
        super(BertForMultiClassSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            return loss
        else:
            return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


def get_model_and_tokenizer(model_name, labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForMultiClassSequenceClassification.from_pretrained(model_name,num_labels=len(labels)).to(config.device)

    model.config.label2id = {label: i for i, label in enumerate(labels)}
    model.config.id2label = {i: label for i, label in enumerate(labels)}

    return model, tokenizer
