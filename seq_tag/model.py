import os
import torch
import torch.nn as nn

from transformers import BertModel

from .layers import CRF

here = os.path.dirname(os.path.abspath(__file__))


class BertBilstmCrf(nn.Module):

    def __init__(self, hparams):
        super(BertBilstmCrf, self).__init__()

        self.pretrained_model_path = hparams.pretrained_model_path or 'bert-base-chinese'
        self.embedding_dim = hparams.embedding_dim
        self.rnn_hidden_dim = hparams.rnn_hidden_dim
        self.rnn_num_layers = hparams.rnn_num_layers
        self.rnn_bidirectional = hparams.rnn_bidirectional
        self.dropout = hparams.dropout
        self.tagset_size = hparams.tagset_size

        self.bert_model = BertModel.from_pretrained(self.pretrained_model_path)
        self.lstm = nn.LSTM(self.embedding_dim,
                            self.rnn_hidden_dim // (2 if self.rnn_bidirectional else 1),
                            num_layers=self.rnn_num_layers, batch_first=True,
                            bidirectional=self.rnn_bidirectional)
        self.drop = nn.Dropout(self.dropout)
        self.hidden2tag = nn.Linear(self.rnn_hidden_dim, self.tagset_size)
        self.crf = CRF(num_tags=self.tagset_size)

    def _get_emission_scores(self, input_ids, token_type_ids=None, attention_mask=None):
        embeds = self.bert_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=False)[0]
        lstm_out, _ = self.lstm(embeds)
        lstm_dropout = self.drop(lstm_out)
        emissions = self.hidden2tag(lstm_dropout)
        return emissions

    def forward(self, input_ids, tags, token_type_ids=None, attention_mask=None):
        emissions = self._get_emission_scores(input_ids, token_type_ids, attention_mask)
        loss = -1 * self.crf(emissions, tags, mask=attention_mask.byte())
        return loss

    def decode(self, input_ids, token_type_ids=None, attention_mask=None):
        emissions = self._get_emission_scores(input_ids, token_type_ids, attention_mask)
        tags = self.crf.decode(emissions, mask=attention_mask.byte())
        return tags
