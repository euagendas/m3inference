#!/usr/bin/env python3
# @Zijian Wang

import torch.nn as nn
import torch.nn.functional as F

from .utils import *


class M3InferenceTextModel(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(M3InferenceTextModel, self).__init__()
        self.device = device
        self.batch_size = -1

        self.username_lang_embed = nn.Embedding(EMBEDDING_INPUT_SIZE_LANGS, EMBEDDING_OUTPUT_SIZE_LANGS,
                                                padding_idx=EMB['<empty>'])
        self.screenname_lang_embed = nn.Embedding(EMBEDDING_INPUT_SIZE_LANGS, EMBEDDING_OUTPUT_SIZE_LANGS,
                                                  padding_idx=EMB['<empty>'])
        self.des_lang_embed = nn.Embedding(EMBEDDING_INPUT_SIZE_LANGS, EMBEDDING_OUTPUT_SIZE_LANGS,
                                           padding_idx=EMB['<empty>'])

        self.username_embed = nn.Embedding(EMBEDDING_INPUT_SIZE, EMBEDDING_OUTPUT_SIZE, padding_idx=EMB['<empty>'])
        self.username_dense = nn.Linear(in_features=EMBEDDING_OUTPUT_SIZE + EMBEDDING_OUTPUT_SIZE_LANGS,
                                        out_features=EMBEDDING_OUTPUT_SIZE)
        self.username_lstm = nn.LSTM(input_size=EMBEDDING_OUTPUT_SIZE, hidden_size=LSTM_HIDDEN_SIZE,
                                     num_layers=LSTM_LAYER, batch_first=True, bidirectional=True, dropout=0.25)
        self._init_dense(self.username_dense)

        self.screenname_embed = nn.Embedding(EMBEDDING_INPUT_SIZE_ASCII, EMBEDDING_OUTPUT_SIZE_ASCII,
                                             padding_idx=EMB['<empty>'])
        self.screenname_dense = nn.Linear(in_features=EMBEDDING_OUTPUT_SIZE_ASCII + EMBEDDING_OUTPUT_SIZE_LANGS,
                                          out_features=EMBEDDING_OUTPUT_SIZE_ASCII)
        self.screenname_lstm = nn.LSTM(input_size=EMBEDDING_OUTPUT_SIZE_ASCII, hidden_size=LSTM_HIDDEN_SIZE,
                                       num_layers=LSTM_LAYER, batch_first=True, bidirectional=True, dropout=0.25)
        self._init_dense(self.screenname_dense)

        self.des_embed = nn.Embedding(EMBEDDING_INPUT_SIZE, EMBEDDING_OUTPUT_SIZE, padding_idx=EMB['<empty>'])
        self.des_dense = nn.Linear(in_features=EMBEDDING_OUTPUT_SIZE + EMBEDDING_OUTPUT_SIZE_LANGS,
                                   out_features=EMBEDDING_OUTPUT_SIZE)
        self.des_lstm = nn.LSTM(input_size=EMBEDDING_OUTPUT_SIZE, hidden_size=LSTM_HIDDEN_SIZE,
                                num_layers=LSTM_LAYER_DES, batch_first=True, bidirectional=True, dropout=0.25)
        self._init_dense(self.des_dense)

        self.merge_dense = nn.Linear(in_features=LSTM_HIDDEN_SIZE * 6, out_features=LSTM_HIDDEN_SIZE)
        self.gender_out_dense = nn.Linear(in_features=LSTM_HIDDEN_SIZE, out_features=LINEAR_OUTPUT_SIZE)
        self.org_out_dense = nn.Linear(in_features=LSTM_HIDDEN_SIZE, out_features=LINEAR_OUTPUT_SIZE)
        self.age_out_dense = nn.Linear(in_features=LSTM_HIDDEN_SIZE, out_features=4)

        self._init_dense(self.merge_dense)
        self._init_dense(self.gender_out_dense)
        self._init_dense(self.org_out_dense)
        self._init_dense(self.age_out_dense)

    def _init_dense(self, layer):
        nn.init.kaiming_normal_(layer.weight)
        nn.init.uniform_(layer.bias)

    def _init_hidden(self):

        self.username_h0 = torch.zeros(2 * LSTM_LAYER, self.batch_size, LSTM_HIDDEN_SIZE).to(self.device)
        self.username_c0 = torch.zeros(2 * LSTM_LAYER, self.batch_size, LSTM_HIDDEN_SIZE).to(self.device)

        self.screenname_h0 = torch.zeros(2 * LSTM_LAYER, self.batch_size, LSTM_HIDDEN_SIZE).to(self.device)
        self.screenname_c0 = torch.zeros(2 * LSTM_LAYER, self.batch_size, LSTM_HIDDEN_SIZE).to(self.device)

        self.des_h0 = torch.zeros(2 * LSTM_LAYER_DES, self.batch_size, LSTM_HIDDEN_SIZE).to(self.device)
        self.des_c0 = torch.zeros(2 * LSTM_LAYER_DES, self.batch_size, LSTM_HIDDEN_SIZE).to(self.device)

    def forward(self, data, label=None):

        lang, username, username_len, screenname, screenname_len, des, des_len = data
        self.batch_size = len(lang)
        self._init_hidden()

        username_lang_embed = self.username_lang_embed(lang)
        screenname_lang_embed = self.screenname_lang_embed(lang)
        des_lang_embed = self.des_lang_embed(lang)

        self.merge_layer = []

        username_embed = self.username_embed(username)

        username_embed = self.username_dense(torch.cat([username_embed,
                                                        username_lang_embed.unsqueeze(1).expand(self.batch_size,
                                                                                                USERNAME_LEN,
                                                                                                EMBEDDING_OUTPUT_SIZE_LANGS)],
                                                       2))
        username_pack, username_unsort = pack_wrapper(username_embed, username_len)
        self.username_lstm.flatten_parameters()
        username_out, (self.username_h0, self.username_c0) = self.username_lstm(username_pack, (
            self.username_h0, self.username_c0))
        username_output = unpack_wrapper(username_out, username_unsort)

        self.merge_layer.append(torch.cat([username_output[torch.arange(0, self.batch_size, dtype=torch.int64),
                                           username_len - 1, :LSTM_HIDDEN_SIZE],
                                           username_output[torch.arange(0, self.batch_size, dtype=torch.int64),
                                           torch.zeros(self.batch_size, dtype=torch.long), LSTM_HIDDEN_SIZE:]], 1))

        screenname_embed = self.screenname_embed(screenname)

        screenname_embed = self.screenname_dense(torch.cat([screenname_embed,
                                                            screenname_lang_embed.unsqueeze(1).expand(
                                                                self.batch_size, SCREENNAME_LEN,
                                                                EMBEDDING_OUTPUT_SIZE_LANGS)], 2))

        screenname_pack, screenname_unsort = pack_wrapper(screenname_embed, screenname_len)
        self.screenname_lstm.flatten_parameters()
        screenname_out, (self.screenname_h0, self.screenname_c0) = self.screenname_lstm(screenname_pack, (
            self.screenname_h0, self.screenname_c0))
        screenname_output = unpack_wrapper(screenname_out, screenname_unsort)
        self.merge_layer.append(torch.cat([screenname_output[torch.arange(0, self.batch_size, dtype=torch.int64),
                                           screenname_len - 1, :LSTM_HIDDEN_SIZE],
                                           screenname_output[torch.arange(0, self.batch_size, dtype=torch.int64),
                                           torch.zeros(self.batch_size, dtype=torch.long), LSTM_HIDDEN_SIZE:]], 1))

        des_embed = self.des_embed(des)

        des_embed = self.des_dense(torch.cat([des_embed,
                                              des_lang_embed.unsqueeze(1).expand(self.batch_size, DES_LEN,
                                                                                 EMBEDDING_OUTPUT_SIZE_LANGS)],
                                             2))

        des_pack, des_unsort = pack_wrapper(des_embed, des_len)
        self.des_lstm.flatten_parameters()
        des_out, (self.des_h0, self.des_c0) = self.des_lstm(des_pack, (self.des_h0, self.des_c0))
        des_output = unpack_wrapper(des_out, des_unsort)
        self.merge_layer.append(torch.cat(
            [des_output[torch.arange(0, self.batch_size, dtype=torch.int64), des_len - 1, :LSTM_HIDDEN_SIZE],
             des_output[torch.arange(0, self.batch_size, dtype=torch.int64),
             torch.zeros(self.batch_size, dtype=torch.long), LSTM_HIDDEN_SIZE:]], 1))

        merged_cat = torch.cat(self.merge_layer, 1)

        dense = F.relu(self.merge_dense(merged_cat), inplace=True)
        if label == "gender":
            return F.softmax(self.gender_out_dense(dense), dim=1)
        elif label == "age":
            return F.softmax(self.age_out_dense(dense), dim=1)
        elif label == "org":
            return F.softmax(self.org_out_dense(dense), dim=1)
        else:
            return F.softmax(self.gender_out_dense(dense), dim=1), \
                   F.softmax(self.age_out_dense(dense), dim=1), \
                   F.softmax(self.org_out_dense(dense), dim=1)


if __name__ == '__main__':
    # python -m m3inference.text_model
    # sanity check that the model init. is working
    m3 = M3InferenceTextModel()
