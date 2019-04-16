# coding = utf-8
# author = xy

import torch
from torch import nn
from modules import embedding, encoder
from modules.noise import GaussianNoise
import torch.nn.functional as F
from torchcrf import CRF


class ModelSbj(nn.Module):
    def __init__(self, param):
        super(ModelSbj, self).__init__()

        self.mode = param['mode']
        self.hidden_size = param['hidden_size']
        self.dropout_p = param['dropout_p']
        self.encoder_dropout_p = param['encoder_dropout_p']
        self.encoder_layer_num = param['encoder_layer_num']
        self.is_bn = False
        self.embedding = embedding.Embedding(param['embedding'])

        self.embedding_tag = nn.Embedding(
            num_embeddings=100,
            embedding_dim=4,
            padding_idx=0
        )
        self.gaussian_noise = GaussianNoise()

        # 语义编码
        self.encoder = encoder.Rnn(
            mode=self.mode,
            input_size=300,
            hidden_size=self.hidden_size,
            dropout_p=self.encoder_dropout_p,
            bidirectional=True,
            layer_num=self.encoder_layer_num,
            is_bn=self.is_bn
        )

        # sbj位置映射
        self.sbj = nn.Linear(self.hidden_size*2, 37)

        # crf
        self.crf = CRF(num_tags=37)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.sbj.weight)
        torch.nn.init.xavier_uniform_(self.embedding_tag.weight)

        torch.nn.init.constant_(self.sbj.bias, 0.0)

    def forward(self, batch, is_train=True):
        if is_train:
            text, tag, sbj = batch

            # 语义编码
            text_mask = torch.ne(text, 0)
            max_len = text_mask.sum(dim=1).max().item()
            text_mask = text_mask[:, :max_len]
            text = text[:, :max_len]
            text_emb = self.embedding(text)
            text_vec = self.encoder(text_emb, text_mask)

            # rnn_feat
            sbj_feat = self.sbj(text_vec)  # (seq_len, b, 37)
            sbj = sbj[:, :max_len].transpose(0, 1)  # (seq_len, b)
            text_mask = text_mask.transpose(0, 1)  # (seq_len, b)

            loss = -1 * self.crf(sbj_feat, sbj, mask=text_mask, reduction='token_mean')

            return loss
        else:
            text, tag = batch

            # 语义编码
            text_mask = torch.ne(text, 0)
            max_len = text_mask.sum(dim=1).max().item()
            text_mask = text_mask[:, :max_len]
            text = text[:, :max_len]
            text_emb = self.embedding(text)
            text_vec = self.encoder(text_emb, text_mask)

            # rnn_feat
            sbj_feat = self.sbj(text_vec)  # (seq_len, b, 37)
            text_mask = text_mask.transpose(0, 1)  # (seq_len, b)

            # decoder
            sbj = self.crf.decode(sbj_feat, mask=text_mask)

            return sbj
