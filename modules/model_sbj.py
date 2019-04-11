# coding = utf-8
# author = xy

import torch
from torch import nn
from modules import embedding, encoder
import torch.nn.functional as F


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
            embedding_dim=8,
            padding_idx=0
        )

        # 语义编码
        self.encoder = encoder.Rnn(
            mode=self.mode,
            input_size=300 + 8,
            hidden_size=self.hidden_size,
            dropout_p=self.encoder_dropout_p,
            bidirectional=True,
            layer_num=self.encoder_layer_num,
            is_bn=self.is_bn
        )

        # sbj位置映射
        self.sbj_start_fc = nn.Linear(self.hidden_size*2, 1)
        self.sbj_end_fc = nn.Linear(self.hidden_size*2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.sbj_start_fc.weight)
        torch.nn.init.xavier_uniform_(self.sbj_end_fc.weight)

        torch.nn.init.constant_(self.sbj_start_fc.bias, 0.0)
        torch.nn.init.constant_(self.sbj_end_fc.bias, 0.0)

    def forward(self, batch, is_train=True):
        if is_train:
            text, tag, sbj_start, sbj_end = batch

            # 语义编码
            text_mask = torch.ne(text, 0)
            max_len = text_mask.sum(dim=1).max().item()
            text_mask = text_mask[:, :max_len]
            text = text[:, :max_len]
            tag = tag[:, : max_len]
            text_emb = self.embedding(text)
            tag_emb = self.embedding_tag(tag).transpose(0, 1)
            vec = torch.cat([text_emb, tag_emb], dim=2)

            text_vec = self.encoder(vec, text_mask)

            # 位置压缩
            sbj_start = sbj_start[:, :max_len]
            sbj_end = sbj_end[:, :max_len]

            # sbj位置映射
            s1 = torch.sigmoid(self.sbj_start_fc(text_vec)).squeeze().transpose(0, 1)  # (b, seq_len)
            s2 = torch.sigmoid(self.sbj_end_fc(text_vec)).squeeze().transpose(0, 1)

            # loss
            text_mask = text_mask.float()
            value_num = text_mask.sum().item()
            loss_sbj_s = F.binary_cross_entropy(s1, sbj_start.float(), reduction='none')
            loss_sbj_s = (loss_sbj_s * text_mask).sum() / value_num
            loss_sbj_e = F.binary_cross_entropy(s2, sbj_end.float(), reduction='none')
            loss_sbj_e = (loss_sbj_e * text_mask).sum() / value_num
            loss = loss_sbj_s + loss_sbj_e

            return loss
        else:
            text, tag = batch

            # 语义编码
            text_mask = torch.ne(text, 0)
            max_len = text_mask.sum(dim=1).max().item()
            text_mask = text_mask[:, :max_len]
            text = text[:, :max_len]
            tag = tag[:, :max_len]
            text_emb = self.embedding(text)
            tag_emb = self.embedding_tag(tag).transpose(0, 1)
            vec = torch.cat([text_emb, tag_emb], dim=2)
            text_vec = self.encoder(vec, text_mask)

            # sbj位置映射
            s1 = torch.sigmoid(self.sbj_start_fc(text_vec)).squeeze().transpose(0, 1)  # (b, seq_len)
            s2 = torch.sigmoid(self.sbj_end_fc(text_vec)).squeeze().transpose(0, 1)

            s1 = s1 * text_mask.float()
            s2 = s2 * text_mask.float()

            return s1, s2
