# coding = utf-8
# author = xy

import torch
from torch import nn
from modules import embedding, encoder
from modules.noise import GaussianNoise
import torch.nn.functional as F
from torchcrf import CRF
import pickle


class ModelNer(nn.Module):
    def __init__(self, param):
        super(ModelNer, self).__init__()

        self.mode = param['mode']
        self.hidden_size = param['hidden_size']
        self.dropout_p = param['dropout_p']
        self.encoder_dropout_p = param['encoder_dropout_p']
        self.encoder_layer_num = param['encoder_layer_num']
        self.is_bn = False

        with open('../data/char_dict.pkl', 'rb') as f:
            char2i = pickle.load(f)['char2i']
        self.embedding = nn.Embedding(num_embeddings=len(char2i), embedding_dim=256, padding_idx=0)

        self.embedding_tag = nn.Embedding(
            num_embeddings=100,
            embedding_dim=4,
            padding_idx=0
        )
        self.gaussian_noise = GaussianNoise()

        # 语义编码
        self.encoder = encoder.Rnn(
            mode=self.mode,
            input_size=256,
            hidden_size=self.hidden_size,
            dropout_p=self.encoder_dropout_p,
            bidirectional=True,
            layer_num=self.encoder_layer_num,
            is_bn=self.is_bn
        )

        # p分类
        self.p_fc = nn.Linear(self.hidden_size*2, 49)

        # ner位置映射
        self.ner = nn.Linear(self.hidden_size*2+49, 5)

        # crf
        self.crf = CRF(num_tags=5)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.ner.weight)

        torch.nn.init.constant_(self.ner.bias, 0.0)

    def forward(self, batch, is_train=True):
        if is_train:
            text, ner, p = batch

            # 语义编码
            text_mask = torch.ne(text, 0)
            max_len = text_mask.sum(dim=1).max().item()
            text_mask = text_mask[:, :max_len]
            text = text[:, :max_len]
            text_emb = self.embedding(text).transpose(0, 1)
            text_vec = self.encoder(text_emb, text_mask)

            # p
            p_vec = text_vec.mean(dim=0)
            p_vec = torch.sigmoid(self.p_fc(p_vec))
            loss_p = F.binary_cross_entropy(p_vec, p.float())

            # rnn_feat
            p_vec = p_vec.unsqueeze(0).expand(text_vec.size(0), text_vec.size(1), 49)
            ner_feat = torch.cat([text_vec, p_vec], dim=-1)
            ner_feat = self.ner(ner_feat)  # (seq_len, b, 5)
            ner = ner[:, :max_len].transpose(0, 1)  # (seq_len, b)
            text_mask = text_mask.transpose(0, 1)  # (seq_len, b)

            loss_ner = -1 * self.crf(ner_feat, ner, mask=text_mask, reduction='token_mean')

            loss = loss_p * 0.2 + loss_ner

            return loss
        else:
            text, _ = batch

            # 语义编码
            text_mask = torch.ne(text, 0)
            max_len = text_mask.sum(dim=1).max().item()
            text_mask = text_mask[:, :max_len]
            text = text[:, :max_len]
            text_emb = self.embedding(text).transpose(0, 1)
            text_vec = self.encoder(text_emb, text_mask)

            # p
            p_vec = text_vec.mean(dim=0)
            p_vec = torch.sigmoid(self.p_fc(p_vec))

            # rnn_feat
            p_vec = p_vec.unsqueeze(0).expand(text_vec.size(0), text_vec.size(1), 49)
            ner_feat = torch.cat([text_vec, p_vec], dim=-1)
            ner_feat = self.ner(ner_feat)  # (seq_len, b, 5)
            text_mask = text_mask.transpose(0, 1)  # (seq_len, b)

            # decoder
            ner = self.crf.decode(ner_feat, mask=text_mask)

            return ner
