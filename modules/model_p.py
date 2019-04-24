# coding = utf-8
# author = xy

import pickle
import torch
from torch import nn
from modules import encoder
from modules.noise import GaussianNoise
import torch.nn.functional as F


class ModelP(nn.Module):
    def __init__(self, param):
        super(ModelP, self).__init__()

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
        self.p_fc = nn.Linear(self.hidden_size*4, 49)

        self.drop = nn.Dropout(self.dropout_p)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.p_fc.weight)
        torch.nn.init.constant_(self.p_fc.bias, 0.0)

    def forward(self, batch, is_train=True):
        if is_train:
            text, sbj_bounds, obj_bounds, ps = batch

            # 裁剪
            text_mask = torch.ne(text, 0)
            max_len = text_mask.sum(dim=1).max().item()
            text_mask = text_mask[:, :max_len]
            text = text[:, :max_len]

            # embedding
            text_emb = self.embedding(text).transpose(0, 1)

            # encoder
            text_vec = self.encoder(text_emb, text_mask)

            # 构造p训练数据
            text_vec = text_vec.transpose(0, 1)
            batch_size = text.size(0)
            max_samples = sbj_bounds.size(1)
            x = []
            y = []
            for i in range(batch_size):
                vec = text_vec[i]
                sbj_bound = sbj_bounds[i]
                obj_bound = obj_bounds[i]
                p = ps[i]
                # vec_mean = vec.mean(dim=0)

                for j in range(max_samples):
                    if sbj_bound[j].sum() < 0:
                        break
                    vec_sbj_tmp = vec.index_select(dim=0, index=sbj_bound[j]).mean(dim=0)
                    vec_obj_tmp = vec.index_select(dim=0, index=obj_bound[j]).mean(dim=0)
                    vec_tmp = torch.cat([vec_sbj_tmp, vec_obj_tmp]).reshape(1, -1)
                    x.append(vec_tmp)
                    y.append(p[j].reshape(1, -1))
            x = torch.cat(x, dim=0)  # (*, h*6)
            y = torch.cat(y, dim=0)  # (*, 49)

            # p
            x = self.p_fc(x)
            loss_p = F.binary_cross_entropy_with_logits(x, y.float())

            return loss_p

        else:
            text, sbj_bounds, obj_bounds = batch

            # 裁剪
            text_mask = torch.ne(text, 0)
            max_len = text_mask.sum(dim=1).max().item()
            text_mask = text_mask[:, :max_len]
            text = text[:, :max_len]

            # embedding
            text_emb = self.embedding(text).transpose(0, 1)

            # encoder
            text_vec = self.encoder(text_emb, text_mask)

            # p
            result_sbj_bounds = []
            result_obj_bounds = []
            result_vecs = []
            batch_size = len(text)
            max_samples = sbj_bounds.size(1)
            text_vec = text_vec.transpose(0, 1)
            for i in range(batch_size):
                vec = text_vec[i]
                sbj_bound = sbj_bounds[i]
                obj_bound = obj_bounds[i]
                # vec_mean = vec.mean(dim=0)

                result_sbj_bound = []
                result_obj_bound = []
                result_vec = []
                for j in range(max_samples):
                    if sbj_bound[j].sum() < 0:
                        break
                    vec_sbj_tmp = vec.index_select(dim=0, index=sbj_bound[j]).mean(dim=0)
                    vec_obj_tmp = vec.index_select(dim=0, index=obj_bound[j]).mean(dim=0)
                    vec_tmp = torch.cat([vec_sbj_tmp, vec_obj_tmp]).reshape(1, -1)

                    result_sbj_bound.append(sbj_bound[j].cpu().numpy().tolist())
                    result_obj_bound.append(obj_bound[j].cpu().numpy().tolist())
                    result_vec.append(vec_tmp)

                result_vec = torch.cat(result_vec, dim=0)
                result_vec = torch.sigmoid(self.p_fc(result_vec)).detach().cpu().numpy()

                result_sbj_bounds.append(result_sbj_bound)
                result_obj_bounds.append(result_obj_bound)
                result_vecs.append(result_vec)

            return result_sbj_bounds, result_obj_bounds, result_vecs
