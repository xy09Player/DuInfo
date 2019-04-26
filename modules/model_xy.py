# coding = utf-8
# author = xy

import torch
from torch import nn
from modules import embedding, encoder
from modules.noise import GaussianNoise
import torch.nn.functional as F
from torchcrf import CRF
import pickle
import numpy as np


class ModelXy(nn.Module):
    def __init__(self, param):
        super(ModelXy, self).__init__()

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

        # sbj 位置映射 + crf
        self.sbj_position = nn.Linear(self.hidden_size*2, 5)
        self.sbj_crf = CRF(num_tags=5)

        # obj 位置映射 + crf
        self.obj_position = nn.Linear(self.hidden_size*2, 5)
        self.obj_crf = CRF(num_tags=5)

        # p分类
        self.p_fc = nn.Linear(self.hidden_size*4, 49)

        self.drop = nn.Dropout(self.dropout_p)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.sbj_position.weight)
        torch.nn.init.xavier_uniform_(self.obj_position.weight)
        torch.nn.init.xavier_uniform_(self.p_fc.weight)

        torch.nn.init.constant_(self.sbj_position.bias, 0.0)
        torch.nn.init.constant_(self.obj_position.bias, 0.0)
        torch.nn.init.constant_(self.p_fc.bias, 0.0)

    def forward(self, batch, is_train, have_p):
        if is_train:
            text, sbjs, objs, sbj_bounds, obj_bounds, ps = batch

            # 裁剪
            text_mask = torch.ne(text, 0)
            max_len = text_mask.sum(dim=1).max().item()
            text_mask = text_mask[:, :max_len]
            text = text[:, :max_len]
            sbjs = sbjs[:, :max_len]
            objs = objs[:, :max_len]

            # 编码
            text_emb = self.embedding(text).transpose(0, 1)
            text_vec = self.encoder(text_emb, text_mask)

            # sbj
            sbj_feat = self.sbj_position(text_vec)
            sbjs = sbjs.transpose(0, 1)
            sbj_mask = text_mask.transpose(0, 1)
            loss_sbj = -1 * self.sbj_crf(sbj_feat, sbjs, mask=sbj_mask, reduction='token_mean')

            # obj
            obj_feat = self.obj_position(text_vec)
            objs = objs.transpose(0, 1)
            obj_mask = text_mask.transpose(0, 1)
            loss_obj = -1 * self.obj_crf(obj_feat, objs, mask=obj_mask, reduction='token_mean')

            if have_p:
                # p索引字典
                p_dict = []
                batch_size = text_vec.size(1)
                sbj_bounds = sbj_bounds.cpu().numpy()
                obj_bounds = obj_bounds.cpu().numpy()
                ps = ps.cpu().numpy()
                for i in range(batch_size):
                    p_dict_tmp = {}
                    sbj_bound = sbj_bounds[i].tolist()
                    obj_bound = obj_bounds[i].tolist()
                    p = ps[i].tolist()
                    for s_bound, o_bound, ppp in zip(sbj_bound, obj_bound, p):
                        if sum(s_bound) < 0:
                            break
                        s_bound = tuple(s_bound)
                        o_bound = tuple(o_bound)
                        p_dict_tmp[(s_bound, o_bound)] = ppp
                    p_dict.append(p_dict_tmp)

                # 构造p训练数据
                sbj_gen = self.sbj_crf.decode(sbj_feat, mask=sbj_mask)
                obj_gen = self.obj_crf.decode(obj_feat, mask=obj_mask)
                text_vec = text_vec.transpose(0, 1)

                vec_tmp = []
                sbj_tmp = []
                obj_tmp = []
                p_dict_tmp = []
                for ith in range(batch_size):
                    vec_item = text_vec[ith]
                    sbj_item = sbj_gen[ith]
                    obj_item = obj_gen[ith]
                    p_dict_item = p_dict[ith]

                    if sum(sbj_item) != 0 and sum(obj_item) != 0:
                        vec_tmp.append(vec_item)

                        index = -9
                        tmp = []
                        for i in range(len(sbj_item)):
                            if i <= index:
                                continue
                            if sbj_item[i] != 0:
                                index_s = i
                                for j in range(len(sbj_item[i:])):
                                    if sbj_item[i:][j] == 0:
                                        index_e = i + j - 1
                                        break
                                    if j == len(sbj_item[i:]) - 1:
                                        index_e = len(sbj_item) - 1
                                tmp.append((index_s, index_e))
                                index = index_e
                        sbj_tmp.append(tmp)

                        index = -9
                        tmp = []
                        for i in range(len(obj_item)):
                            if i <= index:
                                continue
                            if obj_item[i] != 0:
                                index_s = i
                                for j in range(len(obj_item[i:])):
                                    if obj_item[i:][j] == 0:
                                        index_e = i + j - 1
                                        break
                                    if j == len(obj_item[i:]) - 1:
                                        index_e = len(obj_item) - 1
                                tmp.append((index_s, index_e))
                                index = index_e
                        obj_tmp.append(tmp)

                        p_dict_tmp.append(p_dict_item)

                samples = len(vec_tmp)
                x = []
                y = []
                p_0 = [0 for _ in range(49)]
                for ith in range(samples):
                    vec_item = vec_tmp[ith]
                    sbj_item = sbj_tmp[ith]
                    obj_item = obj_tmp[ith]
                    p_dict_item = p_dict_tmp[ith]

                    for sbj_index in sbj_item:
                        for obj_index in obj_item:
                            vec_sbj_tmp = vec_item.index_select(dim=0, index=torch.LongTensor(sbj_index).cuda()).mean(dim=0)
                            vec_obj_tmp = vec_item.index_select(dim=0, index=torch.LongTensor(obj_index).cuda()).mean(dim=0)
                            vec_feat = torch.cat([vec_sbj_tmp, vec_obj_tmp]).reshape(1, -1)
                            x.append(vec_feat)
                            if (sbj_index, obj_index) in p_dict_item:
                                y.append(p_dict_item[(sbj_index, obj_index)])
                            else:
                                y.append(p_0)
                if x:
                    x = torch.cat(x, dim=0)  # (*, h*4)
                    y = torch.FloatTensor(y).cuda()  # (*, 49)

                    # p
                    x = self.p_fc(x)
                    loss_p = F.binary_cross_entropy_with_logits(x, y)
                else:
                    loss_p = torch.Tensor([0]).cuda()

                loss = 2 * (loss_sbj + loss_obj) + loss_p
                return loss, loss_sbj, loss_obj, loss_p

            else:
                loss = 2 * (loss_sbj + loss_obj)
                return loss, loss_sbj, loss_obj, torch.Tensor([0]).cuda()
        else:
            text, _ = batch

            # 裁剪
            text_mask = torch.ne(text, 0)
            max_len = text_mask.sum(dim=1).max().item()
            text_mask = text_mask[:, :max_len]
            text = text[:, :max_len]

            # 编码
            text_emb = self.embedding(text).transpose(0, 1)
            text_vec = self.encoder(text_emb, text_mask)

            # sbj
            sbj_feat = self.sbj_position(text_vec)
            sbj_mask = text_mask.transpose(0, 1)
            sbjs = self.sbj_crf.decode(sbj_feat, mask=sbj_mask)

            # obj
            obj_feat = self.obj_position(text_vec)
            obj_mask = text_mask.transpose(0, 1)
            objs = self.obj_crf.decode(obj_feat, mask=obj_mask)

            # p
            text_vec = text_vec.transpose(0, 1)
            batch_size = text_vec.size(0)
            r_sbj_bounds = []
            r_obj_bounds = []
            r_ps = []
            for ith in range(batch_size):
                vec_item = text_vec[ith]
                sbj_item = sbjs[ith]
                obj_item = objs[ith]

                sbj_bound = []
                obj_bound = []
                ps = []
                if sum(sbj_item) == 0 or sum(obj_item) == 0:
                    sbj_bound.append((-1, -1))
                    obj_bound.append((-1, -1))
                    ps = np.array([[0 for _ in range(49)]])
                else:
                    index = -9
                    sbjs_tmp = []
                    for i in range(len(sbj_item)):
                        if i <= index:
                            continue
                        if sbj_item[i] != 0:
                            index_s = i
                            for j in range(len(sbj_item[i:])):
                                if sbj_item[i:][j] == 0:
                                    index_e = i + j - 1
                                    break
                                if j == len(sbj_item[i:]) - 1:
                                    index_e = len(sbj_item) - 1
                            sbjs_tmp.append((index_s, index_e))
                            index = index_e

                    index = -9
                    objs_tmp = []
                    for i in range(len(obj_item)):
                        if i <= index:
                            continue
                        if obj_item[i] != 0:
                            index_s = i
                            for j in range(len(obj_item[i:])):
                                if obj_item[i:][j] == 0:
                                    index_e = i + j - 1
                                    break
                                if j == len(obj_item[i:]) - 1:
                                    index_e = len(obj_item) - 1
                            objs_tmp.append((index_s, index_e))
                            index = index_e

                    x = []
                    for sbj_index in sbjs_tmp:
                        for obj_index in objs_tmp:
                            sbj_bound.append(sbj_index)
                            obj_bound.append(obj_index)
                            vec_sbj_tmp = vec_item.index_select(dim=0, index=torch.LongTensor(sbj_index).cuda()).mean(dim=0)
                            vec_obj_tmp = vec_item.index_select(dim=0, index=torch.LongTensor(obj_index).cuda()).mean(dim=0)
                            vec_feat = torch.cat([vec_sbj_tmp, vec_obj_tmp]).reshape(1, -1)
                            x.append(vec_feat)
                    x = torch.cat(x, dim=0)
                    x = self.p_fc(x)
                    ps = torch.sigmoid(x).detach().cpu().numpy()

                r_sbj_bounds.append(sbj_bound)
                r_obj_bounds.append(obj_bound)
                r_ps.append(ps)

            return r_sbj_bounds, r_obj_bounds, r_ps
