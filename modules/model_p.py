# coding = utf-8
# author = xy

import torch
from torch import nn
from modules import embedding, encoder, model_ner
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

        # 载入模型
        model_sbj_ = model_ner.ModelNer(param)
        model_path = '../model/' + param['model_path_sbj'] + '.pkl'
        state = torch.load(model_path)
        model_sbj_.load_state_dict(state['model_state'])
        print('sbj, loss:%.4f, epoch:%2d, step:%4d, time:%4d' % (state['loss'], state['epoch'], state['steps'],
                                                                 state['time']))
        model_obj_ = model_ner.ModelNer(param)
        model_path = '../model/' + param['model_path_obj'] + '.pkl'
        state = torch.load(model_path)
        model_obj_.load_state_dict(state['model_state'])
        print('obj, loss:%.4f, epoch:%2d, step:%4d, time:%4d' % (state['loss'], state['epoch'], state['steps'],
                                                                 state['time']))

        # embedding
        self.embedding_sbj = model_sbj_.embedding
        self.embedding_obj = model_obj_.embedding

        # tag embedding
        self.embedding_tag_sbj = model_sbj_.embedding_tag
        self.gaussian_noise_sbj = model_sbj_.gaussian_noise
        self.embedding_tag_obj = model_obj_.embedding_tag
        self.gaussian_noise_obj = model_obj_.gaussian_noise

        # 语义编码
        self.encoder_sbj = model_sbj_.encoder
        self.encoder_obj = model_obj_.encoder

        # ner位置映射
        self.position_sbj = model_sbj_.ner
        self.crf_sbj = model_sbj_.crf
        self.position_obj = model_obj_.ner
        self.crf_obj = model_obj_.crf

        # p分类
        self.p_fc = nn.Linear(self.hidden_size*4, 49)

        self.drop = nn.Dropout(self.dropout_p)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.p_fc.weight)
        torch.nn.init.constant_(self.p_fc.bias, 0.0)

    def get_ner(self, ner):
        tmp = []
        if sum(ner) != 0:
            index = -9
            for i in range(len(ner)):
                if i <= index:
                    continue
                if ner[i] != 0:
                    index_s = i
                    for j in range(len(ner[i:])):
                        if ner[i:][j] == 0:
                            index_e = i + j - 1
                            break
                        if j == len(ner[i:]) - 1:
                            index_e = len(ner) - 1
                    tmp.append((index_s, index_e, ner[i]))
                    index = index_e
        return tmp

    def forward(self, batch, is_train=True):
        if is_train:
            text, sbj, obj, sbj_bounds, obj_bounds, ps = batch

            # 裁剪
            text_mask = torch.ne(text, 0)
            max_len = text_mask.sum(dim=1).max().item()
            text_mask = text_mask[:, :max_len]
            text = text[:, :max_len]
            sbj = sbj[:, :max_len]
            obj = obj[:, :max_len]

            # embedding
            text_emb_sbj = self.embedding_sbj(text).transpose(0, 1)
            text_emb_obj = self.embedding_obj(text).transpose(0, 1)

            # encoder
            text_vec_sbj = self.encoder_sbj(text_emb_sbj, text_mask)
            text_vec_obj = self.encoder_obj(text_emb_obj, text_mask)

            # sbj: nn + crf
            sbj_feat = self.position_sbj(text_vec_sbj)
            loss_sbj = -1 * self.crf_sbj(sbj_feat, sbj.transpose(0, 1), mask=text_mask.transpose(0, 1),
                                         reduction='token_mean')

            # obj: nn + crf
            obj_feat = self.position_obj(text_vec_obj)
            loss_obj = -1 * self.crf_obj(obj_feat, obj.transpose(0, 1), mask=text_mask.transpose(0, 1),
                                         reduction='token_mean')

            # 构造p训练数据
            text_vec_sbj = text_vec_sbj.transpose(0, 1)
            text_vec_obj = text_vec_obj.transpose(0, 1)
            batch_size = text.size(0)
            max_samples = sbj_bounds.size(1)
            x = []
            y = []
            for i in range(batch_size):
                vec_sbj = text_vec_sbj[i]
                vec_obj = text_vec_obj[i]
                sbj_bound = sbj_bounds[i]
                obj_bound = obj_bounds[i]
                p = ps[i]

                for j in range(max_samples):
                    if sbj_bound[j].sum() < 0:
                        break

                    vec_sbj_tmp = vec_sbj.index_select(dim=0, index=sbj_bound[j]).mean(dim=0)
                    vec_obj_tmp = vec_obj.index_select(dim=0, index=obj_bound[j]).mean(dim=0)
                    vec_tmp = torch.cat([vec_sbj_tmp, vec_obj_tmp]).reshape(1, -1)
                    x.append(vec_tmp)
                    y.append(p[j].reshape(1, -1))
            x = torch.cat(x, dim=0)  # (*, h*4)
            y = torch.cat(y, dim=0)  # (*, 49)

            # p
            x = self.p_fc(x)
            loss_p = F.binary_cross_entropy_with_logits(x, y.float())

            # loss
            loss = 0.2 * (loss_sbj + loss_obj) + loss_p

            return loss, loss_sbj, loss_obj, loss_p

        else:
            text, _ = batch

            # 裁剪
            text_mask = torch.ne(text, 0)
            max_len = text_mask.sum(dim=1).max().item()
            text_mask = text_mask[:, :max_len]
            text = text[:, :max_len]

            # embedding
            text_emb_sbj = self.embedding_sbj(text).transpose(0, 1)
            text_emb_obj = self.embedding_obj(text).transpose(0, 1)

            # encoder
            text_vec_sbj = self.encoder_sbj(text_emb_sbj, text_mask)
            text_vec_obj = self.encoder_obj(text_emb_obj, text_mask)

            # sbj: nn + crf
            sbj_feat = self.position_sbj(text_vec_sbj)
            sbj = self.crf_sbj.decode(sbj_feat, mask=text_mask.transpose(0, 1))  # (b, seq_len)

            # obj: nn + crf
            obj_feat = self.position_obj(text_vec_obj)
            obj = self.crf_obj.decode(obj_feat, mask=text_mask.transpose(0, 1))

            # sbj_bound
            sbj_bounds = []
            for sbj_item in sbj:
                bound = self.get_ner(sbj_item)
                sbj_bounds.append(bound)
            # obj_bound
            obj_bounds = []
            for obj_item in obj:
                bound = self.get_ner(obj_item)
                obj_bounds.append(bound)

            # p
            result_bound = []
            result_vec = []
            batch_size = len(sbj_bounds)
            text_vec_sbj = text_vec_sbj.transpose(0, 1)
            text_vec_obj = text_vec_obj.transpose(0, 1)
            for i in range(batch_size):
                vec_sbj = text_vec_sbj[i]
                vec_obj = text_vec_obj[i]
                sbj_bound = sbj_bounds[i]
                obj_bound = obj_bounds[i]
                r_tmp = []
                vec_batch = []
                for sbj_bound_item in sbj_bound:
                    for obj_bound_item in obj_bound:
                        vec_sbj_tmp = vec_sbj.index_select(dim=0, index=torch.LongTensor(sbj_bound_item[:2]).cuda()).mean(dim=0)
                        vec_obj_tmp = vec_obj.index_select(dim=0, index=torch.LongTensor(obj_bound_item[:2]).cuda()).mean(dim=0)
                        vec_tmp = torch.cat([vec_sbj_tmp, vec_obj_tmp]).reshape(1, -1)
                        vec_batch.append(vec_tmp)
                        r_tmp.append([sbj_bound_item, obj_bound_item])
                if vec_batch:
                    vec_batch = torch.cat(vec_batch, dim=0)
                    vec_batch = torch.sigmoid(self.p_fc(vec_batch)).detach().cpu().numpy()

                result_bound.append(r_tmp)
                result_vec.append(vec_batch)

            return result_bound, result_vec
