# coding = utf-8
# author = xy

import torch
from torch import nn
from modules import embedding, encoder
import torch.nn.functional as F
import loader


class ModelBase(nn.Module):
    def __init__(self, param):
        super(ModelBase, self).__init__()

        self.mode = param['mode']
        self.hidden_size = param['hidden_size']
        self.dropout_p = param['dropout_p']
        self.encoder_dropout_p = param['encoder_dropout_p']
        self.encoder_layer_num = param['encoder_layer_num']
        self.is_bn = False
        self.embedding = embedding.Embedding(param['embedding'])

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
        self.sbj_start_fc = nn.Linear(self.hidden_size*2, 1)
        self.sbj_end_fc = nn.Linear(self.hidden_size*2, 1)

        # obj位置映射
        self.obj_start_fc = nn.Linear(self.hidden_size*4, 50)
        self.obj_end_fc = nn.Linear(self.hidden_size*4, 50)

        self.drop = nn.Dropout(self.dropout_p)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.sbj_start_fc.weight)
        torch.nn.init.xavier_uniform_(self.sbj_end_fc.weight)
        torch.nn.init.xavier_uniform_(self.obj_start_fc.weight)
        torch.nn.init.xavier_uniform_(self.obj_end_fc.weight)

        torch.nn.init.constant_(self.sbj_start_fc.bias, 0.0)
        torch.nn.init.constant_(self.sbj_end_fc.bias, 0.0)
        torch.nn.init.constant_(self.obj_start_fc.bias, 0.0)
        torch.nn.init.constant_(self.obj_end_fc.bias, 0.0)

    def forward(self, batch, is_train=True, test_value=0.5):
        if is_train:
            text, sbj_start, sbj_end, sbj_bound, obj_start, obj_end = batch
            batch_size = text.size(0)

            # 语义编码
            text_mask = torch.ne(text, 0)
            max_len = text_mask.sum(dim=1).max().item()
            text_mask = text_mask[:, :max_len]
            text = text[:, :max_len]
            text_emb = self.embedding(text)
            text_vec = self.encoder(text_emb, text_mask)

            # 位置压缩
            sbj_start = sbj_start[:, :max_len]
            sbj_end = sbj_end[:, :max_len]
            obj_start = obj_start[:, :max_len]
            obj_end = obj_end[:, :max_len]

            # sbj位置映射
            s1 = torch.sigmoid(self.sbj_start_fc(text_vec)).squeeze().transpose(0, 1)  # (b, seq_len)
            s2 = torch.sigmoid(self.sbj_end_fc(text_vec)).squeeze().transpose(0, 1)

            # text + sbj
            text_vec = text_vec.transpose(0, 1)
            sbj_vec = torch.zeros(batch_size, self.hidden_size*2).cuda()
            for i in range(batch_size):
                tmp = torch.arange(sbj_bound[i][0].item(), sbj_bound[i][1].item()+1).cuda()
                tmp = text_vec[i].index_select(dim=0, index=tmp)
                sbj_vec[i] = tmp.mean(dim=0)
            sbj_vec = sbj_vec.unsqueeze(1).expand(text_vec.size())
            text_sbj_vec = torch.cat([text_vec, sbj_vec], dim=2)

            # obj位置映射
            o1 = self.obj_start_fc(text_sbj_vec)  # (b, seq_len, class_num)
            o2 = self.obj_end_fc(text_sbj_vec)

            # loss
            text_mask = text_mask.float()
            value_num = text_mask.sum().item()
            loss_sbj_s = F.binary_cross_entropy(s1, sbj_start.float(), reduction='none')
            loss_sbj_s = (loss_sbj_s * text_mask).sum() / value_num

            loss_sbj_e = F.binary_cross_entropy(s2, sbj_end.float(), reduction='none')
            loss_sbj_e = (loss_sbj_e * text_mask).sum() / value_num

            text_mask = text_mask.reshape(-1)
            loss_obj_s = F.cross_entropy(o1.reshape(-1, 50), obj_start.reshape(-1), reduction='none')
            loss_obj_s = (loss_obj_s * text_mask).sum() / value_num

            loss_obj_e = F.cross_entropy(o2.reshape(-1, 50), obj_end.reshape(-1), reduction='none')
            loss_obj_e = (loss_obj_e * text_mask).sum() / value_num

            loss = 2 * (loss_sbj_s + loss_sbj_e) + loss_obj_s + loss_obj_e

            return loss

        else:
            text = batch
            batch_size = text.size(0)

            # 语义编码
            text_mask = torch.ne(text, 0)
            max_len = text_mask.sum(dim=1).max().item()
            text_mask = text_mask[:, :max_len]
            text = text[:, :max_len]
            text_emb = self.embedding(text)
            text_vec = self.encoder(text_emb, text_mask)

            # sbj位置映射
            s1 = torch.sigmoid(self.sbj_start_fc(text_vec)).squeeze().transpose(0, 1)  # (b, seq_len)
            s2 = torch.sigmoid(self.sbj_end_fc(text_vec)).squeeze().transpose(0, 1)

            text_vec = text_vec.transpose(0, 1)
            R = []
            for i in range(batch_size):
                R_tmp = []
                sjb_bounds = []
                for j, s1_item in enumerate(s1[i]):
                    if s1_item >= test_value:
                        for k, s2_item in enumerate(s2[i][j:]):
                            if s2_item >= test_value:
                                sjb_bounds.append([j, j+k])
                                break
                if sjb_bounds:
                    sjb_vec = torch.zeros(len(sjb_bounds), self.hidden_size*2).cuda()
                    for m, sjb_bound in enumerate(sjb_bounds):
                        tmp = torch.arange(sjb_bound[0], sjb_bound[1]+1).cuda()
                        tmp = text_vec[i].index_select(dim=0, index=tmp)
                        sjb_vec[m] = tmp.mean(dim=0)

                    text_vec_i = text_vec[i].unsqueeze(0).expand(len(sjb_bounds), text_vec.size(1), text_vec.size(2))
                    sjb_vec = sjb_vec.unsqueeze(1).expand(len(sjb_bounds), text_vec.size(1), text_vec.size(2))
                    text_sjb_vec = torch.cat([text_vec_i, sjb_vec], dim=2)

                    o1_i = self.obj_start_fc(text_sjb_vec)
                    o1_i = torch.argmax(o1_i, dim=2)
                    o2_i = self.obj_end_fc(text_sjb_vec)
                    o2_i = torch.argmax(o2_i, dim=2)

                    for m in range(len(sjb_bounds)):
                        for n, o1_item in enumerate(o1_i[m]):
                            if o1_item > 0:
                                for nn, o2_item in enumerate(o2_i[m][n:]):
                                    if o1_item == o2_item:
                                        sbj = [sjb_bounds[m][0], sjb_bounds[m][1]]
                                        obj = [n, n+nn]
                                        p = o1_item.item()
                                        R_tmp.append([sbj, p, obj])
                                        break
                R.append(R_tmp)
            return R
