# coding = utf-8
# author = xy

import torch
from torch import nn
from modules import embedding, encoder, model_sbj
from modules.noise import GaussianNoise
import torch.nn.functional as F


class ModelSpo(nn.Module):
    def __init__(self, param):
        super(ModelSpo, self).__init__()

        self.mode = param['mode']
        self.hidden_size = param['hidden_size']
        self.dropout_p = param['dropout_p']
        self.encoder_dropout_p = param['encoder_dropout_p']
        self.encoder_layer_num = param['encoder_layer_num']
        self.is_bn = False
        self.embedding = embedding.Embedding(param['embedding'])

        model_sbj_ = model_sbj.ModelSbj(param)
        model_path = '../model/' + param['model_path_sbj'] + '.pkl'
        state = torch.load(model_path)
        model_sbj_.load_state_dict(state['model_state'])
        print('load model_sbj, loss:%.4f, epoch:%2d, step:%4d, time:%4d' % (state['loss'], state['epoch'],
                                                                            state['steps'], state['time']))
        # tag embedding
        self.embedding_tag = model_sbj_.embedding_tag
        self.gaussian_noise = model_sbj_.gaussian_noise

        # 语义编码
        self.encoder = model_sbj_.encoder

        # sbj位置映射
        self.sbj = model_sbj_.sbj
        self.crf = model_sbj_.crf

        # obj位置映射
        self.obj_start_fc = nn.Linear(self.hidden_size*4, 50)
        self.obj_end_fc = nn.Linear(self.hidden_size*4, 50)

        self.drop = nn.Dropout(self.dropout_p)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.obj_start_fc.weight)
        torch.nn.init.xavier_uniform_(self.obj_end_fc.weight)

        torch.nn.init.constant_(self.obj_start_fc.bias, 0.0)
        torch.nn.init.constant_(self.obj_end_fc.bias, 0.0)

    def forward(self, batch, is_train=True, test_value=0.5):
        if is_train:
            text, tag, sbj, sbj_bound, obj_start, obj_end = batch
            batch_size = text.size(0)

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
            sbj_mask = text_mask.transpose(0, 1)  # (seq_len, b)

            # loss_sbj
            loss_sbj = -1 * self.crf(sbj_feat, sbj, mask=sbj_mask, reduction='token_mean')

            # text + sbj
            obj_start = obj_start[:, :max_len]
            obj_end = obj_end[:, :max_len]
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

            # loss_spo
            text_mask = text_mask.float()
            value_num = text_mask.sum().item()
            text_mask = text_mask.reshape(-1)
            loss_obj_s = F.cross_entropy(o1.reshape(-1, 50), obj_start.reshape(-1), reduction='none')

            loss_obj_s = (loss_obj_s * text_mask).sum() / value_num

            loss_obj_e = F.cross_entropy(o2.reshape(-1, 50), obj_end.reshape(-1), reduction='none')
            loss_obj_e = (loss_obj_e * text_mask).sum() / value_num

            loss = 0.1 * loss_sbj + loss_obj_s + loss_obj_e

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
            sbj_mask = text_mask.transpose(0, 1)  # (seq_len, b)

            # decoder_sbj
            sbj = self.crf.decode(sbj_feat, mask=sbj_mask)

            text_vec = text_vec.transpose(0, 1).squeeze()  # (seq_len, h*2)
            R = []
            index = -9
            for i in range(len(sbj)):
                sbj_ = ''
                if i <= index:
                    continue
                if sbj[i] != 0:
                    index_s = i
                    for j in range(len(sbj[i:])):
                        if sbj[i:][j] == 0:
                            index_e = i + j - 1
                            break
                        if j == len(sbj[i:]) - 1:
                            index_e = len(sbj) - 1
                    sbj_ = [index_s, index_e]
                    index = index_e

                if sbj_:
                    sbj_range = torch.arange(sbj_[0], sbj_[1]+1).cuda()
                    sbj_vec = text_vec.index_select(dim=0, index=sbj_range)
                    sbj_vec = sbj_vec.mean(dim=0)
                    text_sbj_vec = torch.cat([text_vec, sbj_vec.expand(text_vec.size())], dim=1)

                    o1_i = self.obj_start_fc(text_sbj_vec)
                    o1_i = torch.argmax(o1_i, dim=1)
                    o2_i = self.obj_end_fc(text_sbj_vec)
                    o2_i = torch.argmax(o2_i, dim=1)

                    for j, o1_item in enumerate(o1_i):
                        if o1_item > 0:
                            for m, o2_item in enumerate(o2_i[j:]):
                                if o1_item == o2_item:
                                    R.append([sbj_, o1_item.item(), [j, j+m]])
                                    break

            return R
