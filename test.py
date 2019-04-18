# encoding = utf-8
# author = xy


import time
import numpy as np
import loader
import torch
import json
import pickle
from tqdm import tqdm
from config import config_ner
from config import config_p
from modules.model_ner import ModelNer
from modules.model_p import ModelP


def deal_p_r(result_bound, result_vec, test_value):
    result = []
    for r, vec in zip(result_bound, result_vec):
        for i in range(len(vec)):
            if vec[i] >= test_value:
                result.append([r[0], r[1], i])
    return result


def test(flag='val', task=None, test_value=0.5, config=None):
    time_start = time.time()
    embedding = np.load('../data/embedding.pkl.npy')
    data_loader = loader.build_loader(
        file_path='../data/dev_data.json' if flag == 'val' else '../data/test1_data_postag.json',
        batch_size=config.test_batch_size,
        shuffle=False,
        drop_last=False,
        is_train=False,
        task=task
    )
    param = {
        'embedding': embedding,
        'mode': config.mode,
        'hidden_size': config.hidden_size,
        'dropout_p': config.dropout_p,
        'encoder_dropout_p': 0.1,
        'encoder_layer_num': config.encoder_layer_num,
        'model_path_sbj': config.model_path_sbj if config.name in ['ModelP'] else 'xxx',
        'model_path_obj': config.model_path_obj if config.name in ['ModelP'] else 'xxx'
    }
    model = eval(config.name)(param)
    model.cuda()
    model.eval()
    model_path = '../model/' + config.model_path + '.pkl'
    state = torch.load(model_path)
    model.load_state_dict(state['model_state'])
    print('load model, loss:%.4f, epoch:%2d, step:%4d, time:%4d' % (state['loss'], state['epoch'],
                                                                    state['steps'], state['time']))
    with open('../data/p_dict.pkl', 'rb') as f:
        i2p = pickle.load(f)['i2p']

    R = []
    R_bound = []
    R_vec = []
    for batch in tqdm(data_loader):
        batch = [b.cuda() for b in batch]
        if task != 'p':
            ners = model(batch, is_train=False)
            for ner in ners:
                tmp = []
                if sum(ner) == 0:
                    tmp.append((0, 0, 0))
                else:
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
                R.append(tmp)
        else:
            r_bound, r_vec = model(batch, is_train=False)
            R_bound += r_bound
            R_vec += r_vec

    if flag == 'val' and task != 'p':
        text_lists, _, result = loader.gen_test_data('../data/dev_data.json', get_answer=True, task=task)
        A, B, C = 1e-10, 1e-10, 1e-10
        for i in range(len(text_lists)):
            t = result[i]
            text = text_lists[i]
            r = set([''.join(text[r_i[0]: r_i[1]+1]) for r_i in R[i]])
            if r != t:
                print(t)
                print(r)
                print('')
            A += len(r & t)
            B += len(r)
            C += len(t)
        f1 = A * 2 / (B + C)
        precision = A / B
        recall = A / C
        print('%s model, f1:%.4f, precision:%.4f, recall:%.4f\n' % (task, f1, precision, recall))

    if flag == 'val' and task == 'p':
        text_lists, _, result = loader.gen_test_data('../data/dev_data.json', get_answer=True, task=task)
        A, B, C = 1e-10, 1e-10, 1e-10
        for i in range(len(text_lists)):
            t = set(result[i])
            text = text_lists[i]
            R_i_tmp = deal_p_r(R_bound[i], R_vec[i], test_value=test_value)
            r = set()
            for r_i in R_i_tmp:
                sbj = ''.join(text[r_i[0][0]: r_i[0][1]+1])
                obj = ''.join(text[r_i[1][0]: r_i[1][1]+1])
                p = i2p[r_i[2]]
                r.add((sbj, p, obj))

            p_set = set([r_i[1] for r_i in r])
            if '妻子' in p_set and '丈夫' not in p_set:
                r_tmp = set([])
                for r_i in r:
                    r_tmp.add(r_i)
                    if r_i[1] == '妻子':
                        r_tmp.add((r_i[2], '丈夫', r_i[0]))
                r = r_tmp
            if '妻子' not in p_set and '丈夫' in p_set:
                r_tmp = set([])
                for r_i in r:
                    r_tmp.add(r_i)
                    if r_i[1] == '丈夫':
                        r_tmp.add((r_i[2], '妻子', r_i[0]))
                r = r_tmp

            A += len(r & t)
            B += len(r)
            C += len(t)
        f1 = A * 2 / (B + C)
        precision = A / B
        recall = A / C
        print('spo model, f1:%.4f, precision:%.4f, recall:%.4f\n' % (f1, precision, recall))

    if flag == 'test':
        text_lists, _, texts = loader.gen_test_data('../data/test1_data_postag.json', get_answer=False, task='p')
        result_writer = open('../result/result.json', 'w')
        for i in range(len(text_lists)):
            text = text_lists[i]
            R_i_tmp = deal_p_r(R_bound[i], R_vec[i], test_value=test_value)
            r = set()
            for r_i in R_i_tmp:
                sbj = ''.join(text[r_i[0][0]: r_i[0][1]+1])
                obj = ''.join(text[r_i[1][0]: r_i[1][1]+1])
                p = i2p[r_i[2]]
                r.add((sbj, p, obj))

            # p_set = set([r_i[1] for r_i in r])
            # if '妻子' in p_set and '丈夫' not in p_set:
            #     r_tmp = set([])
            #     for r_i in r:
            #         r_tmp.add(r_i)
            #         if r_i[1] == '妻子':
            #             r_tmp.add((r_i[2], '丈夫', r_i[0]))
            #     r = r_tmp
            # if '妻子' not in p_set and '丈夫' in p_set:
            #     r_tmp = set([])
            #     for r_i in r:
            #         r_tmp.add(r_i)
            #         if r_i[1] == '丈夫':
            #             r_tmp.add((r_i[2], '妻子', r_i[0]))
            #     r = r_tmp

            text_dic = {"text": texts[i]}
            if len(r) == 0:
                spo_list = []
            else:
                spo_list = [{'object_type': 'xx', 'predicate': p, 'object': obj, 'subject_type': 'xx', 'subject': sbj}
                            for sbj, p, obj in r]
            text_dic["spo_list"] = spo_list
            result_writer.write(json.dumps(text_dic, ensure_ascii=False))
            result_writer.write('\n')
        result_writer.close()

    if flag == 'val':
        return f1, precision, recall

    print(f'time:{time.time()-time_start}')


# def test_ensemble(flag='val', is_sbj=True, test_value=0.5, config=None, model_paths=None):
#     time_start = time.time()
#     embedding = np.load('../data/embedding.pkl.npy')
#     data_loader = loader.build_loader(
#         file_path='../data/dev_data.json' if flag == 'val' else '../data/test1_data_postag.json',
#         batch_size=config.test_batch_size if is_sbj else 1,
#         shuffle=False,
#         drop_last=False,
#         is_train=False,
#         is_sbj=is_sbj
#     )
#     param = {
#         'embedding': embedding,
#         'mode': config.mode,
#         'hidden_size': config.hidden_size,
#         'dropout_p': config.dropout_p,
#         'encoder_dropout_p': 0.1,
#         'encoder_layer_num': config.encoder_layer_num,
#         'model_path_sbj': config.model_path_sbj if config.name in ['ModelSpo'] else 'xxx'
#     }
#
#     with open('../data/schemas.pkl', 'rb') as f:
#         i2s = pickle.load(f)['i2s']
#
#     model = eval(config.name)(param)
#     model.cuda()
#     model.eval()
#     R = [[] for _ in range(len(model_paths))]
#     for model_num, model_path in enumerate(model_paths):
#         model_path = '../model/' + model_path + '.pkl'
#         state = torch.load(model_path)
#         model.load_state_dict(state['model_state'])
#         print('load model, loss:%.4f, epoch:%2d, step:%4d, time:%4d' % (state['loss'], state['epoch'],
#                                                                         state['steps'], state['time']))
#
#         for batch in tqdm(data_loader):
#             batch = [b.cuda() for b in batch]
#             if is_sbj:
#                 s1, s2 = model(batch, is_train=False)
#                 s1 = s1.detach().cpu().numpy()
#                 s2 = s2.detach().cpu().numpy()
#
#                 for s1_i, s2_i in zip(s1, s2):
#                     r = []
#                     for i, s1_i_i in enumerate(s1_i):
#                         if s1_i_i >= test_value:
#                             for j, s2_i_j in enumerate(s2_i[i:]):
#                                 if s2_i_j >= test_value:
#                                     r.append([i, i+j+1])
#                                     break
#                     R[model_num].append(r)
#             else:
#                 r = model(batch, is_train=False, test_value=test_value)
#                 R[model_num].append(r)
#
#     if flag == 'val' and is_sbj:
#         text_lists, _, result = loader.gen_test_data('../data/dev_data.json', get_answer=True, is_sbj=True)
#         A, B, C = 1e-10, 1e-10, 1e-10
#         for i in range(len(text_lists)):
#             t = result[i]
#             text = text_lists[i]
#             r = set([])
#             for j in range(len(R)):
#                 r_tmp = set([''.join(text[r_i[0]: r_i[1]]) for r_i in R[j][i]])
#                 r = r | r_tmp
#             A += len(r & t)
#             B += len(r)
#             C += len(t)
#         f1 = A * 2 / (B + C)
#         precision = A / B
#         recall = A / C
#         print('sbj model, f1:%.4f, precision:%.4f, recall:%.4f\n' % (f1, precision, recall))
#
#     if flag == 'val' and (not is_sbj):
#         text_lists, _, result = loader.gen_test_data('../data/dev_data.json', get_answer=True, is_sbj=False)
#         A, B, C = 1e-10, 1e-10, 1e-10
#         for i in range(len(text_lists)):
#             t = set(result[i])
#             text = text_lists[i]
#             r = set([])
#             for j in range(len(R)):
#                 r_tmp = set([(''.join(text[r_i[0][0]: r_i[0][1]+1]), i2s[r_i[1]], ''.join(text[r_i[2][0]: r_i[2][1]+1]))
#                              for r_i in R[j][i]])
#                 r = r | r_tmp
#
#             p_set = set([r_i[1] for r_i in r])
#             if '妻子' in p_set and '丈夫' not in p_set:
#                 r_tmp = set([])
#                 for r_i in r:
#                     r_tmp.add(r_i)
#                     if r_i[1] == '妻子':
#                         r_tmp.add((r_i[2], '丈夫', r_i[0]))
#                 r = r_tmp
#             if '妻子' not in p_set and '丈夫' in p_set:
#                 r_tmp = set([])
#                 for r_i in r:
#                     r_tmp.add(r_i)
#                     if r_i[1] == '丈夫':
#                         r_tmp.add((r_i[2], '妻子', r_i[0]))
#                 r = r_tmp
#
#             A += len(r & t)
#             B += len(r)
#             C += len(t)
#         f1 = A * 2 / (B + C)
#         precision = A / B
#         recall = A / C
#         print('spo model, f1:%.4f, precision:%.4f, recall:%.4f\n' % (f1, precision, recall))
#
#     if flag == 'test':
#         text_lists, _, texts = loader.gen_test_data('../data/test1_data_postag.json', get_answer=False, is_sbj=False)
#         result_writer = open('../result/result.json', 'w')
#         for i in range(len(text_lists)):
#             text = text_lists[i]
#             r = set([])
#             for j in range(len(R)):
#                 r_tmp = set([(''.join(text[r_i[0][0]: r_i[0][1]+1]), i2s[r_i[1]], ''.join(text[r_i[2][0]: r_i[2][1]+1]))
#                              for r_i in R[j][i]])
#                 r = r | r_tmp
#
#             p_set = set([r_i[1] for r_i in r])
#             if '妻子' in p_set and '丈夫' not in p_set:
#                 r_tmp = set([])
#                 for r_i in r:
#                     r_tmp.add(r_i)
#                     if r_i[1] == '妻子':
#                         r_tmp.add((r_i[2], '丈夫', r_i[0]))
#                 r = r_tmp
#             if '妻子' not in p_set and '丈夫' in p_set:
#                 r_tmp = set([])
#                 for r_i in r:
#                     r_tmp.add(r_i)
#                     if r_i[1] == '丈夫':
#                         r_tmp.add((r_i[2], '妻子', r_i[0]))
#                 r = r_tmp
#
#             text_dic = {"text": texts[i]}
#             if len(r) == 0:
#                 spo_list = []
#             else:
#                 spo_list = [{'object_type': 'xx', 'predicate': p, 'object': obj, 'subject_type': 'xx', 'subject': sbj}
#                             for sbj, p, obj in r]
#             text_dic["spo_list"] = spo_list
#             result_writer.write(json.dumps(text_dic, ensure_ascii=False))
#             result_writer.write('\n')
#         result_writer.close()
#
#     if flag == 'val':
#         return f1, precision, recall
#
#     print(f'time:{time.time()-time_start}')


if __name__ == '__main__':
    # ner
    if False:
        config = config_ner.config
        config.model_path = 'model_obj_single_1'
        test(flag='val', task='obj', test_value=0.5, config=config)

    # spo
    if True:
        config = config_p.config
        config.model_path = 'model_p_single_1'
        config.model_path_sbj = 'model_sbj_single_1'
        config.model_path_obj = 'model_obj_single_1'
        test(flag='test', task='p', test_value=0.5, config=config)

    # # 集成: sbj
    # if False:
    #     config = config_ner.config
    #     model_paths = ['model_sbj_single_1', 'model_sbj_single_2', 'model_sbj_single_3']
    #     test_ensemble(flag='val', is_sbj=True, test_value=0.5, config=config, model_paths=model_paths)
    #
    # # 集成：spo
    # if False:
    #     config = config_spo.config
    #     config.model_path_sbj = 'model_sbj_0'
    #     model_paths = ['model_spo_single_1', 'model_spo_single_2']
    #     test_ensemble(flag='test', is_sbj=False, test_value=0.65, config=config, model_paths=model_paths)


