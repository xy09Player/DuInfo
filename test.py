# encoding = utf-8
# author = xy


import time
import numpy as np
import loader
import torch
import json
import pickle
from tqdm import tqdm
from config import config_p
from modules.model_p import ModelP


def deal_p_r(sbj_bound, obj_bound, result_vec, test_value):
    result = []
    for s_b, o_b, vec in zip(sbj_bound, obj_bound, result_vec):
        for i in range(len(vec)):
            if vec[i] >= test_value:
                result.append([s_b, o_b, i])
    return result


def test(config, data_type, model_paths, test_value):
    time_start = time.time()

    if data_type == 'val':
        file_path = '../data/dev_data.json'
    elif data_type == 'test':
        file_path = '../data/test1_data_postag.json'
    else:
        print('wrong data_type')
        assert 1 == -1

    data_loader = loader.build_loader(
        file_path=file_path,
        batch_size=config.test_batch_size,
        shuffle=False,
        drop_last=False,
        is_train=False,
        task='p'
    )
    param = {
        'mode': config.mode,
        'hidden_size': config.hidden_size,
        'dropout_p': config.dropout_p,
        'encoder_dropout_p': 0.1,
        'encoder_layer_num': config.encoder_layer_num,
    }

    with open('../data/p_dict.pkl', 'rb') as f:
        i2p = pickle.load(f)['i2p']

    with open('../data/sbj_val.pkl', 'rb') as f:
        sbj_val = pickle.load(f)

    with open('../data/obj_val.pkl', 'rb') as f:
        obj_val = pickle.load(f)

    R_sbj_bounds = [[] for _ in range(len(model_paths))]
    R_obj_bounds = [[] for _ in range(len(model_paths))]
    R_vecs = [[] for _ in range(len(model_paths))]
    if data_type == 'val':
        char_lists, result = loader.gen_test_data(file_path, get_answer=True, task='p')
    else:
        char_lists = []
        texts = []
        with open(file_path, 'r') as f:
            for line in f:
                tmp = json.loads(line)
                texts.append(tmp['text'])
                char_list = loader.split_word(tmp['text'])
                char_lists.append(char_list)
    for ith, model_p in enumerate(model_paths):
        model = eval(config.name)(param)
        model.cuda()
        model.eval()
        model_path = '../model/' + model_p + '.pkl'
        state = torch.load(model_path)
        model.load_state_dict(state['model_state'])
        print('model:%s, loss:%.4f, epoch:%2d, step:%4d, time:%4d' % (model_p, state['loss'], state['epoch'],
                                                                      state['steps'], state['time']))

        for batch in tqdm(data_loader):
            batch = [b.cuda() for b in batch]
            r_sbj_bounds, r_obj_bounds, r_vecs = model(batch, False)
            R_sbj_bounds[ith] += r_sbj_bounds
            R_obj_bounds[ith] += r_obj_bounds
            R_vecs[ith] += r_vecs

        if data_type == 'val':
            A, B, C = 1e-10, 1e-10, 1e-10
            for i in range(len(char_lists)):
                t = set(result[i])
                text = char_lists[i]
                r_tmp = deal_p_r(R_sbj_bounds[ith][i], R_obj_bounds[ith][i], R_vecs[ith][i], test_value)
                r = set()
                for r_i in r_tmp:
                    sbj = ''.join(text[r_i[0][0]: r_i[0][1]+1])
                    obj = ''.join(text[r_i[1][0]: r_i[1][1]+1])
                    p = i2p[r_i[2]]
                    if True:
                        r.add((sbj, p, obj))
                    else:
                        r.add((sbj, p, obj))
                        r.add((obj, p, sbj))

                if r != t:
                    print('spo:', t)
                    print('spo_p:', r)
                    print('sbj:', set([tt[0] for tt in t]))
                    print('sbj_p:', sbj_val[i])
                    print('obj:', set([tt[2] for tt in t]))
                    print('obj_p:', obj_val[i])
                    print('')

                A += len(r & t)
                B += len(r)
                C += len(t)
            f1 = A * 2 / (B + C)
            precision = A / B
            recall = A / C
            print('%s, f1:%.4f, precision:%.4f, recall:%.4f\n' % (model_p, f1, precision, recall))

    if False:
        R_vecs_tmp = R_vecs[0]
        for i in range(len(R_vecs[0])):
            tmp = R_vecs_tmp[i]
            for j in range(1, len(R_vecs)):
                tmp += R_vecs[j][i]
            tmp = tmp / len(model_paths)
            R_vecs_tmp[i] = tmp

        if data_type == 'val':
            A, B, C = 1e-10, 1e-10, 1e-10
            for i in range(len(char_lists)):
                t = set(result[i])
                text = char_lists[i]
                r_tmp = deal_p_r(R_sbj_bounds[0][i], R_obj_bounds[0][i], R_vecs_tmp[i], test_value)
                r = set()
                for r_i in r_tmp:
                    sbj = ''.join(text[r_i[0][0]: r_i[0][1]+1])
                    obj = ''.join(text[r_i[1][0]: r_i[1][1]+1])
                    p = i2p[r_i[2]]
                    # if p not in ['妻子', '丈夫']:
                    if True:
                        r.add((sbj, p, obj))
                    else:
                        r.add((sbj, p, obj))
                        r.add((obj, p, sbj))

                    # if r != t:
                    #     print(t)
                    #     print(r)
                    #     print('')

                A += len(r & t)
                B += len(r)
                C += len(t)
            f1 = A * 2 / (B + C)
            precision = A / B
            recall = A / C
            print('ensemble, f1:%.4f, precision:%.4f, recall:%.4f\n' % (f1, precision, recall))
        else:
            result_writer = open('../result/result.json', 'w')
            for i in range(len(char_lists)):
                text = char_lists[i]
                r_tmp = deal_p_r(R_sbj_bounds[0][i], R_obj_bounds[0][i], R_vecs_tmp[i], test_value)
                r = set()
                for r_i in r_tmp:
                    sbj = ''.join(text[r_i[0][0]: r_i[0][1]+1])
                    obj = ''.join(text[r_i[1][0]: r_i[1][1]+1])
                    p = i2p[r_i[2]]
                    # if p not in ['妻子', '丈夫']:
                    if True:
                        r.add((sbj, p, obj))
                    else:
                        r.add((sbj, p, obj))
                        r.add((obj, p, sbj))

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

    print(f'time:{time.time()-time_start}')

if __name__ == '__main__':
    # p
    config = config_p.config
    # model_paths = ['model_p_1', 'model_p_2', 'model_p_3', 'model_p_4', 'model_p_5']
    model_paths = ['model_p_1']
    test(config, 'val', model_paths, test_value=0.5)



