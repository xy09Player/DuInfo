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
from config import config_xy
from modules.model_p import ModelP
from modules.model_xy import ModelXy


def deal_p_r(sbj_bound, obj_bound, result_vec, test_value):
    result = []
    for s_b, o_b, vec in zip(sbj_bound, obj_bound, result_vec):
        for i in range(len(vec)):
            if vec[i] >= test_value:
                result.append([s_b, o_b, i])
    return result


def deal_p_r_ensemble(i_dict, topn, test_value):
    result = []
    for so, pxs in i_dict.items():
        if len(pxs) >= topn:
            s_b, o_b = so
            pxs = np.array(pxs).mean(axis=0)
            for i in range(len(pxs)):
                if pxs[i] >= test_value:
                    result.append([s_b, o_b, i])
    return result


def test(config, data_type, model_paths, ner_files, top_ns, test_values):
    time_start = time.time()

    if data_type == 'val':
        file_path = '../data/dev_data.json'
    elif data_type == 'test':
        file_path = '../data/test1_data_postag.json'
    else:
        print('wrong data_type')
        assert 1 == -1

    with open('../data/p_dict.pkl', 'rb') as f:
        i2p = pickle.load(f)['i2p']

    param = {
        'mode': config.mode,
        'hidden_size': config.hidden_size,
        'dropout_p': config.dropout_p,
        'encoder_dropout_p': 0.1,
        'encoder_layer_num': config.encoder_layer_num,
    }

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

    R_sbj_bounds = [[] for _ in range(len(model_paths))]
    R_obj_bounds = [[] for _ in range(len(model_paths))]
    R_vecs = [[] for _ in range(len(model_paths))]
    for ith, (model_p, ner_file) in enumerate(zip(model_paths, ner_files)):
        model = eval(config.name)(param)
        model.cuda()
        model.eval()
        model_path = '../model/' + model_p + '.pkl'
        state = torch.load(model_path)
        model.load_state_dict(state['model_state'])
        print('%s, loss:%.4f, epoch:%2d, step:%4d, time:%4d' % (model_p, state['loss'], state['epoch'],
                                                                      state['steps'], state['time']))

        data_loader = loader.build_loader(
            file_path=file_path,
            batch_size=config.test_batch_size,
            shuffle=False,
            drop_last=False,
            is_train=False,
            task='p',
            ner_file=ner_file
        )

        for batch in tqdm(data_loader):
            batch = [b.cuda() for b in batch]
            r_sbj_bounds, r_obj_bounds, r_vecs = model(batch, is_train=False)
            R_sbj_bounds[ith] += r_sbj_bounds
            R_obj_bounds[ith] += r_obj_bounds
            R_vecs[ith] += r_vecs

        if data_type == 'val':
            A, B, C = 1e-10, 1e-10, 1e-10
            A_ner, B_ner, C_ner = 1e-10, 1e-10, 1e-10
            A_sbj_obj, B_sbj_obj, C_sbj_obj = 1e-10, 1e-10, 1e-10
            for i in range(len(char_lists)):
                t = set(result[i])
                text = char_lists[i]
                r_tmp = deal_p_r(R_sbj_bounds[ith][i], R_obj_bounds[ith][i], R_vecs[ith][i], test_value=0.5)
                r = set()
                for r_i in r_tmp:
                    sbj = ''.join(text[r_i[0][0]: r_i[0][1]+1])
                    obj = ''.join(text[r_i[1][0]: r_i[1][1]+1])
                    p = i2p[r_i[2]]
                    if sbj == obj:
                        continue
                    # if True:
                    if p not in ['妻子', '丈夫']:
                        r.add((sbj, p, obj))
                    else:
                        if p == '妻子':
                            r.add((sbj, p, obj))
                            r.add((obj, '丈夫', sbj))
                        else:
                            r.add((sbj, p, obj))
                            r.add((obj, '妻子', sbj))

                sbjs = set([x[0] for x in t])
                objs = set([x[2] for x in t])
                ners = sbjs | objs

                ners_p = set()
                for index_s, index_e in R_sbj_bounds[ith][i]:
                    ners_p.add(''.join(text[index_s: index_e+1]))
                for index_s, index_e in R_obj_bounds[ith][i]:
                    ners_p.add(''.join(text[index_s: index_e+1]))

                sbjs_objs = set([(xx[0], xx[2]) for xx in t])
                sbjs_objs_p = set()
                for s in ners_p:
                    for o in ners_p:
                        if s != o:
                            sbjs_objs_p.add((s, o))

                # if r != t:
                #     print('spo:', t)
                #     print('spo_p:', r)
                #     print('ner:', ners)
                #     print('ner_p:', ners_p)
                #     print('')

                A += len(r & t)
                B += len(r)
                C += len(t)

                A_ner += len(ners_p & ners)
                B_ner += len(ners_p)
                C_ner += len(ners)

                A_sbj_obj += len(sbjs_objs & sbjs_objs_p)
                B_sbj_obj += len(sbjs_objs_p)
                C_sbj_obj += len(sbjs_objs)

            f1 = A * 2 / (B + C)
            precision = A / B
            recall = A / C
            print('%s, join, f1:%.4f, precision:%.4f, recall:%.4f' % (model_p, f1, precision, recall))

            f1_ner = A_ner * 2 / (B_ner + C_ner)
            precision_ner = A_ner / B_ner
            recall_ner = A_ner / C_ner
            print('%s, ner, f1:%.4f, precision:%.4f, recall:%.4f' % (model_p, f1_ner, precision_ner, recall_ner))

            f1_sbj_obj = A_sbj_obj * 2 / (B_sbj_obj + C_sbj_obj)
            precision_sbj_obj = A_sbj_obj / B_sbj_obj
            recall_sbj_obj = A_sbj_obj / C_sbj_obj
            print('%s, so, f1:%.4f, precision:%.4f, recall:%.4f\n' % (model_p, f1_sbj_obj, precision_sbj_obj, recall_sbj_obj))

    # ensemble
    if True:
        R_result = [{} for _ in range(len(R_vecs[0]))]
        for i in range(len(R_result)):
            for j in range(len(R_vecs)):
                for sbj_index, obj_index, px in zip(R_sbj_bounds[j][i], R_obj_bounds[j][i], R_vecs[j][i]):
                    xxx = (tuple(sbj_index), tuple(obj_index))
                    if xxx in R_result[i]:
                        R_result[i][xxx].append(px)
                    else:
                        R_result[i][xxx] = [px]

        # val
        if data_type == 'val':
            for topn in top_ns:
                for test_value in test_values:
                    A, B, C = 1e-10, 1e-10, 1e-10
                    for i in range(len(char_lists)):
                        t = set(result[i])
                        text = char_lists[i]
                        r_tmp = deal_p_r_ensemble(R_result[i], topn, test_value)
                        r = set()
                        for r_i in r_tmp:
                            sbj = ''.join(text[r_i[0][0]: r_i[0][1]+1])
                            obj = ''.join(text[r_i[1][0]: r_i[1][1]+1])
                            p = i2p[r_i[2]]
                            if sbj == obj:
                                continue

                            # if True:
                            if p not in ['妻子', '丈夫']:
                                r.add((sbj, p, obj))
                            else:
                                if p == '妻子':
                                    r.add((sbj, p, obj))
                                    r.add((obj, '丈夫', sbj))
                                else:
                                    r.add((sbj, p, obj))
                                    r.add((obj, '妻子', sbj))

                        A += len(r & t)
                        B += len(r)
                        C += len(t)
                    f1 = A * 2 / (B + C)
                    precision = A / B
                    recall = A / C
                    print('ensemble, top_n:%d, test_value:%.2f, f1:%.4f, precision:%.4f, recall:%.4f' %
                          (topn, test_value, f1, precision, recall))
                print('')

        # test
        if data_type == 'test':
            topn = top_ns
            result_writer = open('../result/result.json', 'w')
            for i in range(len(char_lists)):
                text = char_lists[i]
                r_tmp = deal_p_r_ensemble(R_result[i], topn, test_value=test_values)
                r = set()
                for r_i in r_tmp:
                    sbj = ''.join(text[r_i[0][0]: r_i[0][1]+1])
                    obj = ''.join(text[r_i[1][0]: r_i[1][1]+1])
                    p = i2p[r_i[2]]
                    if sbj == obj:
                        continue
                    # if True:
                    if p not in ['妻子', '丈夫']:
                        r.add((sbj, p, obj))
                    else:
                        if p == '妻子':
                            r.add((sbj, p, obj))
                            r.add((obj, '丈夫', sbj))
                        else:
                            r.add((sbj, p, obj))
                            r.add((obj, '妻子', sbj))

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
    # val
    if False:
        config = config_p.config
        model_paths = ['model_p_1', 'model_p_2', 'model_p_3', 'model_p_4']
        ner_fils = ['val_ner_1', 'val_ner_2', 'val_ner_3', 'val_ner_4']
        topns = [2, 3]
        test_values = [0.5]
        test(config, 'val', model_paths, ner_fils, topns, test_values)

    # test
    if True:
        config = config_p.config
        model_paths = ['model_p_1', 'model_p_2', 'model_p_3', 'model_p_4']
        ner_fils = ['test_ner_1', 'test_ner_2', 'test_ner_3', 'test_ner_4']
        topns = 2
        test_values = 0.5
        test(config, 'test', model_paths, ner_fils, topns, test_values)



