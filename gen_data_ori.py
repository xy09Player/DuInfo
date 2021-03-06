# coding = utf-8
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


def gen_p(config, ner_type, model_paths, data_type):
    time_start = time.time()

    if data_type == 'train':
        file_path = '../data/train_data.json'
    elif data_type == 'val':
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
        task=ner_type
    )

    param = {
        'mode': config.mode,
        'hidden_size': config.hidden_size,
        'dropout_p': config.dropout_p,
        'encoder_dropout_p': 0.1,
        'encoder_layer_num': config.encoder_layer_num,
    }

    R = [[] for _ in range(len(model_paths))]
    R_result = [[] for _ in range(len(model_paths))]
    if data_type != 'test':
        char_lists, result = loader.gen_test_data(file_path, get_answer=True, task=ner_type)
    else:
        char_lists, _ = loader.gen_test_data(file_path, get_answer=False, task=ner_type)
    for ith, model_ner in enumerate(model_paths):
        model = eval(config.name)(param)
        model.cuda()
        model.eval()
        model_path = '../model/' + model_ner + '.pkl'
        state = torch.load(model_path)
        model.load_state_dict(state['model_state'])
        print('model:%s, loss:%.4f, epoch:%2d, step:%4d, time:%4d' % (model_ner, state['loss'], state['epoch'],
                                                                      state['steps'], state['time']))

        for batch in tqdm(data_loader):
            batch = [b.cuda() for b in batch]
            ners = model(batch, is_train=False)
            for ner in ners:
                tmp = []
                if sum(ner) == 0:
                    tmp.append((-9, -9))
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
                            tmp.append((index_s, index_e))
                            index = index_e
                R[ith].append(tmp)

        if data_type != 'test':
            A, B, C = 1e-10, 1e-10, 1e-10
            for i in range(len(char_lists)):
                t = result[i]
                text = char_lists[i]
                r = set()
                for s_item, e_item in R[ith][i]:
                    if s_item >= 0 and e_item >= 0:
                        r.add(''.join(text[s_item: e_item+1]))
                R_result[ith].append(r)
                A += len(r & t)
                B += len(r)
                C += len(t)
            f1 = A * 2 / (B + C)
            precision = A / B
            recall = A / C
            print('%s , f1:%.4f, precision:%.4f, recall:%.4f\n' % (model_ner, f1, precision, recall))
        else:
            for i in range(len(char_lists)):
                text = char_lists[i]
                r = set()
                for s_item, e_item in R[ith][i]:
                    if s_item >= 0 and e_item >= 0:
                        r.add(''.join(text[s_item: e_item+1]))
                R_result[ith].append(r)
            print(f'{model_ner} finish.')

    # result
    R_result_tmp = []
    if data_type != 'test':
        A, B, C = 1e-10, 1e-10, 1e-10
        for i in range(len(char_lists)):
            t = result[i]
            r = set()
            for j in range(len(model_paths)):
                r = r | R_result[j][i]
            R_result_tmp.append(r)
            A += len(r & t)
            B += len(r)
            C += len(t)
        f1 = A * 2 / (B + C)
        precision = A / B
        recall = A / C
        print('ensemble, f1:%.4f, precision:%.4f, recall:%.4f\n' % (f1, precision, recall))

    else:
        for i in range(len(char_lists)):
            r = set()
            for j in range(len(model_paths)):
                r = r | R_result[j][i]
            R_result_tmp.append(r)

    if ner_type == 'sbj' and data_type == 'train':
        data_path = '../data/sbj_train.pkl'
    elif ner_type == 'sbj' and data_type == 'val':
        data_path = '../data/sbj_val.pkl'
    elif ner_type == 'sbj' and data_type == 'test':
        data_path = '../data/sbj_test.pkl'
    elif ner_type == 'obj' and data_type == 'train':
        data_path = '../data/obj_train.pkl'
    elif ner_type == 'obj' and data_type == 'val':
        data_path = '../data/obj_val.pkl'
    elif ner_type == 'obj' and data_type == 'test':
        data_path = '../data/obj_test.pkl'
    else:
        print('wrong ner_type,data_type')
        assert 1 == -1

    with open(data_path, 'wb') as f:
        pickle.dump(R_result_tmp, f)

    print(f'time:{time.time()-time_start}\n')


if __name__ == '__main__':
    # sbj
    if True:
        config = config_ner.config
        ner_type = 'sbj'
        model_paths = ['model_sbj_1', 'model_sbj_2', 'model_sbj_3', 'model_sbj_4', 'model_sbj_5']

        data_type = 'val'
        print('sbj, val...')
        gen_p(config, ner_type, model_paths, data_type)

        data_type = 'train'
        print('sbj, train...')
        gen_p(config, ner_type, model_paths, data_type)

        data_type = 'test'
        print('sbj, test...')
        gen_p(config, ner_type, model_paths, data_type)

    # obj
    if True:
        config = config_ner.config
        ner_type = 'obj'
        model_paths = ['model_obj_1', 'model_obj_2', 'model_obj_3', 'model_obj_4', 'model_obj_5']

        data_type = 'val'
        print('obj, val...')
        gen_p(config, ner_type, model_paths, data_type)

        data_type = 'train'
        print('obj, train...')
        gen_p(config, ner_type, model_paths, data_type)

        data_type = 'test'
        print('obj, test...')
        gen_p(config, ner_type, model_paths, data_type)





