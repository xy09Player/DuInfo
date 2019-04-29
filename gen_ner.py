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


def gen_ner(config, model_path, data_type, index_xx):
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
        task='ner'
    )

    param = {
        'mode': config.mode,
        'hidden_size': config.hidden_size,
        'dropout_p': config.dropout_p,
        'encoder_dropout_p': 0.1,
        'encoder_layer_num': config.encoder_layer_num,
    }

    R = []
    R_result = []
    if data_type != 'test':
        char_lists, result = loader.gen_test_data(file_path, get_answer=True, task='p')
    else:
        char_lists, _ = loader.gen_test_data(file_path, get_answer=False, task='ner')

    model = eval(config.name)(param)
    model.cuda()
    model.eval()
    state = torch.load('../model/' + model_path + '.pkl')
    model.load_state_dict(state['model_state'])
    print('%s, loss:%.4f, epoch:%2d, step:%4d, time:%4d' % (model_path, state['loss'], state['epoch'],
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
                            if ner[i:][j] == 3:
                                index_e = i + j
                                break
                            if ner[i:][j] == 4:
                                index_e = i + j
                                break
                            if j == len(ner[i:]) - 1:
                                index_e = len(ner) - 1
                        tmp.append((index_s, index_e))
                        index = index_e
            R.append(tmp)

    if data_type != 'test':
        A, B, C = 1e-10, 1e-10, 1e-10
        A_so, B_so, C_so = 1e-10, 1e-10, 1e-10
        for i in range(len(char_lists)):
            t = set()
            for xx in result[i]:
                t.add(xx[0])
                t.add(xx[2])

            text = char_lists[i]
            r = set()
            for s_item, e_item in R[i]:
                if s_item >= 0 and e_item >= 0:
                    r.add(''.join(text[s_item: e_item+1]))
            R_result.append(r)

            sbjs_objs = set([(xx[0], xx[2]) for xx in result[i]])
            sbjs_objs_p = set()
            for s in r:
                for o in r:
                    if s != o:
                        sbjs_objs_p.add((s, o))

            # if r != t:
            #     print('ner:', t)
            #     print('ner_p:', r)
            #     print('')

            A += len(r & t)
            B += len(r)
            C += len(t)

            A_so += len(sbjs_objs_p & sbjs_objs)
            B_so += len(sbjs_objs_p)
            C_so += len(sbjs_objs)

        f1 = A * 2 / (B + C)
        precision = A / B
        recall = A / C
        print('%s, ner, f1:%.4f, precision:%.4f, recall:%.4f\n' % (model_path, f1, precision, recall))

        f1_so = A_so * 2 / (B_so + C_so)
        precision_so = A_so / B_so
        recall_so = A_so / C_so
        print('%s, so, f1:%.4f, precision:%.4f, recall:%.4f\n' % (model_path, f1_so, precision_so, recall_so))
    else:
        for i in range(len(char_lists)):
            text = char_lists[i]
            r = set()
            for s_item, e_item in R[i]:
                if s_item >= 0 and e_item >= 0:
                    r.add(''.join(text[s_item: e_item+1]))
            R_result.append(r)
        print(f'{model_path} finish.')

    if data_type == 'train':
        data_path = '../data/train_ner_' + str(index_xx) + '.pkl'
    elif data_type == 'val':
        data_path = '../data/val_ner_' + str(index_xx) + '.pkl'
    elif data_type == 'test':
        data_path = '../data/test_ner_' + str(index_xx) + '.pkl'
    else:
        print('wrong ner_type,data_type')
        assert 1 == -1

    with open(data_path, 'wb') as f:
        pickle.dump(R_result, f)

    print(f'time:{time.time()-time_start}\n')


if __name__ == '__main__':
    # ner
    config = config_ner.config
    ner_type = 'ner'
    model_paths = ['model_ner_1']

    for i, model_path in enumerate(model_paths):
        i = i + 1
        data_type = 'val'
        print(f'{model_path}, val...')
        gen_ner(config, model_path, data_type, i)

        # data_type = 'train'
        # print(f'{model_path}, train...')
        # gen_ner(config, model_path, data_type, i)

        data_type = 'test'
        print(f'{model_path}, test...')
        gen_ner(config, model_path, data_type, i)
