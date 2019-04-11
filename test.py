# encoding = utf-8
# author = xy


import time
import numpy as np
import loader
import torch
import json
import pickle
from tqdm import tqdm
from config import config_sbj
from config import config_spo
from modules.model_sbj import ModelSbj
from modules.model_spo import ModelSpo


# config = config_sbj.config
config = config_spo.config


def test(flag='val', is_sbj=True, test_value=0.5):
    time_start = time.time()
    embedding = np.load('../data/embedding.pkl.npy')
    data_loader = loader.build_loader(
        file_path='../data/dev_data.json' if flag == 'val' else '../data/test1_data_postag.json',
        batch_size=config.test_batch_size if is_sbj else 1,
        shuffle=False,
        drop_last=False,
        is_train=False,
        is_sbj=is_sbj
    )
    param = {
        'embedding': embedding,
        'mode': config.mode,
        'hidden_size': config.hidden_size,
        'dropout_p': config.dropout_p,
        'encoder_dropout_p': 0.1,
        'encoder_layer_num': config.encoder_layer_num,
        'model_path_sbj': config.model_path_sbj if config.name in ['ModelSpo'] else 'xxx'
    }
    model = eval(config.name)(param)
    model.cuda()
    model.eval()
    model_path = '../model/' + config.model_path + '.pkl'
    state = torch.load(model_path)
    model.load_state_dict(state['model_state'])
    print('load model, loss:%.4f, epoch:%2d, step:%4d, time:%4d' % (state['loss'], state['epoch'],
                                                                    state['steps'], state['time']))
    with open('../data/schemas.pkl', 'rb') as f:
        i2s = pickle.load(f)['i2s']
    R = []
    for batch in tqdm(data_loader):
        batch = batch.cuda()
        if is_sbj:
            s1, s2 = model(batch, is_train=False)
            s1 = s1.detach().cpu().numpy()
            s2 = s2.detach().cpu().numpy()
            for s1_i, s2_i in zip(s1, s2):
                r = []
                for i, s1_i_i in enumerate(s1_i):
                    if s1_i_i >= test_value:
                        for j, s2_i_j in enumerate(s2_i[i:]):
                            if s2_i_j >= test_value:
                                r.append([i, i+j+1])
                                break
                R.append(r)
        else:
            r = model(batch, is_train=False, test_value=test_value)
            R.append(r)

    if flag == 'val' and is_sbj:
        text_lists, result = loader.gen_test_data('../data/dev_data.json', get_answer=True, is_sbj=True)
        A, B, C = 1e-10, 1e-10, 1e-10
        for i in range(len(text_lists)):
            t = result[i]
            text = text_lists[i]
            r = set([''.join(text[r_i[0]: r_i[1]]) for r_i in R[i]])
            r = set(r)
            A += len(r & t)
            B += len(r)
            C += len(t)
        f1 = A * 2 / (B + C)
        precision = A / B
        recall = A / C
        print('sbj model, f1:%.4f, precision:%.4f, recall:%.4f' % (f1, precision, recall))

    if flag == 'val' and (not is_sbj):
        text_lists, result = loader.gen_test_data('../data/dev_data.json', get_answer=True, is_sbj=False)
        A, B, C = 1e-10, 1e-10, 1e-10
        for i in range(len(text_lists)):
            t = set(result[i])
            text = text_lists[i]
            r = set([(''.join(text[r_i[0][0]: r_i[0][1]+1]), i2s[r_i[1]], ''.join(text[r_i[2][0]: r_i[2][1]+1]))
                     for r_i in R[i]])
            A += len(r & t)
            B += len(r)
            C += len(t)
            f1 = A * 2 / (B + C)
        precision = A / B
        recall = A / C
        print('spo model, f1:%.4f, precision:%.4f, recall:%.4f' % (f1, precision, recall))

    if flag == 'test':
        text_lists, texts = loader.gen_test_data('../data/test1_data_postag.json', get_answer=False, is_sbj=False)
        result_writer = open('../result/result.json', 'w')
        for i in range(len(text_lists)):
            text = text_lists[i]
            r = set([(''.join(text[r_i[0][0]: r_i[0][1]+1]), i2s[r_i[1]], ''.join(text[r_i[2][0]: r_i[2][1]+1]))
                     for r_i in R[i]])
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
    # for i in np.arange(0.2, 0.51, 0.05):
    #     print(f'{i}th')
    #     test(flag='val', is_sbj=False, test_value=i)

    test(flag='test', is_sbj=False, test_value=0.4)




