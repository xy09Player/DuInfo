# encoding = utf-8
# author = xy


import time
import numpy as np
import loader
from modules import model_spo
import torch
import pickle
from tqdm import tqdm
from config import config_sbj
from modules.model_sbj import ModelSbj


config = config_sbj.config


def test(flag='val'):
    time_start = time.time()
    embedding = np.load('../data/embedding.pkl.npy')
    data_loader = loader.build_loader(
        file_path='../data/dev_data.json' if flag == 'val' else '../data/test1_data_postag.json',
        batch_size=128,
        shuffle=False,
        drop_last=False,
        is_train=False
    )
    param = {
        'embedding': embedding,
        'mode': 'LSTM',
        'hidden_size': 64,
        'dropout_p': 0.2,
        'encoder_dropout_p': 0.1,
        'encoder_layer_num': 1
    }
    model = model_spo.ModelBase(param)
    model.cuda()
    model.eval()
    state = torch.load('../model/baseline.pkl')
    model.load_state_dict(state['model_state'])
    print('load model, loss:%.4f, epoch:%2d, step:%4d, time:%4d' % (state['loss'], state['epoch'],
                                                                    state['steps'], state['time']))
    R = []
    for batch in tqdm(data_loader):
        batch = batch.cuda()
        R_tmp = model(batch, is_train=False, test_value=0.5)
        R += R_tmp

    with open('../data/schemas.pkl', 'rb') as f:
        i2s = pickle.load(f)['i2s']
    if flag == 'val':
        text_lists, result = loader.gen_test_data('../data/dev_data.json', get_answer=True)
        A, B, C = 1e-10, 1e-10, 1e-10
        for i in range(len(text_lists)):
            t = set(result[i])
            text = text_lists[i]
            r = [(''.join(text[rr[0][0]: rr[0][1]+1]), i2s[rr[1]], ''.join(text[rr[2][0]: rr[2][1]+1])) for rr in R[i]]
            r = set(r)
            A += len(r & t)
            B += len(r)
            C += len(t)
        f1 = A * 2 / (B + C)
        precision = A / B
        recall = A / C
        print('f1:%.4f, precision:%.4f, recall:%.4f' % (f1, precision, recall))

    else:
        text_lists, texts = loader.gen_test_data('test1_data_postag.json', get_answer=False)
        result = []
        for i in range(len(text_lists)):
            text = text_lists[i]
            r = [(''.join(text[rr[0][0]: rr[0][1]+1]), i2s[rr[1]], ''.join(text[rr[2][0]: rr[2][1]+1])) for rr in R[i]]
            r = set(r)
            text_str = texts[i]
            if len(r) == 0:
                spo_list = []
            else:
                spo_list = [{'object_type': 'xx', 'predicate': p, 'object': obj, 'subject_type': 'xx', 'subject': sbj}
                            for sbj, p, obj in r]



    print(f'time:{time.time()-time_start}')





if __name__ == '__main__':
    test('val')




