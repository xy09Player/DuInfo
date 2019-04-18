# coding = utf-8
# author = xy

import os
import time
import loader
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from config import config_ner
from config import config_p
from modules.model_ner import ModelNer
from modules.model_p import ModelP


def train(task, config):
    time_start = time.time()
    embedding = np.load('../data/embedding.pkl.npy')
    train_loader = loader.build_loader(
        file_path=config.train_path,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        is_train=True,
        task=task
    )
    val_loader = loader.build_loader(
        file_path=config.val_path,
        batch_size=config.test_batch_size,
        shuffle=False,
        drop_last=False,
        is_train=True,
        task=task
    )
    param = {
        'embedding': embedding,
        'mode': config.mode,
        'hidden_size': config.hidden_size,
        'dropout_p': config.dropout_p,
        'encoder_dropout_p': 0.1,
        'encoder_layer_num': config.encoder_layer_num,
        'model_path_sbj': config.model_path_sbj if task == 'p' else 'xxx',
        'model_path_obj': config.model_path_obj if task == 'p' else 'xxx'
    }
    model = eval(config.name)(param)
    model.cuda()

    optimizer_param = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(optimizer_param, lr=config.lr)
    model_param_num = 0
    for p in model.parameters():
        if p.requires_grad:
            model_param_num += p.nelement()
    print(f'model param_num:{model_param_num}')

    plt.ion()
    train_loss = 0
    train_c = 0
    train_loss_list = []
    val_loss_list = []
    steps = []
    early_stop = 0
    for e in range(config.epochs):
        for batch in train_loader:
            batch = [b.cuda() for b in batch]
            model.train()
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_c += 1

            if train_c % 500 == 0:
                val_loss = 0
                val_c = 0
                with torch.no_grad():
                    model.eval()
                    for batch in val_loader:
                        batch = [b.cuda() for b in batch]
                        loss = model(batch)
                        val_loss += loss.item()
                        val_c += 1

                train_loss_list.append(train_loss / train_c)
                val_loss_list.append(val_loss / val_c)
                steps.append(train_c)

                print('training, epoch:%2d, steps:%5d, train_loss:%.4f, val_loss:%.4f, time:%4d' %
                      (e, sum(steps), train_loss/train_c, val_loss/val_c, time.time()-time_start))
                train_loss = 0
                train_c = 0

                # draw
                plt.cla()
                x = np.cumsum(steps)
                plt.plot(
                    x,
                    train_loss_list,
                    color='r',
                    label='train'
                )
                plt.plot(
                    x,
                    val_loss_list,
                    color='b',
                    label='val'
                )
                fig_path = '../result/' + config.model_path + '.png'
                plt.xlabel('steps')
                plt.ylabel('loss')
                plt.legend()
                plt.pause(0.000000001)
                plt.savefig(fig_path)
                plt.show()

                model_path = '../model/' + config.model_path + '.pkl'
                if os.path.isfile(model_path):
                    state = torch.load(model_path)
                else:
                    state = {}

                if state == {} or state['loss'] > (val_loss/val_c):
                    early_stop = 0
                    state['model_state'] = model.state_dict()
                    state['loss'] = val_loss/val_c
                    state['epoch'] = e
                    state['steps'] = sum(steps)
                    state['time'] = time.time() - time_start
                    torch.save(state, model_path)
                else:
                    early_stop += 1
                    if early_stop >= config.early_stop:
                        break
        if early_stop >= config.early_stop:
            break


if __name__ == '__main__':
    # ner: sbj
    if False:
        config = config_ner.config
        config.model_path = 'model_sbj_single_1'
        train(task='sbj', config=config)

    # ner: obj
    if False:
        config = config_ner.config
        config.model_path = 'model_obj_single_1'
        train(task='obj', config=config)

    # spo
    if True:
        config = config_p.config
        config.model_path = 'model_p_single'
        config.model_path_sbj = 'model_sbj_single'
        config.model_path_obj = 'model_obj_single'
        train(task='p', config=config)

    # 10-fold 集成
    if False:
        for i in range(10):
            print(f'{i}-th, training....')
            config = config_ner.config
            config.train_path = '../data/' + str(i) + '_train_data.json'
            config.val_path = '../data/' + str(i) + '_val_data.json'
            config.model_path = 'model_sbj_' + str(i)
            train(task=True, config=config)

            config = config_p.config
            config.train_path = '../data/' + str(i) + '_train_data.json'
            config.val_path = '../data/' + str(i) + '_val_data.json'
            config.model_path = 'model_spo_' + str(i)
            config.model_path_sbj = 'model_sbj_' + str(i)
            train(task=False, config=config)
            print('\n')















