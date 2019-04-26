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
from config import config_xy
from modules.model_ner import ModelNer
from modules.model_p import ModelP
from modules.model_xy import ModelXy


def train(task, config):
    time_start = time.time()
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
        'mode': config.mode,
        'hidden_size': config.hidden_size,
        'dropout_p': config.dropout_p,
        'encoder_dropout_p': 0.1,
        'encoder_layer_num': config.encoder_layer_num,
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
    plt.figure(figsize=[12, 7])
    train_loss = 0
    train_loss_sbj = 0
    train_loss_obj = 0
    train_loss_p = 0
    train_c = 0

    train_loss_list = []
    train_loss_sbj_list = []
    train_loss_obj_list = []
    train_loss_p_list = []
    val_loss_list = []
    val_loss_sbj_list = []
    val_loss_obj_list = []
    val_loss_p_list = []

    steps = []
    early_stop = 0
    for e in range(config.epochs):
        for batch in train_loader:
            batch = [b.cuda() for b in batch]
            model.train()
            optimizer.zero_grad()
            loss, loss_sbj, loss_obj, loss_p = model(batch, is_train=True, have_p=False if e == 0 else True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loss_sbj += loss_sbj.item()
            train_loss_obj += loss_obj.item()
            train_loss_p += loss_p.item()
            train_c += 1

            if train_c % 500 == 0:
                val_loss = 0
                val_loss_sbj = 0
                val_loss_obj = 0
                val_loss_p = 0
                val_c = 0
                with torch.no_grad():
                    model.eval()
                    for batch in val_loader:
                        batch = [b.cuda() for b in batch]
                        loss, loss_sbj, loss_obj, loss_p = model(batch, is_train=True, have_p=False if e == 0 else True)
                        val_loss += loss.item()
                        val_loss_sbj += loss_sbj.item()
                        val_loss_obj += loss_obj.item()
                        val_loss_p += loss_p.item()
                        val_c += 1

                train_loss_list.append(train_loss / train_c)
                train_loss_sbj_list.append(train_loss_sbj / train_c)
                train_loss_obj_list.append(train_loss_obj / train_c)
                train_loss_p_list.append(train_loss_p / train_c)

                val_loss_list.append(val_loss / val_c)
                val_loss_sbj_list.append(val_loss_sbj / val_c)
                val_loss_obj_list.append(val_loss_obj / val_c)
                val_loss_p_list.append(val_loss_p / val_c)

                steps.append(train_c)

                print('training, epoch:%2d, steps:%5d, '
                      'train_loss:%.4f, train_sbj_loss:%.4f, train_obj_loss:%.4f, train_p_loss:%.4f, '
                      'val_loss:%.4f, val_sbj_loss:%.4f, val_obj_loss:%.4f, val_p_loss:%.4f, '
                      'time:%4d' %
                      (e, sum(steps),
                       train_loss/train_c, train_loss_sbj / train_c, train_loss_obj / train_c, train_loss_p / train_c,
                       val_loss / val_c, val_loss_sbj / val_c, val_loss_obj / val_c, val_loss_p / val_c,
                       time.time()-time_start))

                train_loss = 0
                train_loss_sbj = 0
                train_loss_obj = 0
                train_loss_p = 0
                train_c = 0

                # draw
                plt.cla()
                x = np.cumsum(steps)
                plt.subplot(2, 2, 1)
                plt.title('loss')
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

                plt.subplot(2, 2, 2)
                plt.title('p_loss')
                plt.plot(
                    x,
                    train_loss_p_list,
                    color='r',
                    label='train'
                )
                plt.plot(
                    x,
                    val_loss_p_list,
                    color='b',
                    label='val'
                )

                plt.subplot(2, 2, 3)
                plt.title('sbj_loss')
                plt.plot(
                    x,
                    train_loss_sbj_list,
                    color='r',
                    label='train'
                )
                plt.plot(
                    x,
                    val_loss_sbj_list,
                    color='b',
                    label='val'
                )

                plt.subplot(2, 2, 4)
                plt.title('obj_loss')
                plt.plot(
                    x,
                    train_loss_obj_list,
                    color='r',
                    label='train'
                )
                plt.plot(
                    x,
                    val_loss_obj_list,
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
    # single
    if True:
        config = config_xy.config
        config.model_path = 'model_xy_test'
        train(task='join', config=config)

    # muti
    if False:
        config = config_xy.config
        for xx in range(1, 2):
            config.model_path = 'model_xy_' + str(xx)
        train(task='ner', config=config)


