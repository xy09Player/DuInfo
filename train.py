# coding = utf-8
# author = xy

import os
import sys
import time
import loader
import numpy as np
import matplotlib.pyplot as plt
from modules import model_baseline
import torch
from torch.optim import Adam


def train():
    time_start = time.time()
    embedding = np.load('../data/embedding.pkl.npy')
    train_loader = loader.build_loader(
        file_path='../data/train_data.json',
        batch_size=64,
        shuffle=True,
        drop_last=True
    )
    val_loader = loader.build_loader(
        file_path='../data/dev_data.json',
        batch_size=256,
        shuffle=False,
        drop_last=False
    )
    param = {
        'embedding': embedding,
        'mode': 'LSTM',
        'hidden_size': 64,
        'dropout_p': 0.2,
        'encoder_dropout_p': 0.1,
        'encoder_layer_num': 1
    }
    model = model_baseline.ModelBase(param)
    model.cuda()

    optimizer_param = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(optimizer_param, lr=1e-3)
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
    for e in range(50):
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
                plt.xlabel('steps')
                plt.ylabel('loss')
                plt.legend()
                plt.pause(0.000000001)
                plt.savefig('../result/train.png')
                plt.show()

                if os.path.isfile('../model/baseline.pkl'):
                    state = torch.load('../model/baseline.pkl')
                else:
                    state = {}

                if state == {} or state['loss'] > (val_loss/val_c):
                    early_stop = 0
                    state['model_state'] = model.state_dict()
                    state['loss'] = val_loss/val_c
                    state['epoch'] = e
                    state['steps'] = sum(steps)
                    state['time'] = time.time() - time_start
                    torch.save(state, '../model/baseline.pkl')
                else:
                    early_stop += 1
                    if early_stop == 5:
                        sys.exit()


if __name__ == '__main__':
    train()














