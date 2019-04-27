# coding = utf-8
# author = xy

from gen_ner import gen_ner
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


def train(task, config, train_ner_file=None, val_ner_file=None):
    time_start = time.time()
    train_loader = loader.build_loader(
        file_path=config.train_path,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        is_train=True,
        task=task,
        ner_file=train_ner_file
    )
    val_loader = loader.build_loader(
        file_path=config.val_path,
        batch_size=config.test_batch_size,
        shuffle=False,
        drop_last=False,
        is_train=True,
        task=task,
        ner_file=val_ner_file
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
    # plt.figure(figsize=[12, 7])
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
            loss = model(batch, is_train=True)
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
                        loss = model(batch, is_train=True)
                        val_loss += loss.item()
                        val_c += 1

                train_loss_list.append(train_loss / train_c)
                val_loss_list.append(val_loss / val_c)

                steps.append(train_c)

                print('epoch:%2d, steps:%5d, train_loss:%.4f, val_loss:%.4f, time:%4d' %
                      (e, sum(steps), train_loss/train_c, val_loss / val_c, time.time()-time_start))

                train_loss = 0
                train_c = 0

                # draw
                plt.cla()
                x = np.cumsum(steps)
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
    for i in range(1, 2):
        # ner
        # config = config_ner.config
        # config.model_path = 'model_ner_' + str(i)
        # print(f'{config.model_path}, training...')
        # train(task='ner', config=config)
        #
        # # ner gen
        # ner_type = 'ner'
        # data_type = 'val'
        # print(f'{config.model_path}, gen val ner...')
        # gen_ner(config, config.model_path, data_type, i)
        #
        # data_type = 'train'
        # print(f'{config.model_path}, gen train ner')
        # gen_ner(config, config.model_path, data_type, i)
        #
        # data_type = 'test'
        # print(f'{config.model_path}, gen test ner')
        # gen_ner(config, config.model_path, data_type, i)

        # p
        config = config_p.config
        config.model_path = 'model_p_' + str(i)
        train_ner_file = 'train_ner_' + str(i) + '.pkl'
        val_ner_file = 'val_ner_' + str(i) + '.pkl'
        train(task='p', config=config, train_ner_file=train_ner_file, val_ner_file=val_ner_file)











