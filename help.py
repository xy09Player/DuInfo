# coding = utf-8
# author = xy


import numpy as np
import json
from sklearn.model_selection import KFold


def get_k_fold_data(k=10):
    data = []
    train_path = '../data/train_data.json'
    with open(train_path, 'r') as f:
        for line in f:
            tmp = json.loads(line)
            data.append(tmp)
    val_path = '../data/dev_data.json'
    with open(val_path, 'r') as f:
        for line in f:
            tmp = json.loads(line)
            data.append(tmp)

    data_np = np.array(data)
    splits = KFold(n_splits=k, shuffle=True, random_state=333).split(list(range(len(data_np))))
    for i, (train_idx, val_idx) in enumerate(splits):
        train_data = data_np[train_idx]
        val_data = data_np[val_idx]
        train_path = '../data/' + str(i) + '_train_data.json'
        val_path = '../data/' + str(i) + '_val_data.json'

        with open(train_path, 'w') as f:
            for d in train_data:
                f.write(json.dumps(d, ensure_ascii=False))
                f.write('\n')

        with open(val_path, 'w') as f:
            for d in val_data:
                f.write(json.dumps(d, ensure_ascii=False))
                f.write('\n')


if __name__ == '__main__':
    # get_k_fold_data(k=10)
    pass
