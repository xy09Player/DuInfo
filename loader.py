# coding = utf-8
# author = xy

import numpy as np
import json
import gensim
import pickle
import jieba
from torch.utils.data import Dataset, DataLoader
import torch


def get_dict_schemas():
    schemas = []
    with open('../data/all_50_schemas') as f:
        for line in f:
            tmp = json.loads(line)
            schemas.append(tmp['predicate'])
    schemas = set(schemas)
    s2i = {}
    i2s = {}
    counts = 1
    for schema in schemas:
        s2i[schema] = counts
        i2s[counts] = schema
        counts += 1
    print(f's2i_num:{len(schemas)}')
    schemes_dict = {'s2i': s2i, 'i2s': i2s}
    with open('../data/schemas.pkl', 'wb') as f:
        pickle.dump(schemes_dict, f)


def build_vocab_embedding():
    word_list = []
    datas = ['../data/train_data.json', '../data/dev_data.json', '../data/test1_data_postag.json']
    for data in datas:
        with open(data, 'r') as f:
            for line in f:
                tmp = json.loads(line)
                # words = [item['word'] for item in tmp['postag']]
                words = jieba.lcut(tmp['text'], HMM=False)
                word_list += words

    print(f'words_num: {len(word_list)}')

    vocab = set()
    for word in word_list:
        vocab.add(word)

    w2v_model = gensim.models.KeyedVectors.load_word2vec_format('../data/merge_sgns_bigram_char300.txt')
    vocab_kwn = set()
    for word in vocab:
        if word in w2v_model:
            vocab_kwn.add(word)

    words_known_num = 0
    for word in word_list:
        if word in vocab_kwn:
            words_known_num += 1
    print('known_word_num:%d/%d, radio:%.4f' % (words_known_num, len(word_list), words_known_num/len(word_list)))
    print('word_num in pre-embedding:%d/%d, radio:%.4f' % (len(vocab_kwn), len(vocab), len(vocab_kwn)/len(vocab)))

    w2i = {'<pad>': 0}
    i2w = {0: '<pad>'}
    count = 1
    embedding = np.zeros([len(vocab_kwn) + 2, 300])
    for word in vocab_kwn:
        w2i[word] = count
        i2w[count] = word
        embedding[count] = w2v_model[word]
        count += 1
    w2i['<unk>'] = count
    i2w[count] = ['<unk>']
    lang = {'w2i': w2i, 'i2w': i2w}
    assert len(vocab_kwn) + 2 == len(w2i)
    assert len(w2i) == embedding.shape[0]
    print('vocab_size:%d' % (len(vocab_kwn)+2))

    with open('../data/lang.pkl', 'wb') as f:
        pickle.dump(lang, f)
    np.save('../data/embedding.pkl', embedding)





def word2index(word_lists):
    with open('../data/lang.pkl', 'rb') as f:
        lang = pickle.load(f)
        w2i = lang['w2i']
    result = []
    for words in word_lists:
        tmp = [w2i[word] if word in w2i else w2i['<unk>'] for word in words]
        result.append(tmp)
    return result


def padding(index_lists, max_len=0):
    if max_len == 0:
        lens = [len(index) for index in index_lists]
        max_len = max(lens)
    result = []
    for index in index_lists:
        if len(index) > max_len:
            result.append(index[: max_len])
        else:
            result.append(index + [0] * (max_len - len(index)))
    return result


def gen_train_data_sbj(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            tmp = json.loads(line)
            data.append({
                'text': tmp['text'],
                'spo_list': [(spo['subject'], spo['predicate'], spo['object']) for spo in tmp['spo_list']]
            })

    texts = []
    sbj_starts = []
    sbj_ends = []
    for d in data:
        text = d['text']
        text_list = jieba.lcut(text, HMM=False)
        text_len = len(text_list)
        sbj_start = np.zeros(text_len).tolist()
        sbj_end = np.zeros(text_len).tolist()

        flag = 0
        for spo in d['spo_list']:
            sbj_s = -1
            sbj_e = -1
            sbj_list = jieba.lcut(spo[0], HMM=False)
            sbj_len = len(sbj_list)
            for i in range(0, text_len-sbj_len+1):
                if text_list[i: i+sbj_len] == sbj_list:
                    sbj_s = i
                    sbj_e = i + sbj_len - 1
                    break
            if sbj_s != -1 and sbj_e != -1:
                sbj_start[sbj_s] = 1
                sbj_end[sbj_e] = 1
                flag = 1

        if flag:
            texts.append(text_list)
            sbj_starts.append(sbj_start)
            sbj_ends.append(sbj_end)

    spo_lists = [d['spo_list'] for d in data]
    all_nums = 0
    for spo_list in spo_lists:
        all_nums += len(set([spo[0] for spo in spo_list]))

    sample_nums = 0
    for sbj_start in sbj_starts:
        sample_nums += sum(sbj_start)

    print('sbj, make samples, nums:%d/%d, radio:%.4f' % (sample_nums, all_nums, sample_nums/all_nums))

    return texts, sbj_starts, sbj_ends


def gen_train_data_spo(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            tmp = json.loads(line)
            data.append({
                'text': tmp['text'],
                'spo_list': [(spo['subject'], spo['predicate'], spo['object']) for spo in tmp['spo_list']]
            })
    with open('../data/schemas.pkl', 'rb') as f:
        s2i = pickle.load(f)['s2i']
    texts = []
    sbj_starts = []
    sbj_ends = []
    sbj_bounds = []
    obj_starts = []
    obj_ends = []
    sbj_nums = 0
    for d in data:
        sbj_num = len(set([spo[0] for spo in d['spo_list']]))
        sbj_nums += sbj_num
        text = d['text']
        text_list = jieba.lcut(text, HMM=False)
        text_len = len(text_list)
        sbj_start = np.zeros(text_len).tolist()
        sbj_end = np.zeros(text_len).tolist()
        sbj_bound = {}
        obj_start = {}
        obj_end = {}
        for spo in d['spo_list']:
            sbj_s = -1
            sbj_e = -1
            sbj_list = jieba.lcut(spo[0], HMM=False)
            sbj_len = len(sbj_list)
            for i in range(0, text_len-sbj_len+1):
                if text_list[i: i+sbj_len] == sbj_list:
                    sbj_s = i
                    sbj_e = i + sbj_len - 1
                    break

            obj_s = -1
            obj_e = -1
            obj_list = jieba.lcut(spo[2], HMM=False)
            obj_len = len(obj_list)
            for i in range(0, text_len-obj_len+1):
                if text_list[i: i+obj_len] == obj_list:
                    obj_s = i
                    obj_e = i + obj_len - 1
                    break

            if sbj_s != -1 and sbj_e != -1 and obj_s != -1 and obj_e != -1:
                sbj_start[sbj_s] = 1
                sbj_end[sbj_e] = 1
                sbj_bound[spo[0]] = [sbj_s, sbj_e]
                if spo[0] not in obj_start:
                    obj_start[spo[0]] = np.zeros(text_len).tolist()
                    obj_end[spo[0]] = np.zeros(text_len).tolist()
                obj_start[spo[0]][obj_s] = s2i[spo[1]]
                obj_end[spo[0]][obj_e] = s2i[spo[1]]

        for sbj, sbj_index in sbj_bound.items():
            texts.append(text_list)
            sbj_starts.append(sbj_start)
            sbj_ends.append(sbj_end)
            sbj_bounds.append(sbj_index)
            obj_starts.append(obj_start[sbj])
            obj_ends.append(obj_end[sbj])

    print('spo make samples, nums:%d/%d, radio:%.4f' % (len(texts), sbj_nums, len(texts)/sbj_nums))
    return texts, sbj_starts, sbj_ends, sbj_bounds, obj_starts, obj_ends


def gen_test_data(file_path, get_answer, is_sbj=True):
    data = []
    result = []
    texts = []
    with open(file_path, 'r') as f:
        for line in f:
            tmp = json.loads(line)
            data.append(jieba.lcut(tmp['text'], HMM=False))
            texts.append(tmp['text'])
            if get_answer:
                if is_sbj:
                    result.append(set([spo['subject'] for spo in tmp['spo_list']]))
                else:
                    result.append([(spo['subject'], spo['predicate'], spo['object']) for spo in tmp['spo_list']])

    if get_answer:
        return data, result
    else:
        return data, texts


class MyDatasetSbj(Dataset):
    def __init__(self, file_path, is_train=True):
        super(Dataset, self).__init__()
        self.is_train = is_train
        if is_train:
            self.texts, self.sbj_starts, self.sbj_ends = gen_train_data_sbj(file_path)
            self.sbj_starts = padding(self.sbj_starts)
            self.sbj_ends = padding(self.sbj_ends)
        else:
            self.texts, _ = gen_test_data(file_path, False, True)

        self.texts = word2index(self.texts)
        self.texts = padding(self.texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        if self.is_train:
            return torch.LongTensor(self.texts[item]), torch.LongTensor(self.sbj_starts[item]),\
                    torch.LongTensor(self.sbj_ends[item])
        else:
            return torch.LongTensor(self.texts[item])


class MyDatasetSpo(Dataset):
    def __init__(self, file_path, is_train=True):
        super(Dataset, self).__init__()
        self.is_train = is_train
        if is_train:
            self.texts, self.sbj_starts, self.sbj_ends, self.sbj_bounds, self.obj_starts, self.obj_ends = \
                gen_train_data_spo(file_path)
            self.sbj_starts = padding(self.sbj_starts)
            self.sbj_ends = padding(self.sbj_ends)
            self.obj_starts = padding(self.obj_starts)
            self.obj_ends = padding(self.obj_ends)
        else:
            self.texts, _ = gen_test_data(file_path, False, False)

        self.texts = word2index(self.texts)
        self.texts = padding(self.texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        if self.is_train:
            return torch.LongTensor(self.texts[item]), torch.LongTensor(self.sbj_starts[item]),\
                    torch.LongTensor(self.sbj_ends[item]), torch.LongTensor(self.sbj_bounds[item]),\
                    torch.LongTensor(self.obj_starts[item]), torch.LongTensor(self.obj_ends[item])
        else:
            return torch.LongTensor(self.texts[item])


def build_loader(file_path, batch_size, shuffle, drop_last, is_train=True, is_sbj=True):
    if is_sbj:
        dataset = MyDatasetSbj(file_path, is_train=is_train)
    else:
        dataset = MyDatasetSpo(file_path, is_train=is_train)
    data_iter = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return data_iter


if __name__ == '__main__':
    # build_vocab_embedding()
    # get_dict_schemas()
    pass
