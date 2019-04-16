# coding = utf-8
# author = xy

import numpy as np
import json
import gensim
import pickle
from jieba import posseg
from torch.utils.data import Dataset, DataLoader
import torch


# type_dict
def get_type_dict_sbj():
    sbj_dict = {
        '书籍': '书籍',
        '图书作品': '书籍',
        '网络小说': '书籍',
        '企业': '机构',
        '机构': '机构',
        '学科专业': '学科专业',
        '历史人物': '人物',
        '人物': '人物',
        '地点': '地点',
        '景点': '景点',
        '国家': '国家',
        '行政区': '行政区',
        '生物': '生物',
        '歌曲': '歌曲',
        '电视综艺': '电视综艺',
        '影视作品': '影视作品'
    }
    return sbj_dict


def get_dict_schemas():
    sbj_dict = get_type_dict_sbj()
    sbjs = set()
    p_objs = set()
    with open('../data/all_50_schemas') as f:
        for line in f:
            tmp = json.loads(line)
            sbjs.add(sbj_dict[tmp['subject_type']])
            p_objs.add(tmp['predicate'])

    # sbj字典
    sbj2i = {}
    i2sbj = {}
    counts = 1
    for sbj in sbjs:
        sbj2i[sbj] = counts
        i2sbj[counts] = sbj
        counts += 1
        i2sbj[counts] = sbj
        counts += 1
        i2sbj[counts] = sbj
        counts += 1

    print(f'sbj2i_num:{len(sbj2i)}')
    sbj_dict = {'sbj2i': sbj2i, 'i2sbj': i2sbj}
    with open('../data/sbj_dict.pkl', 'wb') as f:
        pickle.dump(sbj_dict, f)

    # p_obj字典
    p2i = {}
    i2p = {}
    counts = 1
    for p_obj in p_objs:
        p2i[p_obj] = counts
        i2p[counts] = p_obj
        counts += 1
    print(f'p2i_num:{len(p2i)}')
    p_dict = {'p2i': p2i, 'i2p': i2p}
    with open('../data/p_dict.pkl', 'wb') as f:
        pickle.dump(p_dict, f)


def build_vocab_embedding():
    word_list = []
    tag_list = []
    datas = ['../data/train_data.json', '../data/dev_data.json', '../data/test1_data_postag.json']
    for data in datas:
        with open(data, 'r') as f:
            for line in f:
                tmp = json.loads(line)
                words, tags = list(zip(*posseg.lcut(tmp['text'], HMM=False)))
                word_list += list(words)
                tag_list += list(tags)

    # tag lang
    t2i = {'<pad>': 0}
    i2t = {0: '<pad>'}
    count = 1
    tag_set = set(tag_list)
    for tag in tag_set:
        t2i[tag] = count
        i2t[count] = tag
        count += 1
    t2i['<unk>'] = count
    i2t[count] = '<unk>'
    print(f'tag_num:{len(t2i)}')
    with open('../data/tag_lang.pkl', 'wb') as f:
        tag_lang = {'t2i': t2i, 'i2t': i2t}
        pickle.dump(tag_lang, f)

    # word lang
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


def tag2index(tag_lists):
    with open('../data/tag_lang.pkl', 'rb') as f:
        tag_lang = pickle.load(f)
        t2i = tag_lang['t2i']
    result = []
    for tags in tag_lists:
        tmp = [t2i[tag] if tag in t2i else t2i['<unk>'] for tag in tags]
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
    sbj_dict = get_type_dict_sbj()
    texts = []
    text_lists = []
    tag_lists = []
    spo_lists = []
    with open(file_path, 'r') as f:
        for line in f:
            tmp = json.loads(line)
            texts.append(tmp['text'])
            text_list, tag_list = list(zip(*posseg.lcut(tmp['text'], HMM=False)))
            text_list = list(text_list)
            tag_list = list(tag_list)
            text_lists.append(text_list)
            tag_lists.append(tag_list)
            spo_lists.append(tmp['spo_list'])

    with open('../data/sbj_dict.pkl', 'rb') as f:
        sbj2i = pickle.load(f)['sbj2i']

    r_text_lists = []
    r_tag_lists = []
    r_sbjs = []
    nums = 0
    for text_list, tag_list, spo_list in zip(text_lists, tag_lists, spo_lists):
        text_len = len(text_list)
        sbj = np.zeros(text_len)

        spo_extract = set()
        for spo in spo_list:
            sbj_list, _ = list(zip(*posseg.lcut(spo['subject'], HMM=False)))
            sbj_list = list(sbj_list)
            sbj_len = len(sbj_list)
            for i in range(0, text_len-sbj_len+1):
                if text_list[i: i+sbj_len] == sbj_list:
                    sbj_s = i
                    sbj_e = i + sbj_len - 1
                    xxx = sbj_dict[spo['subject_type']]
                    if sbj_s == sbj_e:
                        sbj[sbj_s] = sbj2i[xxx]
                    elif sbj_e - sbj_s == 1:
                        sbj[sbj_s] = sbj2i[xxx]
                        sbj[sbj_e] = sbj2i[xxx] + 2
                    elif sbj_e - sbj_s > 1:
                        sbj[sbj_s] = sbj2i[xxx]
                        sbj[sbj_s+1: sbj_e] = sbj2i[xxx] + 1
                        sbj[sbj_e] = sbj2i[xxx] + 2
                    else:
                        print('wrong')
                        assert 1 == -1
                    spo_extract.add(spo['subject'])
        nums += len(spo_extract)
        if len(spo_extract) != 0:
            r_text_lists.append(text_list)
            r_tag_lists.append(tag_list)
            r_sbjs.append(sbj.tolist())

    all_nums = 0
    for spo_list in spo_lists:
        all_nums += len(set([spo['subject'] for spo in spo_list]))

    print('sbj, make samples_num:%d, sbj_nums:%d/%d, radio:%.4f' % (len(r_text_lists), nums, all_nums, nums/all_nums))

    return r_text_lists, r_tag_lists, r_sbjs


def gen_train_data_spo(file_path):
    sbj_dict = get_type_dict_sbj()
    texts = []
    text_lists = []
    tag_lists = []
    spo_lists = []
    with open(file_path, 'r') as f:
        for line in f:
            tmp = json.loads(line)
            texts.append(tmp['text'])
            text_list, tag_list = list(zip(*posseg.lcut(tmp['text'], HMM=False)))
            text_list = list(text_list)
            tag_list = list(tag_list)
            text_lists.append(text_list)
            tag_lists.append(tag_list)
            spo_lists.append(tmp['spo_list'])

    with open('../data/sbj_dict.pkl', 'rb') as f:
        sbj2i = pickle.load(f)['sbj2i']
    with open('../data/p_dict.pkl', 'rb') as f:
        p2i = pickle.load(f)['p2i']

    r_text_lists = []
    r_tag_lists = []
    r_sbjs = []
    r_sbj_bounds = []
    r_obj_starts = []
    r_obj_ends = []
    for text_list, tag_list, spo_list in zip(text_lists, tag_lists, spo_lists):
        text_len = len(text_list)
        sbj = np.zeros(text_len)
        sbj_bound = {}
        obj_start = {}
        obj_end = {}
        for spo in spo_list:
            sbj_list, _ = list(zip(*posseg.lcut(spo['subject'], HMM=False)))
            sbj_list = list(sbj_list)
            sbj_len = len(sbj_list)
            sbj_tmp = []
            for i in range(0, text_len-sbj_len+1):
                if text_list[i: i+sbj_len] == sbj_list:
                    sbj_s = i
                    sbj_e = i + sbj_len - 1
                    sbj_tmp.append([sbj_s, sbj_e])
                    xxx = sbj_dict[spo['subject_type']]
                    if sbj_s == sbj_e:
                        sbj[sbj_s] = sbj2i[xxx]
                    elif sbj_e - sbj_s == 1:
                        sbj[sbj_s] = sbj2i[xxx]
                        sbj[sbj_e] = sbj2i[xxx] + 2
                    elif sbj_e - sbj_s > 1:
                        sbj[sbj_s] = sbj2i[xxx]
                        sbj[sbj_s+1: sbj_e] = sbj2i[xxx] + 1
                        sbj[sbj_e] = sbj2i[xxx] + 2
                    else:
                        print('wrong')
                        assert 1 == -1

            obj_list, _ = list(zip(*posseg.lcut(spo['object'], HMM=False)))
            obj_list = list(obj_list)
            obj_len = len(obj_list)
            obj_start_tmp = np.zeros(text_len)
            obj_end_tmp = np.zeros(text_len)
            for i in range(0, text_len-obj_len+1):
                if text_list[i: i+obj_len] == obj_list:
                    obj_s = i
                    obj_e = i + obj_len - 1
                    obj_start_tmp[obj_s] = p2i[spo['predicate']]
                    obj_end_tmp[obj_e] = p2i[spo['predicate']]

            if len(sbj_tmp) != 0 and obj_start_tmp.sum() != 0:
                if spo['subject'] in sbj_bound:
                    sbj_bound[spo['subject']] += sbj_tmp
                    obj_start[spo['subject']] += obj_start_tmp
                    obj_end[spo['subject']] += obj_end_tmp
                else:
                    sbj_bound[spo['subject']] = sbj_tmp
                    obj_start[spo['subject']] = obj_start_tmp
                    obj_end[spo['subject']] = obj_end_tmp

        for sbj_i, sbj_index in sbj_bound.items():
            r_text_lists.append(text_list)
            r_tag_lists.append(tag_list)
            r_sbjs.append(sbj.tolist())
            if len(sbj_index) == 1:
                r_sbj_bounds.append(sbj_index[0])
            else:
                xx = np.random.choice(range(len(sbj_index)))
                r_sbj_bounds.append(sbj_index[xx])
            r_obj_starts.append(obj_start[sbj_i].tolist())
            r_obj_ends.append(obj_end[sbj_i].tolist())

    sbj_nums = 0
    for spo_list in spo_lists:
        sbj_nums += len(set([spo['subject'] for spo in spo_list]))
    print('spo make samples, nums:%d/%d, radio:%.4f' % (len(r_text_lists), sbj_nums, len(r_text_lists)/sbj_nums))

    return r_text_lists, r_tag_lists, r_sbjs, r_sbj_bounds, r_obj_starts, r_obj_ends


def gen_test_data(file_path, get_answer, is_sbj=True):
    data = []
    tags = []
    result = []
    texts = []
    with open(file_path, 'r') as f:
        for line in f:
            tmp = json.loads(line)
            word_list, tag_list = list(zip(*posseg.lcut(tmp['text'], HMM=False)))
            data.append(list(word_list))
            tags.append(list(tag_list))
            texts.append(tmp['text'])
            if get_answer:
                if is_sbj:
                    result.append(set([spo['subject'] for spo in tmp['spo_list']]))
                else:
                    result.append([(spo['subject'], spo['predicate'], spo['object']) for spo in tmp['spo_list']])

    if get_answer:
        return data, tags, result
    else:
        return data, tags, texts


class MyDatasetSbj(Dataset):
    def __init__(self, file_path, is_train=True):
        super(Dataset, self).__init__()
        self.is_train = is_train
        if is_train:
            self.texts, self.tags, self.sbjs = gen_train_data_sbj(file_path)
            self.sbjs = padding(self.sbjs)
        else:
            self.texts, self.tags, _ = gen_test_data(file_path, False, True)

        self.texts = word2index(self.texts)
        self.texts = padding(self.texts)
        self.tags = tag2index(self.tags)
        self.tags = padding(self.tags)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        if self.is_train:
            return torch.LongTensor(self.texts[item]), torch.LongTensor(self.tags[item]), \
                    torch.LongTensor(self.sbjs[item])
        else:
            return torch.LongTensor(self.texts[item]), torch.LongTensor(self.tags[item])


class MyDatasetSpo(Dataset):
    def __init__(self, file_path, is_train=True):
        super(Dataset, self).__init__()
        self.is_train = is_train
        if is_train:
            self.texts, self.tags, self.sbjs, self.sbj_bounds, self.obj_starts, self.obj_ends = \
                gen_train_data_spo(file_path)
            self.sbjs = padding(self.sbjs)
            self.obj_starts = padding(self.obj_starts)
            self.obj_ends = padding(self.obj_ends)
        else:
            self.texts, self.tags, _ = gen_test_data(file_path, False, False)

        self.texts = word2index(self.texts)
        self.texts = padding(self.texts)
        self.tags = tag2index(self.tags)
        self.tags = padding(self.tags)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        if self.is_train:
            return torch.LongTensor(self.texts[item]), torch.LongTensor(self.tags[item]),\
                   torch.LongTensor(self.sbjs[item]), torch.LongTensor(self.sbj_bounds[item]),\
                   torch.LongTensor(self.obj_starts[item]), torch.LongTensor(self.obj_ends[item])
        else:
            return torch.LongTensor(self.texts[item]), torch.LongTensor(self.tags[item])


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
    build_vocab_embedding()
    # get_dict_schemas()
    # pass
