# coding = utf-8
# author = xy

import numpy as np
import json
import gensim
import pickle
import jieba
import random
from jieba import posseg
from torch.utils.data import Dataset, DataLoader
import torch


# with open('../data/jieba_vocab.pkl', 'rb') as f:
#     xxx = pickle.load(f)
# for x in xxx:
#     jieba.del_word(x)


def is_en(c):
    if 'A' <= c <= 'z':
        return True
    else:
        return False


def split_word(word):
    r = []
    tmp = []
    for i in range(len(word)):
        c = word[i]
        if is_en(c):
            tmp.append(c)
            if i == len(word) - 1:
                r.append(''.join(tmp))
        else:
            if len(tmp) == 0:
                r.append(c)
            else:
                r.append(''.join(tmp))
                r.append(c)
                tmp = []
    return r


def get_dict_schemas():
    sbjs = set()
    objs = set()
    ps = set()
    with open('../data/all_50_schemas') as f:
        for line in f:
            tmp = json.loads(line)
            sbjs.add(tmp['subject_type'])
            objs.add(tmp['object_type'])
            ps.add(tmp['predicate'])

    # sbj字典
    sbj2i = {}
    i2sbj = {}
    counts = 1
    for sbj in sbjs:
        sbj2i[sbj] = counts
        i2sbj[counts] = sbj
        counts += 1

    # p字典
    p2i = {}
    i2p = {}
    counts = 0
    for p in ps:
        p2i[p] = counts
        i2p[counts] = p
        counts += 1

    # obj字典
    obj2i = {}
    i2obj = {}
    counts = 1
    for obj in objs:
        obj2i[obj] = counts
        i2obj[counts] = obj
        counts += 1

    # save
    print(f'sbj2i_num:{len(sbj2i)}')
    sbj_dict = {'sbj2i': sbj2i, 'i2sbj': i2sbj}
    with open('../data/sbj_dict.pkl', 'wb') as f:
        pickle.dump(sbj_dict, f)

    print(f'p2i_num:{len(p2i)}')
    p_dict = {'p2i': p2i, 'i2p': i2p}
    with open('../data/p_dict.pkl', 'wb') as f:
        pickle.dump(p_dict, f)

    print(f'obj2i_num:{len(obj2i)}')
    obj_dict = {'obj2i': obj2i, 'i2obj': i2obj}
    with open('../data/obj_dict.pkl', 'wb') as f:
        pickle.dump(obj_dict, f)


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


def build_char():
    char2num = {}
    # datas = ['../data/train_data.json', '../data/dev_data.json', '../data/test1_data_postag.json']
    datas = ['../data/train_data.json']
    for data in datas:
        with open(data, 'r') as f:
            for line in f:
                tmp = json.loads(line)
                text = tmp['text'].lower().strip()
                text_list = split_word(text)
                for char in text_list:
                    char2num[char] = char2num.get(char, 0) + 1
    char2i = {'<pad>': 0, '<unk>': 1}
    i2char = {0: '<pad>', 1: '<unk>'}
    counts = 2
    for char in char2num:
        if char2num[char] >= 2:
            char2i[char] = counts
            i2char[counts] = char
            counts += 1
    print(f'char_num:{len(char2i)}')

    with open('../data/char_dict.pkl', 'wb') as f:
        char_dict = {'char2i': char2i, 'i2char': i2char}
        pickle.dump(char_dict, f)


def word2index(word_lists):
    with open('../data/lang.pkl', 'rb') as f:
        lang = pickle.load(f)
        w2i = lang['w2i']
    result = []
    for words in word_lists:
        tmp = [w2i[word] if word in w2i else w2i['<unk>'] for word in words]
        result.append(tmp)
    return result


def char2index(char_lists):
    with open('../data/char_dict.pkl', 'rb') as f:
        char2i = pickle.load(f)['char2i']
    result = []
    for chars in char_lists:
        tmp = [char2i[char] if char in char2i else char2i['<unk>'] for char in chars]
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


def padding_three_d(tensors):
    lens = [len(tensor) for tensor in tensors]
    max_len = max(lens)
    result = []
    dim = len(tensors[0][0])
    fill = [-9 for _ in range(dim)]
    for tensor in tensors:
        tensor += [fill for _ in range(max_len-len(tensor))]
        result.append(tensor)
    return result, max_len


def gen_train_data_ner(file_path):
    texts = []
    char_lists = []
    spo_lists = []
    with open(file_path, 'r') as f:
        for line in f:
            tmp = json.loads(line)
            text = tmp['text'].lower().strip()
            texts.append(text)
            char_list = split_word(text)
            char_lists.append(char_list)
            spo_lists.append(tmp['spo_list'])

    r_texts = []
    r_char_lists = []
    r_ners = []
    for text, char_list, spo_list in zip(texts, char_lists, spo_lists):
        char_len = len(char_list)
        ner = np.zeros(char_len)

        spo_extract = set()
        sbj_set = set([spo['subject'].lower().strip() for spo in spo_list])
        obj_set = set([spo['object'].lower().strip() for spo in spo_list])
        ner_set = sbj_set | obj_set
        ner_set = list(ner_set)
        np.random.shuffle(ner_set)

        for ner_i in ner_set:
            ner_list = split_word(ner_i)
            ner_len = len(ner_list)
            for i in range(0, char_len-ner_len+1):
                if char_list[i: i+ner_len] == ner_list:
                    ner_s = i
                    ner_e = i + ner_len - 1

                    flag = False
                    for item in ner[ner_s: ner_e+1]:
                        if item != 0:
                            flag = True
                            break
                    if flag:
                        continue

                    if ner_s == ner_e:
                        ner[ner_s] = 4
                    elif ner_e - ner_s == 1:
                        ner[ner_s] = 1
                        ner[ner_e] = 3
                    elif ner_e - ner_s > 1:
                        ner[ner_s] = 1
                        ner[ner_s+1: ner_e] = 2
                        ner[ner_e] = 3
                    else:
                        print('wrong')
                        assert 1 == -1
                    spo_extract.add(ner_i)

        if len(spo_extract) != 0:
            r_texts.append(text)
            r_char_lists.append(char_list)
            r_ners.append(ner.tolist())

    print('ner, make samples_num:%d/%d, radio:%.4f' % (len(r_texts), len(texts), len(r_texts)/len(texts)))

    return r_char_lists, r_ners


def gen_train_data_p(file_path, ner_file):
    texts = []
    char_lists = []
    spo_lists = []
    with open(file_path, 'r') as f:
        for line in f:
            tmp = json.loads(line)
            texts.append(tmp['text'])
            text = tmp['text'].lower().strip()
            char_list = split_word(text)
            char_lists.append(char_list)
            spo_lists.append(tmp['spo_list'])

    ner_file = '../data/' + ner_file
    with open(ner_file, 'rb') as f:
        ner_gens = pickle.load(f)
    assert len(texts) == len(ner_gens)

    with open('../data/p_dict.pkl', 'rb') as f:
        p2i = pickle.load(f)['p2i']

    r_char_lists = []
    r_sbj_bounds = []
    r_obj_bounds = []
    r_ps = []
    max_match_nums = 0
    for char_list, ner_gen, spo_list in zip(char_lists, ner_gens, spo_lists):
        char_len = len(char_list)
        ner_bound_dict = {}
        ner_extract = set()
        ner_set = ner_gen

        # ner
        for ner in ner_set:
            ner_list = split_word(ner)
            ner_len = len(ner_list)
            for i in range(0, char_len-ner_len+1):
                if char_list[i: i+ner_len] == ner_list:
                    ner_s = i
                    ner_e = i + ner_len - 1
                    ner_extract.add(ner)
                    if ner in ner_bound_dict:
                        ner_bound_dict[ner].add((ner_s, ner_e))
                    else:
                        ner_bound_dict[ner] = set([(ner_s, ner_e)])

        # p
        sbj_obj_dict = {}
        for spo in spo_list:
            if (spo['subject'].lower().strip(), spo['object'].lower().strip()) in sbj_obj_dict:
                sbj_obj_dict[(spo['subject'].lower().strip(), spo['object'].lower().strip())].append(spo['predicate'])
            else:
                sbj_obj_dict[(spo['subject'].lower().strip(), spo['object'].lower().strip())] = [spo['predicate']]

        sbj_bounds = []
        obj_bounds = []
        ps = []
        for s in ner_extract:
            for o in ner_extract:
                if s == o:
                    continue
                s_bound = list(ner_bound_dict[s])
                o_bound = list(ner_bound_dict[o])
                p = [0 for _ in range(49)]
                if (s, o) in sbj_obj_dict:
                    ppp = sbj_obj_dict[(s, o)]
                    for pp in ppp:
                        p[p2i[pp]] = 1
                sbj_bounds.append(s_bound)
                obj_bounds.append(o_bound)
                ps.append(p)
                max_match_nums = max(max_match_nums, len(s_bound))
                max_match_nums = max(max_match_nums, len(o_bound))

        if len(ner_extract) > 1:
            r_char_lists.append(char_list)
            r_sbj_bounds.append(sbj_bounds)
            r_obj_bounds.append(obj_bounds)
            r_ps.append(ps)

    print('p, making sample_num:%d/%d, radio:%.4f' % (len(r_char_lists), len(char_lists), len(r_char_lists)/len(char_lists)))

    return r_char_lists, r_sbj_bounds, r_obj_bounds, r_ps, max_match_nums


def gen_test_data_p(file_path, ner_file):
    texts = []
    char_lists = []
    with open(file_path, 'r') as f:
        for line in f:
            tmp = json.loads(line)
            texts.append(tmp['text'])
            text = tmp['text'].lower().strip()
            char_list = split_word(text)
            char_lists.append(char_list)

    ner_file = '../data/' + ner_file + '.pkl'
    with open(ner_file, 'rb') as f:
        ner_gens = pickle.load(f)
    assert len(texts) == len(ner_gens)

    r_char_lists = []
    r_sbj_bounds = []
    r_obj_bounds = []
    max_match_nums = 0
    max_len = 0
    for char_list, ner_gen in zip(char_lists, ner_gens):
        char_len = len(char_list)
        ner_bound_dict = {}
        ner_extract = set()
        ner_set = ner_gen

        # ner
        for ner in ner_set:
            ner_list = split_word(ner)
            ner_len = len(ner_list)
            for i in range(0, char_len-ner_len+1):
                if char_list[i: i+ner_len] == ner_list:
                    ner_s = i
                    ner_e = i + ner_len - 1
                    ner_extract.add(ner)
                    if ner in ner_bound_dict:
                        ner_bound_dict[ner].add((ner_s, ner_e))
                    else:
                        ner_bound_dict[ner] = set([(ner_s, ner_e)])

        sbj_bounds = []
        obj_bounds = []
        for s in ner_extract:
            for o in ner_extract:
                if s == o:
                    continue
                s_bound = list(ner_bound_dict[s])
                o_bound = list(ner_bound_dict[o])
                sbj_bounds.append(s_bound)
                obj_bounds.append(o_bound)

                max_match_nums = max(max_match_nums, len(s_bound))
                max_match_nums = max(max_match_nums, len(o_bound))
        max_len = max(max_len, len(sbj_bounds))

        r_char_lists.append(char_list)
        if len(ner_set) > 1:
            r_sbj_bounds.append(sbj_bounds)
            r_obj_bounds.append(obj_bounds)
        else:
            r_sbj_bounds.append([[(0, 0)]])
            r_obj_bounds.append([[(0, 0)]])

    return r_char_lists, r_sbj_bounds, r_obj_bounds, max_match_nums, max_len


def gen_train_data_join(file_path):
    texts = []
    char_lists = []
    spo_lists = []
    with open(file_path, 'r') as f:
        for line in f:
            tmp = json.loads(line)
            texts.append(tmp['text'])
            text = tmp['text'].lower().strip()
            char_list = split_word(text)
            char_lists.append(char_list)
            spo_lists.append(tmp['spo_list'])

    with open('../data/sbj_dict.pkl', 'rb') as f:
        sbj_type_list = list(pickle.load(f)['sbj2i'].keys())

    with open('../data/obj_dict.pkl', 'rb') as f:
        obj_type_list = list(pickle.load(f)['obj2i'].keys())

    with open('../data/p_dict.pkl', 'rb') as f:
        p2i = pickle.load(f)['p2i']

    r_char_lists = []
    r_sbjs = []
    r_objs = []
    r_sbj_bounds = []
    r_obj_bounds = []
    r_ps = []
    for char_list, spo_list in zip(char_lists, spo_lists):
        char_len = len(char_list)
        sbj = np.zeros(char_len)
        obj = np.zeros(char_len)

        sbj_extract = set()
        obj_extract = set()

        sbj_set = set()
        for spo in spo_list:
            s = spo['subject'].lower().strip()
            sbj_set.add(s)
            if spo['object_type'] in sbj_type_list:
                o = spo['object'].lower().strip()
                sbj_set.add(o)

        obj_set = set()
        for spo in spo_list:
            o = spo['object'].lower().strip()
            obj_set.add(o)
            if spo['subject_type'] in obj_type_list:
                s = spo['subject'].lower().strip()
                obj_set.add(s)

        sbj_set = list(sbj_set)
        np.random.shuffle(sbj_set)
        obj_set = list(obj_set)
        np.random.shuffle(obj_set)

        sbj_bound_dict = {}
        obj_bound_dict = {}

        # sbj
        for sbj_item in sbj_set:
            sbj_list = split_word(sbj_item)
            sbj_len = len(sbj_list)
            for i in range(0, char_len-sbj_len+1):
                if char_list[i: i+sbj_len] == sbj_list:
                    sbj_s = i
                    sbj_e = i + sbj_len - 1

                    flag = False
                    for item in sbj[sbj_s: sbj_e+1]:
                        if item != 0:
                            flag = True
                            break
                    if flag:
                        continue

                    if sbj_s == sbj_e:
                        sbj[sbj_s] = 4
                    elif sbj_e - sbj_s == 1:
                        sbj[sbj_s] = 1
                        sbj[sbj_e] = 3
                    elif sbj_e - sbj_s > 1:
                        sbj[sbj_s] = 1
                        sbj[sbj_s+1: sbj_e] = 2
                        sbj[sbj_e] = 3
                    else:
                        print('wrong')
                        assert 1 == -1

                    sbj_extract.add(sbj_item)

                    if sbj_item in sbj_bound_dict:
                        sbj_bound_dict[sbj_item].add((sbj_s, sbj_e))
                    else:
                        sbj_bound_dict[sbj_item] = set([(sbj_s, sbj_e)])

        # obj
        for obj_item in obj_set:
            obj_list = split_word(obj_item)
            obj_len = len(obj_list)
            for i in range(0, char_len-obj_len+1):
                if char_list[i: i+obj_len] == obj_list:
                    obj_s = i
                    obj_e = i + obj_len - 1

                    flag = False
                    for item in obj[obj_s: obj_e+1]:
                        if item != 0:
                            flag = True
                            break
                    if flag:
                        continue

                    if obj_s == obj_e:
                        obj[obj_s] = 4
                    elif obj_e - obj_s == 1:
                        obj[obj_s] = 1
                        obj[obj_e] = 3
                    elif obj_e - obj_s > 1:
                        obj[obj_s] = 1
                        obj[obj_s+1: obj_e] = 2
                        obj[obj_e] = 3
                    else:
                        print('wrong')
                        assert 1 == -1

                    obj_extract.add(obj_item)

                    if obj_item in obj_bound_dict:
                        obj_bound_dict[obj_item].add((obj_s, obj_e))
                    else:
                        obj_bound_dict[obj_item] = set([(obj_s, obj_e)])

        # p
        sbj_obj_dict = {}
        for spo in spo_list:
            if (spo['subject'].lower().strip(), spo['object'].lower().strip()) in sbj_obj_dict:
                sbj_obj_dict[(spo['subject'].lower().strip(), spo['object'].lower().strip())].append(spo['predicate'])
            else:
                sbj_obj_dict[(spo['subject'].lower().strip(), spo['object'].lower().strip())] = [spo['predicate']]

        if len(sbj_extract) == len(sbj_set) and len(obj_extract) == len(obj_set):
            sbj_bounds = []
            obj_bounds = []
            ps = []
            for so_item, p_list in sbj_obj_dict.items():
                s, o = so_item
                s_bound = list(sbj_bound_dict[s])
                o_bound = list(obj_bound_dict[o])
                p = [0 for _ in range(49)]
                for pp in p_list:
                    p[p2i[pp]] = 1
                for s_index in s_bound:
                    for o_index in o_bound:
                        sbj_bounds.append(s_index)
                        obj_bounds.append(o_index)
                        ps.append(p)
            r_char_lists.append(char_list)
            r_sbjs.append(sbj.tolist())
            r_objs.append(obj.tolist())
            r_sbj_bounds.append(sbj_bounds)
            r_obj_bounds.append(obj_bounds)
            r_ps.append(ps)

    print('p, making sample_num:%d/%d, radio:%.4f' % (len(r_char_lists), len(char_lists), len(r_char_lists)/len(char_lists)))

    return r_char_lists, r_sbjs, r_objs, r_sbj_bounds, r_obj_bounds, r_ps


def gen_test_data(file_path, get_answer, task):
    char_lists = []
    result = []
    texts = []
    with open(file_path, 'r') as f:
        for line in f:
            tmp = json.loads(line)
            char_list = split_word(tmp['text'].lower().strip())
            char_lists.append(char_list)
            texts.append(tmp['text'])
            if get_answer:
                if task == 'ner':
                    result.append(set([spo['subject'].lower().strip() for spo in tmp['spo_list']]) |
                                  set([spo['object'].lower().strip() for spo in tmp['spo_list']]))
                elif task == 'p':
                    result.append([(spo['subject'].lower().strip(), spo['predicate'], spo['object'].lower().strip())
                                   for spo in tmp['spo_list']])
                else:
                    print('wrong')
                    assert 1 == -1

    if get_answer:
        return char_lists, result
    else:
        return char_lists, texts


class MyDatasetNer(Dataset):
    def __init__(self, file_path, is_train=True, task=None):
        super(Dataset, self).__init__()
        self.is_train = is_train
        if is_train:
            self.chars, self.ners = gen_train_data_ner(file_path)
            self.ners = padding(self.ners)
        else:
            self.chars, _ = gen_test_data(file_path, False, task)

        self.chars = char2index(self.chars)
        self.chars = padding(self.chars)

    def __len__(self):
        return len(self.chars)

    def __getitem__(self, item):
        if self.is_train:
            item_1 = torch.LongTensor(self.chars[item])
            item_2 = torch.LongTensor(self.ners[item])

            return item_1, item_2
        else:
            item_1 = torch.LongTensor(self.chars[item])
            return item_1, torch.Tensor([1, 2])


class MyDatasetP(Dataset):
    def __init__(self, file_path, is_train=True, ner_file=None):
        super(Dataset, self).__init__()
        self.is_train = is_train
        if is_train:
            self.chars, self.sbj_bounds, self.obj_bounds, self.ps, self.max_matchs = gen_train_data_p(file_path, ner_file)
            self.ps, self.item_len = padding_three_d(self.ps)

        else:
            self.chars, self.sbj_bounds, self.obj_bounds, self.max_matchs, self.item_len = gen_test_data_p(file_path, ner_file)

        self.chars = char2index(self.chars)
        self.chars = padding(self.chars)

    def __len__(self):
        return len(self.chars)

    def __getitem__(self, item):
        if self.is_train:
            # char_list
            item_1 = torch.LongTensor(self.chars[item])

            # sbj_bound
            sbj_bound = self.sbj_bounds[item]
            sbj_bound_tmp = []
            fill = (-9, -9)
            for s_bound in sbj_bound:
                fill_bound = [fill for _ in range(self.max_matchs - len(s_bound))]
                sbj_bound_tmp.append(s_bound + fill_bound)
            fill = [fill for _ in range(self.max_matchs)]
            sbj_bound_tmp += [fill for _ in range(self.item_len-len(sbj_bound_tmp))]
            item_2 = torch.LongTensor(sbj_bound_tmp)

            # obj_bound
            obj_bound = self.obj_bounds[item]
            obj_bound_tmp = []
            fill = (-9, -9)
            for o_bound in obj_bound:
                fill_bound = [fill for _ in range(self.max_matchs - len(o_bound))]
                obj_bound_tmp.append(o_bound + fill_bound)
            fill = [fill for _ in range(self.max_matchs)]
            obj_bound_tmp += [fill for _ in range(self.item_len-len(obj_bound_tmp))]
            item_3 = torch.LongTensor(obj_bound_tmp)

            # p
            item_4 = torch.LongTensor(self.ps[item])

            return item_1, item_2, item_3, item_4

        else:
            # char_list
            item_1 = torch.LongTensor(self.chars[item])

            # sbj_bound
            sbj_bound = self.sbj_bounds[item]
            sbj_bound_tmp = []
            fill = (-9, -9)
            for s_bound in sbj_bound:
                fill_bound = [fill for _ in range(self.max_matchs - len(s_bound))]
                sbj_bound_tmp.append(s_bound + fill_bound)
            fill = [fill for _ in range(self.max_matchs)]
            sbj_bound_tmp += [fill for _ in range(self.item_len-len(sbj_bound_tmp))]
            item_2 = torch.LongTensor(sbj_bound_tmp)

            # obj_bound
            obj_bound = self.obj_bounds[item]
            obj_bound_tmp = []
            fill = (-9, -9)
            for o_bound in obj_bound:
                fill_bound = [fill for _ in range(self.max_matchs - len(o_bound))]
                obj_bound_tmp.append(o_bound + fill_bound)
            fill = [fill for _ in range(self.max_matchs)]
            obj_bound_tmp += [fill for _ in range(self.item_len-len(obj_bound_tmp))]
            item_3 = torch.LongTensor(obj_bound_tmp)

            return item_1, item_2, item_3


class MyDatasetJoin(Dataset):
    def __init__(self, file_path, is_train=True):
        super(Dataset, self).__init__()
        self.is_train = is_train
        if is_train:
            self.chars, self.sbjs, self.objs, self.sbj_bounds, self.obj_bounds, self.ps = gen_train_data_join(file_path)
            self.sbjs = padding(self.sbjs)
            self.objs = padding(self.objs)
            self.sbj_bounds, _ = padding_three_d(self.sbj_bounds)
            self.obj_bounds, _ = padding_three_d(self.obj_bounds)
            self.ps, _ = padding_three_d(self.ps)
        else:
            self.chars, _ = gen_test_data(file_path, False, 'p')

        self.chars = char2index(self.chars)
        self.chars = padding(self.chars)

    def __len__(self):
        return len(self.chars)

    def __getitem__(self, item):
        if self.is_train:
            item1 = torch.LongTensor(self.chars[item])
            item2 = torch.LongTensor(self.sbjs[item])
            item3 = torch.LongTensor(self.objs[item])
            item4 = torch.LongTensor(self.sbj_bounds[item])
            item5 = torch.LongTensor(self.obj_bounds[item])
            item6 = torch.LongTensor(self.ps[item])

            return item1, item2, item3, item4, item5, item6

        else:
            item1 = torch.LongTensor(self.chars[item])

            return item1, torch.LongTensor([1,2,3])


def build_loader(file_path, batch_size, shuffle, drop_last, is_train=True, task=None, ner_file=None):
    if task == 'p':
        dataset = MyDatasetP(file_path, is_train=is_train, ner_file=ner_file)
    elif task == 'ner':
        dataset = MyDatasetNer(file_path, is_train=is_train, task=task)
    elif task == 'join':
        dataset = MyDatasetJoin(file_path, is_train=is_train)
    else:
        print('wrong task')
        assert 1 == -1

    data_iter = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return data_iter


if __name__ == '__main__':
    # build_vocab_embedding()
    # build_char()
    get_dict_schemas()
    # pass
