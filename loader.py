# coding = utf-8
# author = xy

import numpy as np
import json
import gensim
import pickle
import jieba
from jieba import posseg
from torch.utils.data import Dataset, DataLoader
import torch


with open('../data/jieba_vocab.pkl', 'rb') as f:
    xxx = pickle.load(f)
for x in xxx:
    jieba.del_word(x)


def get_dict_schemas():
    sbjs = set()
    ps = set()
    objs = set()
    with open('../data/all_50_schemas') as f:
        for line in f:
            tmp = json.loads(line)
            sbjs.add(tmp['subject_type'])
            ps.add(tmp['predicate'])
            objs.add(tmp['object_type'])

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
        i2obj[counts] = obj
        counts += 1
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


def padding_three_d(tensors):
    lens = [len(tensor) for tensor in tensors]
    max_len = max(lens)
    result = []
    dim = len(tensors[0][0])
    fill = [-9 for _ in range(dim)]
    for tensor in tensors:
        tensor += [fill for _ in range(max_len-len(tensor))]
        result.append(tensor)
    return result


def gen_train_data_ner(file_path, task):
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

    if task == 'sbj':
        ner1 = 'subject'
    elif task == 'obj':
        ner1 = 'object'
    else:
        assert 1 == -1

    r_text_lists = []
    r_tag_lists = []
    r_ners = []
    nums = 0
    for text_list, tag_list, spo_list in zip(text_lists, tag_lists, spo_lists):
        text_len = len(text_list)
        ner = np.zeros(text_len)

        spo_extract = set()
        ner_set = set([spo[ner1] for spo in spo_list])
        ner_set = np.array(list(ner_set))
        np.random.shuffle(ner_set)
        for ner_i in ner_set:
            ner_list, _ = list(zip(*posseg.lcut(ner_i, HMM=False)))
            ner_list = list(ner_list)
            ner_len = len(ner_list)
            for i in range(0, text_len-ner_len+1):
                if text_list[i: i+ner_len] == ner_list:
                    ner_s = i
                    ner_e = i + ner_len - 1
                    flag = False
                    for item in [1, 2, 3, 4]:
                        if item in ner[ner_s: ner_e+1]:
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

        nums += len(spo_extract)
        if len(spo_extract) != 0:
            r_text_lists.append(text_list)
            r_tag_lists.append(tag_list)
            r_ners.append(ner.tolist())

    all_nums = 0
    for spo_list in spo_lists:
        all_nums += len(set([spo[ner1] for spo in spo_list]))

    print('%s, make samples_num:%d, sbj_nums:%d/%d, radio:%.4f' % (task, len(r_text_lists), nums, all_nums, nums/all_nums))

    return r_text_lists, r_tag_lists, r_ners


def gen_train_data_p(file_path):
    texts = []
    text_lists = []
    tag_lists = []
    spo_lists = []
    # nums = 0
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

            # nums += 1
            # if nums == 1000:
            #     break

    with open('../data/p_dict.pkl', 'rb') as f:
        p2i = pickle.load(f)['p2i']

    r_text_lists = []
    r_tag_lists = []
    r_sbjs = []
    r_objs = []
    r_sbj_bounds = []
    r_obj_bounds = []
    r_ps = []

    for text_list, tag_list, spo_list in zip(text_lists, tag_lists, spo_lists):
        text_len = len(text_list)
        sbj = np.zeros(text_len)
        obj = np.zeros(text_len)
        sbj_bound_dict = {}
        obj_bound_dict = {}

        sbj_extract = set()
        obj_extract = set()
        for spo in spo_list:
            sbj_list, _ = list(zip(*posseg.lcut(spo['subject'], HMM=False)))
            sbj_list = list(sbj_list)
            sbj_len = len(sbj_list)
            for i in range(0, text_len-sbj_len+1):
                if text_list[i: i+sbj_len] == sbj_list:
                    sbj_s = i
                    sbj_e = i + sbj_len - 1
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
                    sbj_extract.add(spo['subject'])
                    if spo['subject'] in sbj_bound_dict:
                        sbj_bound_dict[spo['subject']].add((sbj_s, sbj_e))
                    else:
                        sbj_bound_dict[spo['subject']] = set([(sbj_s, sbj_e)])

            obj_list, _ = list(zip(*posseg.lcut(spo['object'], HMM=False)))
            obj_list = list(obj_list)
            obj_len = len(obj_list)
            for i in range(0, text_len-obj_len+1):
                if text_list[i: i+obj_len] == obj_list:
                    obj_s = i
                    obj_e = i + obj_len - 1
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
                    obj_extract.add(spo['object'])
                    if spo['object'] in obj_bound_dict:
                        obj_bound_dict[spo['object']].add((obj_s, obj_e))
                    else:
                        obj_bound_dict[spo['object']] = set([(obj_s, obj_e)])

        sbj_obj_dict = {}
        for spo in spo_list:
            if (spo['subject'], spo['object']) in sbj_obj_dict:
                sbj_obj_dict[(spo['subject'], spo['object'])].append(spo['predicate'])
            else:
                sbj_obj_dict[(spo['subject'], spo['object'])] = [spo['predicate']]

        sbj_bounds = []
        obj_bounds = []
        ps = []
        for s in sbj_extract:
            for o in obj_extract:
                index = np.random.choice(range(len(sbj_bound_dict[s])))
                s_bound = list(sbj_bound_dict[s])[index]
                index = np.random.choice(range(len(obj_bound_dict[o])))
                o_bound = list(obj_bound_dict[o])[index]
                p = np.zeros(49).tolist()
                if (s, o) in sbj_obj_dict:
                    ppp = sbj_obj_dict[(s, o)]
                    for pp in ppp:
                        p[p2i[pp]] = 1
                sbj_bounds.append(s_bound)
                obj_bounds.append(o_bound)
                ps.append(p)

        if len(sbj_extract) != 0 and len(obj_extract) != 0:
            r_text_lists.append(text_list)
            r_tag_lists.append(tag_list)
            r_sbjs.append(sbj.tolist())
            r_objs.append(obj.tolist())
            r_sbj_bounds.append(sbj_bounds)
            r_obj_bounds.append(obj_bounds)
            r_ps.append(ps)

    print('p, making sample_num:%d/%d, radio:%.4f' % (len(r_text_lists), len(text_lists), len(r_text_lists)/len(text_lists)))

    return r_text_lists, r_tag_lists, r_sbjs, r_objs, r_sbj_bounds, r_obj_bounds, r_ps


def gen_test_data(file_path, get_answer, task):
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
                if task == 'sbj':
                    result.append(set([spo['subject'] for spo in tmp['spo_list']]))
                elif task == 'obj':
                    result.append(set([spo['object'] for spo in tmp['spo_list']]))
                elif task == 'p':
                    result.append([(spo['subject'], spo['predicate'], spo['object']) for spo in tmp['spo_list']])
                else:
                    print('wrong')
                    assert 1 == -1

    if get_answer:
        return data, tags, result
    else:
        return data, tags, texts


class MyDatasetNer(Dataset):
    def __init__(self, file_path, is_train=True, task=None):
        super(Dataset, self).__init__()
        self.is_train = is_train
        if is_train:
            self.texts, self.tags, self.ners = gen_train_data_ner(file_path, task=task)
            self.ners = padding(self.ners)
        else:
            self.texts, self.tags, _ = gen_test_data(file_path, False, task)

        self.texts = word2index(self.texts)
        self.texts = padding(self.texts)
        self.tags = tag2index(self.tags)
        self.tags = padding(self.tags)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        if self.is_train:
            return torch.LongTensor(self.texts[item]), torch.LongTensor(self.tags[item]), \
                    torch.LongTensor(self.ners[item])
        else:
            return torch.LongTensor(self.texts[item]), torch.LongTensor(self.tags[item])


class MyDatasetP(Dataset):
    def __init__(self, file_path, is_train=True):
        super(Dataset, self).__init__()
        self.is_train = is_train
        if is_train:
            self.texts, self.tags, self.sbjs, self.objs, self.sbj_bounds, self.obj_bounds, self.ps = \
                gen_train_data_p(file_path)
            self.sbjs = padding(self.sbjs)
            self.objs = padding(self.objs)
            self.sbj_bounds = padding_three_d(self.sbj_bounds)
            self.obj_bounds = padding_three_d(self.obj_bounds)
            self.ps = padding_three_d(self.ps)

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
                   torch.LongTensor(self.sbjs[item]), torch.LongTensor(self.objs[item]),\
                   torch.LongTensor(self.sbj_bounds[item]), torch.LongTensor(self.obj_bounds[item]),\
                   torch.LongTensor(self.ps[item])
        else:
            return torch.LongTensor(self.texts[item]), torch.LongTensor(self.tags[item])


def build_loader(file_path, batch_size, shuffle, drop_last, is_train=True, task=None):
    if task == 'p':
        dataset = MyDatasetP(file_path, is_train=is_train)
    else:
        dataset = MyDatasetNer(file_path, is_train=is_train, task=task)

    data_iter = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return data_iter


if __name__ == '__main__':
    # build_vocab_embedding()
    get_dict_schemas()
    # pass
