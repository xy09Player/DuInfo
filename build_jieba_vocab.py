# coding = utf-8
# author = xy

import json
from jieba import posseg
import shengshixian
import pickle


def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def get_jieba_filter_word():
    file_paths = ['../data/train_data.json', '../data/dev_data.json']
    text_lists = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for line in f:
                tmp = json.loads(line)
                text_list, _ = list(zip(*posseg.lcut(tmp['text'], HMM=False)))
                text_list = list(text_list)
                text_lists.append(text_list)

    # 省、市、区、县:0.22
    shengshixianqu_set = set()
    for text_list in text_lists:
        for word in text_list:
            if len(word) > 1 and ('省' in word or '市' in word or '县' in word or '区' in word):
                shengshixianqu_set.add(word)

    # 大学: 0.1
    daxue_set = set()
    for text_list in text_lists:
        for word in text_list:
            if len(word) > 2 and not word.endswith('大学') and '大学' in word:
                daxue_set.add(word)

    # 中国:0.9
    zhongguo_set = set()
    for text_list in text_lists:
        for word in text_list:
            if '中国' in word and len(word) > 2:
                zhongguo_set.add(word)

    # 美国:0.02
    meiguo_set = set()
    for text_list in text_lists:
        for word in text_list:
            if '美国' in word and len(word) > 2:
                meiguo_set.add(word)

    # 于:0.07
    yu_set = set()
    for text_list in text_lists:
        for word in text_list:
            if len(word) > 1 and word[0] == '于':
                yu_set.add(word)

    # 出版社：0.17
    chubanshe_set = set()
    for text_list in text_lists:
        for word in text_list:
            if len(word) > 3 and '出版社' in word:
                chubanshe_set.add(word)

    # 省市县+人：0.3
    shengshixianren_set = set()
    for text_list in text_lists:
        for word in text_list:
            if word.endswith('人'):
                for s in shengshixian.shengshixian_set:
                    if s in word:
                        shengshixianren_set.add(word)
                        break

    # 细粒度
    xilidu_set = set()
    for text_list in text_lists:
        for word in text_list:
            if len(word) > 5 and is_Chinese(word):
                xilidu_set.add(word)

    # word_set
    word_set = {
        '是夜', '日出', '英国伦敦', '王之子', '网上', '汉族', '万元', '目的',
        '生于', '市南', '是从', '宗李治', '立陶宛人', '都市人', '年初', '年度',
        '月份', '韩国队', '日经', '清代', '明代', '唐代', '美国人', '日本人',
        '韩国人', '朝鲜人', '德国人', '澳大利亚人', '目下', '葡萄牙人',
    }

    word_set = shengshixianqu_set | daxue_set | zhongguo_set | meiguo_set | yu_set | chubanshe_set | shengshixianren_set | word_set

    with open('../data/jieba_vocab.pkl', 'wb') as f:
        pickle.dump(word_set, f)

if __name__ == '__main__':
    get_jieba_filter_word()
