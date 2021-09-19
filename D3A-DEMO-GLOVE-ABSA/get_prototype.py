from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from time import time

METHOD = '0726-LWPD'

tag_dict = {
'O':      [1, 0, 0, 0],
'B-POS':  [0, 1, 0, 0],
'B-NEU':  [0, 0, 1, 0],
'B-NEG':  [0, 0, 0, 1],
'I-POS':  [0, 1, 0, 0],
'I-NEU':  [0, 0, 1, 0],
'I-NEG':  [0, 0, 0, 1],
}

parser = argparse.ArgumentParser()
# parser.add_argument('--kb', default='none', type=str, help='None')
parser.add_argument('--ablation', default='None', type=str, help='None')
parser.add_argument('--tau', default=3, type=int, help='None')
# parser.add_argument('--size', default='SMALL', type=str, help='set ratio between 0 and 1 for validation support')
opt = parser.parse_args()


def softmax(probs):
    probs = np.array(probs)
    probs -= np.max(probs, axis=-1, keepdims=True)
    return np.exp(probs) / (np.sum(np.exp(probs), axis=-1, keepdims=True) + 1e-6)


def read_data(fname):
    # assert data_type in ['origin', 'bank']
    max_length = 110

    with open('./data/dep.dict', 'r', encoding='utf-8') as f:
        dep_dict = eval(f.read())
    with open('./data/pos.dict', 'r', encoding='utf-8') as f:
        pos_dict = eval(f.read())
    review = open(fname + r'sentence.txt', 'r', encoding='utf-8').readlines()
    pos_data = open(fname + r'pos.txt', 'r', encoding='utf-8').readlines()
    graph_data = pickle.load(open(fname + r'dep.graph', 'rb'))
    label_data = open(fname + r'label.txt', 'r', encoding='utf-8').readlines()

    word_dict = []
    dep_embs = []
    pos_embs = []
    tag_embs = []

    # cut_off = int(opt.percentage * len(review))

    for sen_index, _ in enumerate(review):
        if 'train' in fname and sen_index >= opt.cut_off:
            print('\n>>> Use {} training data, totally {} samples.'.format(opt.size, min(opt.cut_off, len(review))))
            break

        words = review[sen_index].strip().split()
        labels = label_data[sen_index].strip().split()
        dep_graph = graph_data[sen_index]
        pos_line = pos_data[sen_index].strip().split()

        length = len(words)

        '''POS & DEP'''

        for i in range(length):
            word = words[i]
            label = labels[i]

            dep_multihot = [0.] * 40
            dep_slice = dep_graph[i]
            for dep in dep_slice:
                if dep != 0:
                    dep_multihot[dep - 1] = 1.

            pos = pos_line[i]
            pos_onehot = [0.] * 45
            pos_indice = pos_dict[pos] - 1
            pos_onehot[pos_indice] = 1.

            dep_multihot = np.array(dep_multihot)
            pos_onehot = np.array(pos_onehot)
            tag_onehot = np.array(tag_dict[label])

            if word not in word_dict:
                word_dict.append(word)
                dep_embs.append(dep_multihot)
                pos_embs.append(pos_onehot)
                tag_embs.append(tag_onehot)

            else:
                position = word_dict.index(word)
                dep_embs[position] += dep_multihot
                pos_embs[position] += pos_onehot
                tag_embs[position] += tag_onehot

    pass
    return word_dict, np.array(pos_embs), np.array(dep_embs), np.array(tag_embs)

task = 'data'
origins = ['restaurant', 'laptop']
banks = ['restaurant', 'laptop']
processes = ['train', 'test', 'dev']
sizes = ['SMALL', 'MEDIUM', 'LARGE', 'FULL']


TOPK = 10

with open('./{}/word2id.txt'.format(task), 'r', encoding='utf-8') as f:
    word2id_dict = eval(f.read())


def get_word_embedding(words, word2id_dict, emb_type):
    word2vec = np.load('./{}/{}_embedding.npy'.format(task, emb_type))
    word_embs = []
    for word in words:
        word_id = word2id_dict[word]
        word_emb = word2vec[word_id]
        word_embs.append(word_emb)

    return np.array(word_embs)

for size in sizes:
    for process in (processes + ['PSEUDO-FINAL/{}'.format(size)]):
        for origin in origins:
            for bank in banks:
                if bank != origin:
                    continue

                start_time = time()
                opt.size = size
                size_dict = {
                    'SMALL': 50,
                    'MEDIUM': 150,
                    'LARGE': 300,
                    'FULL': 10000

                }
                opt.cut_off = size_dict[opt.size]

                print('ORIGIN: {}, PROCESS: {}, BANK: {}'.format(origin, process, bank))
                oracle_dict = {}

                origin_path = './{}/{}/{}/'.format(task, origin, process)
                bank_path = './{}/{}/train/'.format(task, bank)
                #TODO
                # METHOD = 'prototype'
                METHOD = 'nolabel_prototype'
                prototype_f = open('{}{}_{}.txt'.format(origin_path, METHOD, opt.size), 'w', encoding='utf-8')
                print('WRITE TO FILE: ' + '{}{}_{}.txt'.format(origin_path, METHOD, opt.size) + '\n')

                if process == 'train':
                    map_dict_f = open('{}{}_map_{}.txt'.format(origin_path, METHOD, opt.size), 'w', encoding='utf-8')
                    print('WRITE MAP DICT: ' + '{}{}_map_{}.txt'.format(origin_path, METHOD, opt.size) + '\n')


                    print('processing train data')
                    origin_word_dict, origin_pos_embs, origin_dep_embs, origin_tag_embs = read_data(origin_path)
                    origin_size = len(origin_word_dict)
                    origin_pos_embs = (origin_pos_embs > 0).astype("float32")
                    origin_dep_embs = (origin_dep_embs > 0).astype("float32")
                    origin_tag_embs = (origin_tag_embs > 0).astype("float32")
                    origin_glove_embs = get_word_embedding(origin_word_dict, word2id_dict, 'glove').astype("float32")

                    bank_word_dict, bank_pos_embs, bank_dep_embs, bank_tag_embs = read_data(bank_path)
                    bank_size = len(bank_word_dict)
                    bank_pos_embs = (bank_pos_embs > 0).astype("float32")
                    bank_dep_embs = (bank_dep_embs > 0).astype("float32")
                    bank_tag_embs = (bank_tag_embs > 0).astype("float32")
                    bank_glove_embs = get_word_embedding(bank_word_dict, word2id_dict, 'glove').astype("float32")

                    pos_similarities = (cosine_similarity(origin_pos_embs, bank_pos_embs))
                    dep_similarities = (cosine_similarity(origin_dep_embs, bank_dep_embs))
                    glove_similarities = (cosine_similarity(origin_glove_embs, bank_glove_embs))
                    tag_similarities = (cosine_similarity(origin_tag_embs, bank_tag_embs))

                    # TODO
                    # similarities = (pos_similarities * dep_similarities * glove_similarities * tag_similarities)
                    similarities = (pos_similarities * dep_similarities * glove_similarities)

                    'GET DICT'
                    for origin_index, origin_word in enumerate(origin_word_dict):
                        map_dict_f.write('{}||'.format(origin_word))
                        here_valid_pair = []
                        here_topk_index = np.flipud(np.argsort(similarities[origin_index, :]))

                        for here_k_index in here_topk_index:
                            here_k_word = bank_word_dict[here_k_index]
                            here_k_similarity = similarities[origin_index, here_k_index]
                            if here_k_similarity > 0:
                                here_valid_pair.append((here_k_word, here_k_similarity))

                        for valid_index, valid_pair in enumerate(here_valid_pair):
                            here_valid_word, here_valid_similarity = valid_pair
                            map_dict_f.write('{} {:.2f}'.format(here_valid_word, here_valid_similarity))
                            if valid_index == len(here_valid_pair) - 1:
                                map_dict_f.write('\n')
                            else:
                                map_dict_f.write('||')






                    for origin_index, origin_word in enumerate(origin_word_dict):
                        # topk_index = topk_indices[origin_index]
                        oracle_info = ''
                        topk_index = np.flipud(np.argsort(similarities[origin_index, :]))[:15]
                        topk_words = []
                        topk_similarities = []
                        for k_index in topk_index:
                            if len(topk_words) > TOPK:
                                break

                            k_word = bank_word_dict[k_index]
                            k_similarity = similarities[origin_index, k_index]

                            'In a single domain, we do not need the original word.'
                            # if k_word == origin_word:
                            #     continue

                            topk_words.append(k_word)
                            topk_similarities.append(k_similarity)

                        # topk_similarities = softmax(topk_similarities) #保留真实值，在utils里使用softmax

                        for i in range(len(topk_words)):
                            oracle_info += '{} {:.2f}'.format(topk_words[i], topk_similarities[i])
                            if i != len(topk_words) - 1:
                                oracle_info += '@@@'

                        oracle_dict[origin_word] = oracle_info

                        # print('\n')
                        # print(origin_word)
                        # print(' '.join(topk_words))
                        # print(' '.join(topk_similarities))

                    sentences = open('{}sentence.txt'.format(origin_path), 'r', encoding='utf-8').readlines()
                    for sentence_idx, sentence in enumerate(sentences):
                        if sentence_idx >= opt.cut_off:
                            break
                        words = sentence.strip().split()
                        for word_idx, word in enumerate(words):
                            oracle = oracle_dict[word]
                            # prototype_f.write('{}@@@'.format(word_idx))
                            prototype_f.write(oracle)

                            if word_idx != len(words) - 1:
                                prototype_f.write('###')
                            else:
                                prototype_f.write('\n')

                else:
                    print('processing {} data'.format(process))
                    in_cnt = 0
                    out_cnt = 0

                    auxi_path = './{}/{}/{}/'.format(task, origin, 'train')
                    auxi_word_dict, auxi_pos_embs, auxi_dep_embs, auxi_tag_embs = read_data(auxi_path)
                    auxi_size = len(auxi_word_dict)
                    auxi_pos_embs = (auxi_pos_embs > 0).astype("float32")
                    auxi_dep_embs = (auxi_dep_embs > 0).astype("float32")
                    auxi_tag_embs = (auxi_tag_embs > 0).astype("float32")

                    bank_word_dict, bank_pos_embs, bank_dep_embs, bank_tag_embs = read_data(bank_path)
                    bank_size = len(bank_word_dict)
                    bank_pos_embs = (bank_pos_embs > 0).astype("float32")
                    bank_dep_embs = (bank_dep_embs > 0).astype("float32")
                    bank_tag_embs = (bank_tag_embs > 0).astype("float32")
                    bank_glove_embs = get_word_embedding(bank_word_dict, word2id_dict, 'glove').astype("float32")

                    test_word_dict, _, _, _ = read_data(origin_path)
                    test_glove_embs = get_word_embedding(test_word_dict, word2id_dict, 'glove').astype("float32")

                    '''
                    如何计算词之间的映射关系
                    1.word相似度 * pos相似度 * dep相似度（缺点是信息重复了，pos和dep信息在syntactic增强中已用到）
                    2.仅word相似度（缺点是不考虑出现频次，相似的尾部词会被选中作为模板词）
                    3.word相似度 + PMI筛选（仅考虑与领域PMI大于0的词了，常用词被忽略）
                    4.word相似度 + PMI筛选 + 词频筛选
                    '''

                    with open('./data/dep.dict', 'r', encoding='utf-8') as f:
                        dep_dict = eval(f.read())
                    with open('./data/pos.dict', 'r', encoding='utf-8') as f:
                        pos_dict = eval(f.read())
                    review = open(origin_path  + r'sentence.txt', 'r', encoding='utf-8').readlines()
                    pos_data = open(origin_path + r'pos.txt', 'r', encoding='utf-8').readlines()
                    graph_data = pickle.load(open(origin_path + r'dep.graph', 'rb'))

                    word_dict = []
                    dep_embs = []
                    pos_embs = []
                    tag_embs = []

                    for sen_index, _ in enumerate(tqdm(review)):
                        oracle_sentence = {}

                        words = review[sen_index].strip().split()
                        dep_graph = graph_data[sen_index]
                        pos_line = pos_data[sen_index].strip().split()

                        length = len(words)

                        sentence_dep_emb = []
                        sentence_pos_emb = []
                        sentence_tag_emb = []
                        sentence_word_ids = []

                        '''POS & DEP'''

                        for i in range(length):
                            word = words[i]
                            sentence_word_ids.append(test_word_dict.index(word))
                            if word not in auxi_word_dict:
                                out_cnt += 1

                                dep_multihot = [0.] * 40
                                dep_slice = dep_graph[i]
                                for dep in dep_slice:
                                    if dep != 0:
                                        dep_multihot[dep - 1] = 1.

                                pos = pos_line[i]
                                pos_onehot = [0.] * 45
                                pos_indice = pos_dict[pos] - 1
                                pos_onehot[pos_indice] = 1.

                                dep_multihot = np.array(dep_multihot)
                                pos_onehot = np.array(pos_onehot)
                                tag_onehot = np.array([1, 1, 1, 1])
                                pass

                            else:
                                in_cnt += 1
                                find_index = auxi_word_dict.index(word)
                                dep_multihot = auxi_dep_embs[find_index, :]
                                pos_onehot = auxi_pos_embs[find_index, :]
                                tag_onehot = auxi_tag_embs[find_index, :]

                            sentence_dep_emb.append(dep_multihot)
                            sentence_pos_emb.append(pos_onehot)
                            sentence_tag_emb.append(tag_onehot)

                        sentence_dep_emb = np.array(sentence_dep_emb)
                        sentence_pos_emb = np.array(sentence_pos_emb)
                        sentence_tag_emb = np.array(sentence_tag_emb)
                        sentence_glove_emb = test_glove_embs[sentence_word_ids, :]

                        pos_similarities = (cosine_similarity(sentence_pos_emb, bank_pos_embs))
                        dep_similarities = (cosine_similarity(sentence_dep_emb, bank_dep_embs))
                        glove_similarities = (cosine_similarity(sentence_glove_emb, bank_glove_embs))
                        tag_similarities = (cosine_similarity(sentence_tag_emb, bank_tag_embs))

                        # TODO
                        # similarities = (pos_similarities * dep_similarities * glove_similarities * tag_similarities)
                        similarities = (pos_similarities * dep_similarities * glove_similarities)

                        for sen_word_index, sen_word in enumerate(words):
                            # topk_index = topk_indices[origin_index]
                            oracle_info = ''
                            topk_index = np.flipud(np.argsort(similarities[sen_word_index, :]))[:15]
                            topk_words = []
                            topk_similarities = []
                            for k_index in topk_index:
                                if len(topk_words) > TOPK:
                                    break

                                k_word = bank_word_dict[k_index]
                                k_similarity = similarities[sen_word_index, k_index]

                                'In a single domain, we do not need the original word.'
                                # if k_word == sen_word:
                                #     continue

                                topk_words.append(k_word)
                                topk_similarities.append(k_similarity)

                            # topk_similarities = softmax(topk_similarities) #保留真实值，在utils里使用softmax

                            for i in range(len(topk_words)):
                                oracle_info += '{} {:.2f}'.format(topk_words[i], topk_similarities[i])
                                if i != len(topk_words) - 1:
                                    oracle_info += '@@@'

                            # prototype_f.write('{}@@@'.format(sen_index))
                            prototype_f.write(oracle_info)

                            if sen_word_index != len(words) - 1:
                                prototype_f.write('###')
                            else:
                                prototype_f.write('\n')

                        pass

                    print('In Dict: {}, {:.2f} Out Dict: {}, {:.2f}'.format(in_cnt, in_cnt / (in_cnt+out_cnt), out_cnt, out_cnt / (in_cnt+out_cnt)))









