from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm
import pickle
import numpy as np
# from nltk.tree import Tree
# import spacy
import os

datasets = ['restaurant', 'laptop']
# processes = ['PSEUDO-BEST']
sizes = ['SMALL', 'MEDIUM', 'LARGE', 'FULL']
# datasets = ['device']

'-----generate merged data-----'
'''
[PAD]:  [0, 0, 0, 0, 0, 0, 0]
O:      [1, 0, 0, 0, 0, 0, 0]
B-POS:  [0, 1, 0, 0, 0, 0, 0]
B-NEU:  [0, 0, 1, 0, 0, 0, 0]
B-NEG:  [0, 0, 0, 1, 0, 0, 0]
I-POS:  [0, 0, 0, 0, 1, 0, 0]
I-NEU:  [0, 0, 0, 0, 0, 1, 0]
I-NEG:  [0, 0, 0, 0, 0, 0, 1]
'''

TRANEMNLP = False # 1.原始数据 => 2.文本标签分离
GETDICT = False # 生成DEP和POS词典
# PREPROCESS = False # 2.文本标签分离 => 3.Parsing后数据 + 4.CoNLL格式数据
PREPROCESS = True # 2.文本标签分离 => 3.Parsing后数据 + 4.CoNLL格式数据
WRITETOFILE = False
OUTPUTHEAD = False
DEVSPLIT = False
CHECKDEV = True

# 'CORENLP'
nlp = StanfordCoreNLP(r'/home/chenzhuang/stanford-corenlp-full-2018-10-05')
print(nlp.dependency_parse('the only thing i do n\'t like about this is the cover does suck .'))

if PREPROCESS:
    print('PREPROCESS')
    with open('./data/dep.dict', 'r', encoding='utf-8') as f:
        dep_dict = eval(f.read())
    with open('./data/pos.dict', 'r', encoding='utf-8') as f:
        pos_dict = eval(f.read())

    def merge_dep_result(dep_raw):

        root_cnt = 0
        dep_correct = []

        prefix_len = 0
        for idx, triple in enumerate(dep_raw):
            if triple[0] == 'ROOT' and root_cnt != 0:
                prefix_len = idx

            if triple[0] == 'ROOT':
                dep_correct.append((triple[0], 0, triple[2] + prefix_len))
            else:
                dep_correct.append((triple[0], triple[1]+prefix_len, triple[2]+prefix_len))


            if triple[0] == 'ROOT':
                root_cnt += 1

        return dep_correct

    def stanford_parse(text, is_sentence=False):
        parsed_text = nlp.word_tokenize(text.strip())
        if not is_sentence:
            return parsed_text
        else:
            parsed_pos = []
            corenlp_pos = nlp.pos_tag(text.strip())
            for pair in corenlp_pos:
                pos = pair[1]
                parsed_pos.append(pos)

            seq_len = len(parsed_text)

            parsed_dep = np.zeros((seq_len, seq_len)).astype('int64') # [('ROOT', 0, 6), ('cc', 6, 1), ('det', 3, 2), ('nsubj', 6, 3), ('cop', 6, 4), ('advmod', 6, 5), ('case', 8, 7), ('nmod', 6, 8), ('punct', 6, 9)]
            corenlp_dep_raw = nlp.dependency_parse(text.strip())
            corenlp_dep = merge_dep_result(corenlp_dep_raw)

            # print(corenlp_dep_raw)
            # print(corenlp_dep)

            if len(corenlp_dep) != seq_len:
                raise ValueError

            parsed_dep_pair = {}
            for triple in corenlp_dep:  # (rel, governor, dependent) (rel col-head row-tail)
                dep = triple[0]
                row = triple[2] - 1
                col = (triple[1] if (triple[1] != 0) else triple[2]) - 1

                parsed_dep[row][col] = dep_dict[dep]
                parsed_dep[col][row] = dep_dict[dep]
                parsed_dep_pair[row] = '{}-{}'.format(dep, col)

            assert len(parsed_text) == len(corenlp_dep) == len(corenlp_pos)

            return parsed_text, parsed_pos, parsed_dep, parsed_dep_pair

    def find_term(words, terms, tag):
        labels = ['O'] * len(words)

        positions = []
        if words == ['the', 'takeout', 'menu', 'says', 'to', 'keep', 'an', 'eye', 'out', 'for', 'an', 'expanded', 'menu', 'offering', 'more', 'italian', 'dishes', ',', 'i', 'ca', "n't", 'wait', '!']:
            print('debug')

        except_positions = []
        for term in terms:
            start = term[0]
            length = len(term)

            position = []
            for i, word in enumerate(words):
                flag = 0
                if start == word:
                    flag = 1
                    for j in range(length):
                        if i+j > len(words)-1:
                            flag = 0
                            break
                        else:
                            if term[j] == words[i+j]:
                                flag = 1
                            else:
                                flag = 0
                                break
                if flag == 1:
                    for j in range(length):
                        position.append(i+j)
            if len(position) < len(term):
                # print('debug')
                raise ValueError
            except_positions.extend(position)
            positions.append(position)


        for position in positions:
            for idx, posi in enumerate(position):
                if idx == 0:
                    labels[posi] = 'B-{}'.format(tag)
                elif posi == position[idx-1]+1:
                    labels[posi] = 'I-{}'.format(tag)
                else:
                    labels[posi] = 'B-{}'.format(tag)

        if len(positions) != len(terms):
            # print('debug')
            raise ValueError

        return labels

    for size in sizes:
        for dataset in datasets:
            in_sentence = open('./data/{}/PSEUDO-BEST/{}/sentence.txt'.format(dataset, size), 'r', encoding='utf-8').readlines()
            in_label = open('./data/{}/PSEUDO-BEST/{}/label.txt'.format(dataset, size), 'r', encoding='utf-8').readlines()


            if not os.path.exists('./data/{}/PSEUDO-FINAL/{}'.format(dataset, size)):
                os.makedirs('./data/{}/PSEUDO-FINAL/{}'.format(dataset, size))
            out_sentence = open('./data/{}/PSEUDO-FINAL/{}/sentence.txt'.format(dataset, size), 'w', encoding='utf-8')
            out_label = open('./data/{}/PSEUDO-FINAL/{}/label.txt'.format(dataset, size), 'w', encoding='utf-8')
            out_pos = open('./data/{}/PSEUDO-FINAL/{}/pos.txt'.format(dataset, size), 'w', encoding='utf-8')
            out_rel = open('./data/{}/PSEUDO-FINAL/{}/rel.txt'.format(dataset, size), 'w', encoding='utf-8')
            out_dep = open('./data/{}/PSEUDO-FINAL/{}/dep.graph'.format(dataset, size), 'wb')

            idx2graph = {}
            # for line_idx in tqdm(range(len(sentence_opinion_lines))):

            # head_line = '{}\t{}\t{}\t{}\t{}\n'.format('WORD',
            #                                     'ASPECT',
            #                                     'POLARITY',
            #                                     'POS',
            #                                     'DEP')
            # out_f.write(head_line)
            for line_idx in tqdm(range(0, len(in_sentence))):
                has_aspect = False
                sentence_line = in_sentence[line_idx].strip()
                words = in_sentence[line_idx].strip().split()
                # aspects = in_aspect[line_idx].strip().split()
                # polarities = in_polarity[line_idx].strip().split()
                labels = in_label[line_idx].strip().split()

                'WORD ASPECT OPINION POLARITY POS'

                parsed_words, parsed_pos, parsed_dep, parsed_dep_pair = stanford_parse(sentence_line, is_sentence=True)
                parsed_labels = []
                # for word_idx, word in enumerate(words):
                #     for j in range(len(nlp.pos_tag(word))):
                #         'aspect'
                #         if aspects[word_idx] == '1' and j > 0:
                #             parsed_aspects.append('2')
                #         else:
                #             parsed_aspects.append(aspects[word_idx])
                #         'polairity'
                #         parsed_polarities.append(polarities[word_idx])
                if '2 1/2' in sentence_line:
                    print('debug 1/2')
                raw_idx = 0
                par_idx = 0

                while raw_idx < len(words) and par_idx < len(parsed_words):
                    # parsed_aspects.append(aspects[raw_idx])
                    # parsed_polarities.append(polarities[raw_idx])
                    parsed_labels.append(labels[raw_idx])

                    if words[raw_idx] == parsed_words[par_idx]:
                        raw_idx += 1
                        par_idx += 1
                    else:
                        if len(parsed_words[par_idx]) < len(words[raw_idx]): # 分开情形
                            sub_words = []
                            sub_words.append(parsed_words[par_idx])
                            for sup_idx in range(par_idx+1, len(parsed_words)):
                                sup_word = parsed_words[sup_idx]
                                sub_words.append(sup_word)

                                if labels[raw_idx][0] == 'B':
                                    parsed_labels.append('I' + labels[raw_idx][1:])
                                else:
                                    parsed_labels.append(labels[raw_idx])
                                if ''.join(sub_words) == words[raw_idx]:
                                    par_idx = sup_idx + 1
                                    break
                                if len(''.join(sub_words)) > len(words[raw_idx]):
                                    raise ValueError
                            raw_idx += 1
                        elif len(parsed_words[par_idx]) > len(words[raw_idx]): # 合并情形
                            sub_words = []
                            sub_words.append(words[raw_idx])
                            for sup_idx in range(raw_idx+1, len(words)):
                                sup_word = words[sup_idx]
                                sub_words.append(sup_word)
                                if ' '.join(sub_words) == parsed_words[par_idx]:
                                    raw_idx = sup_idx + 1
                                    break
                            par_idx += 1

                if not len(parsed_words) == len(parsed_labels):
                    print('debug')

                idx2graph[line_idx] = parsed_dep

                refine_words = []
                for parsed_word in parsed_words:
                    refine_words.append(parsed_word.replace(' ', '-'))
                out_sentence.write(' '.join(refine_words) + '\n')
                out_label.write(' '.join(parsed_labels) + '\n')
                out_pos.write(' '.join(parsed_pos) + '\n')
                for i in range(len(refine_words)):
                    out_rel.write(parsed_dep_pair[i])
                    out_rel.write(' ' if (i != len(refine_words) -1) else '\n')

            pickle.dump(idx2graph, out_dep)
