import numpy as np
import random

class ABSADataset():
    def __init__(self, fname, source_word2idx, opt, data_type, baseline=None):
        assert data_type in ['train', 'test', 'dev']
        max_length = opt.max_sentence_len
        lm = opt.lm
        topk = opt.topk
        size_dict= {
            'SMALL': 50,
            'MEDIUM': 150 ,
            'LARGE': 300,
            'FULL': 10000

        }
        opt.cut_off = size_dict[opt.size]
        percentage = 0.

        data = []
        review = open(fname + r'sentence.txt', 'r', encoding='utf-8').readlines()
        ae_data = open(fname + r'label.txt', 'r', encoding='utf-8').readlines()
        lm_data = open(fname + r'{}_{}.txt'.format(opt.lm, opt.size), 'r', encoding='utf-8').readlines()

        # cut_off = int(percentage * len(review))

        for index, _ in enumerate(review):
            if data_type == 'train' and index >= opt.cut_off:
                print('\n>>> Use {} training data, totally {} samples.\n'.format(opt.size, min(opt.cut_off, len(review))))
                break

            '''
            Word Index
            '''
            sptoks = review[index].strip().split()

            idx = []
            mask = []
            len_cnt = 0
            for sptok in sptoks:
                if len_cnt < max_length:
                    idx.append(source_word2idx[sptok.lower()])
                    mask.append(1.)
                    len_cnt += 1
                else:
                    break

            source_data_per = (idx + [0] * (max_length - len(idx)))
            source_mask_per = (mask + [0.] * (max_length - len(idx)))

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

            ae_labels = ae_data[index].strip().split()
            aspect_label = []
            for l in ae_labels:
                if l == 'O' :
                    aspect_label.append([1, 0, 0, 0, 0, 0, 0])
                elif l == 'B-POS':
                    aspect_label.append([0, 1, 0, 0, 0, 0, 0])
                elif l == 'B-NEU':
                    aspect_label.append([0, 0, 1, 0, 0, 0, 0])
                elif l == 'B-NEG':
                    aspect_label.append([0, 0, 0, 1, 0, 0, 0])
                elif l == 'I-POS':
                    aspect_label.append([0, 0, 0, 0, 1, 0, 0])
                elif l == 'I-NEU':
                    aspect_label.append([0, 0, 0, 0, 0, 1, 0])
                elif l == 'I-NEG':
                    aspect_label.append([0, 0, 0, 0, 0, 0, 1])
                else:
                    raise ValueError

            aspect_y_per = (aspect_label + [[0, 0, 0, 0, 0, 0, 0]] * (max_length - len(idx)))

            segments = lm_data[index].strip().split('###')
            words_list = []
            probs_list = []
            for segment in segments:
                pairs = segment.split('@@@')
                words = []
                probs = []
                topk_cnt = 0
                for pair in pairs:
                    if topk_cnt >= topk:
                        break
                    word = source_word2idx[pair.split()[0]]
                    prob = float(pair.split()[1])
                    words.append(word)
                    probs.append(prob)
                    topk_cnt += 1
                words_list.append(words)
                probs_list.append(softmax(probs))
            words_per = (words_list + [[0] * topk] * (max_length - len(idx)))
            probs_per = (probs_list + [[0.] * topk] * (max_length - len(idx)))

            data_per = {'x': np.array(source_data_per, dtype='int64'),
                        'mask': np.array(source_mask_per, dtype='float32'),
                        'aspect_y': np.array(aspect_y_per, dtype='int64'),
                        'lmwords': np.array(words_per, dtype='int64'),
                        'lmprobs': np.array(probs_per, dtype='float32'),
                        'data_type': np.array([1, 0], dtype='float32')}
            data.append(data_per)



        if data_type == 'train' and baseline != None:
            pseudo_dir = './data/{}/PSEUDO-{}/{}'.format(opt.dataset, baseline, opt.size)
            pseudo_review = open('{}/sentence.txt'.format(pseudo_dir), 'r', encoding='utf-8').readlines()
            pseudo_ae_data = open('{}/label.txt'.format(pseudo_dir), 'r', encoding='utf-8').readlines()
            # pseudo_lm_data = open('{}/prototype_{}.txt'.format(pseudo_dir, opt.size), 'r', encoding='utf-8').readlines()
            pseudo_lm_data = open('{}/{}_{}.txt'.format(pseudo_dir, opt.lm, opt.size), 'r', encoding='utf-8').readlines()
            print('\nAdd {} Pseudo Samples from Baseline {}.\n'.format(len(pseudo_review), baseline))

            for index, _ in enumerate(pseudo_review):
                sptoks = pseudo_review[index].strip().split()

                idx = []
                mask = []
                len_cnt = 0
                for sptok in sptoks:
                    if len_cnt < max_length:
                        idx.append(source_word2idx[sptok.lower()])
                        mask.append(1.)
                        len_cnt += 1
                    else:
                        break

                source_data_per = (idx + [0] * (max_length - len(idx)))
                source_mask_per = (mask + [0.] * (max_length - len(idx)))

                ae_labels = pseudo_ae_data[index].strip().split()
                aspect_label = []
                for l in ae_labels:
                    if l == 'O' :
                        aspect_label.append([1, 0, 0, 0, 0, 0, 0])
                    elif l == 'B-POS':
                        aspect_label.append([0, 1, 0, 0, 0, 0, 0])
                    elif l == 'B-NEU':
                        aspect_label.append([0, 0, 1, 0, 0, 0, 0])
                    elif l == 'B-NEG':
                        aspect_label.append([0, 0, 0, 1, 0, 0, 0])
                    elif l == 'I-POS':
                        aspect_label.append([0, 0, 0, 0, 1, 0, 0])
                    elif l == 'I-NEU':
                        aspect_label.append([0, 0, 0, 0, 0, 1, 0])
                    elif l == 'I-NEG':
                        aspect_label.append([0, 0, 0, 0, 0, 0, 1])
                    else:
                        raise ValueError

                aspect_y_per = (aspect_label + [[0, 0, 0, 0, 0, 0, 0]] * (max_length - len(idx)))

                segments = pseudo_lm_data[index].strip().split('###')
                words_list = []
                probs_list = []
                for segment in segments:
                    pairs = segment.split('@@@')
                    words = []
                    probs = []
                    topk_cnt = 0
                    for pair in pairs:
                        if topk_cnt >= topk:
                            break
                        word = source_word2idx[pair.split()[0]]
                        prob = float(pair.split()[1])
                        words.append(word)
                        probs.append(prob)
                        topk_cnt += 1
                    words_list.append(words)
                    probs_list.append(softmax(probs))
                words_per = (words_list + [[0] * topk] * (max_length - len(idx)))
                probs_per = (probs_list + [[0.] * topk] * (max_length - len(idx)))


                data_per = {'x': np.array(source_data_per, dtype='int64'),
                            'mask': np.array(source_mask_per, dtype='float32'),
                            'aspect_y': np.array(aspect_y_per, dtype='int64'),
                            'lmwords': np.array(words_per, dtype='int64'),
                            'lmprobs': np.array(probs_per, dtype='float32'),
                            'data_type': np.array([0, 1], dtype='float32')}
                data.append(data_per)


        self.data = data
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def softmax(probs):
    probs = np.array(probs)
    mask = np.asarray(probs != 0, np.float32)
    probs -= np.max(probs, axis=-1, keepdims=True)
    probs_exp = np.exp(probs) * mask
    return probs_exp / (np.sum(probs_exp, axis=-1, keepdims=True) + 1e-6)


