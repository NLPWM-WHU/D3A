import numpy as np
from collections import Counter
from numpy.random import seed

def data_init(tasks):
    source_count = []
    source_word2idx = {}
    max_sent_len = 0
    for task in tasks:
        for process in ['train', 'test', 'dev',
                        'PSEUDO-2SR/SMALL', 'PSEUDO-2SR/MEDIUM', 'PSEUDO-2SR/LARGE', 'PSEUDO-2SR/FULL',
                        'PSEUDO-0719-KB/SMALL', 'PSEUDO-0719-KB/MEDIUM', 'PSEUDO-0719-KB/LARGE', 'PSEUDO-0719-KB/FULL',
                        'PSEUDO-0719-WORDNET/SMALL', 'PSEUDO-0719-WORDNET/MEDIUM', 'PSEUDO-0719-WORDNET/LARGE', 'PSEUDO-0719-WORDNET/FULL',
                        ]:
            text_path = './data/{}/{}/sentence.txt'.format(task, process)
            print('Processing {}...'.format(text_path))

            with open(text_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                source_words = []
                for line in lines:
                    sptoks = line.strip().split()
                    source_words.extend([sp.lower() for sp in sptoks])
                    if len(sptoks) > max_sent_len:
                        max_sent_len = len(sptoks)

            if len(source_count) == 0:
                source_count.append(['<pad>', 0])
            source_count.extend(Counter(source_words).most_common())
            for word, _ in source_count:
                if word not in source_word2idx:
                    source_word2idx[word] = len(source_word2idx)

            print('length', len(source_word2idx))

            protos= []
            for proto in protos:
                proto_path = './data/{}/{}/{}'.format(task, process, proto)
                print('Processing {}...'.format(proto_path))
                with open(proto_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        lm_segments = line.strip().split('###')
                        for lm_segment in lm_segments:
                            lm_pairs = lm_segment.split('@@@')
                            for lm_pair in lm_pairs:
                                lm_word = lm_pair.split()[0]
                                if lm_word not in source_word2idx:
                                    source_word2idx[lm_word] = len(source_word2idx)

    print('max_sentence_length', max_sent_len)
    with open('./data/word2id.txt', 'w', encoding='utf-8') as f:
        f.write(str(source_word2idx))

    return source_word2idx

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def init_word_embeddings(path, word2idx, embedding_file, oovname, dimension):
    print('path', path)
    wt = np.zeros([len(word2idx), dimension])
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip().split()
            if len(content) == 2: continue
            if content[0] in word2idx:
                #print(is_number(content[1]))
                if is_number(content[1]) == False: continue
                wt[word2idx[content[0]]] = np.array(list(map(np.float32, content[1:])))
    cnt = 0

    with open(path + oovname, "w") as f:
        for w in word2idx:
            if w == '<pad>':
                continue
            if np.sum(wt[word2idx[w]]) == 0.:
                cnt += 1
                f.write(w+"\n")

    # print('HIT rate: {:.2f}%, OOV rate: {:.2f}%'.format(100 - cnt/len(word2idx) * 100, cnt/len(word2idx) * 100))
    return wt

def compute_ngrams(word, min_n, max_n):
    BOW, EOW = ('', '')  # Used by FastText to attach to all words as prefix and suffix
    extended_word = BOW + word + EOW
    ngrams = []
    if 2 < len(word) <= 4:
        min_n = 2
    elif len(word) <= 2:
        min_n = 1
        pass

    for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
        for i in range(0, len(extended_word) - ngram_length + 1):
            ngrams.append(extended_word[i:i + ngram_length])
    return ngrams

def get_oov_vector(path, oov_dict, source_emb, oov_emb, dimension):
    word2ngram_dict = {}
    ngram2idx_dict = {}
    with open(path + oov_dict, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word = line.strip()
            ngrams = compute_ngrams(word, 3, 10)
            # print(word, ngrams)
            word2ngram_dict[word] = ngrams
            for ngram in ngrams:
                if ngram not in ngram2idx_dict:
                    ngram2idx_dict[ngram] = len(ngram2idx_dict)

    ngram_vec = np.zeros([len(ngram2idx_dict), dimension])
    with open(source_emb, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip().split()
            if len(content) == 2: continue
            if content[0] in ngram2idx_dict:
                #print(is_number(content[1]))
                if is_number(content[1]) == False: continue
                ngram_vec[ngram2idx_dict[content[0]]] = np.array(list(map(np.float32, content[1:])))


    # print(word2ngram_dict)
    # print(ngram2idx_dict)
    # print(ngram_vec)

    word2vec_dict = {}
    for word in word2ngram_dict:
        word_vec = np.zeros([dimension])
        ngram_cnt = 0
        ngrams = word2ngram_dict[word]
        for ngram in ngrams:
            if np.sum(ngram_vec[ngram2idx_dict[ngram]]) != 0:
                word_vec += ngram_vec[ngram2idx_dict[ngram]]
                ngram_cnt += 1
        word_vec /= np.float(ngram_cnt)+1e-6
        word2vec_dict[word] = word_vec

    cnt = 0
    for word in word2vec_dict:
        if np.sum(word2vec_dict[word]) == 0.:
            cnt += 1
    print('Ngram HIT rate: {:.2f}%, Ngram OOV rate: {:.2f}%'.format(100 - cnt/len(word2vec_dict) * 100, cnt/len(word2vec_dict) * 100))

    with open(path + oov_emb, 'w', encoding='utf-8') as f:
        for word in word2vec_dict:
            f.write(word + ' ')
            for number in word2vec_dict[word]:
                f.write(str(number) + ' ')
            f.write('\n')

def full_embedding(path, word2idx, iv_txt, oov_txt, outemb, dimension):
    # print('path', path)
    wt = np.zeros([len(word2idx), dimension])

    'IN VOCABULARY'
    with open(iv_txt, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip().split()
            if len(content) == 2: continue
            if content[0] in word2idx:
                #print(is_number(content[1]))
                if is_number(content[1]) == False: continue
                wt[word2idx[content[0]]] = np.array(list(map(np.float32, content[1:])))
    cnt = 0
    for w in word2idx:
        if w == '<pad>':
            continue
        if np.sum(wt[word2idx[w]]) == 0.:
            cnt += 1
    print('Raw   HIT rate: {:.2f}%, Raw   OOV rate: {:.2f}%'.format(100 - cnt/len(word2idx) * 100, cnt/len(word2idx) * 100))

    'OUT OF VOCABULARY'
    with open(path + oov_txt, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip().split()
            if len(content) == 2: continue
            if content[0] in word2idx:
                #print(is_number(content[1]))
                if is_number(content[1]) == False: continue
                if np.sum(wt[word2idx[content[0]]]) == 0.:
                    wt[word2idx[content[0]]] = np.array(list(map(np.float32, content[1:])))
                else:
                    raise ValueError

    cnt = 0
    for w in word2idx:
        if w == '<pad>':
            continue
        if np.sum(wt[word2idx[w]]) == 0.:
            cnt += 1
    print('Plus  HIT rate: {:.2f}%, Plus  OOV rate: {:.2f}%'.format(100 - cnt/len(word2idx) * 100, cnt/len(word2idx) * 100))

    'RANDOM INITIALIZE'
    for idx in range(len(wt)):
        if idx == 0:
            continue
        else:
            if np.sum(wt[idx]) == 0.:
                wt[idx] = np.random.normal(-0.05, 0.05, [dimension])
            else:
                pass

    cnt = 0
    for w in word2idx:
        if w == '<pad>':
            continue
        if np.sum(wt[word2idx[w]]) == 0.:
            cnt += 1
    print('Final HIT rate: {:.2f}%, Final OOV rate: {:.2f}%'.format(100 - cnt/len(word2idx) * 100, cnt/len(word2idx) * 100))

    wt = np.asarray(wt, dtype=np.float32)
    wt[0,:] = 0.0
    np.save(path + outemb, wt)
    return wt

tasks = ['restaurant', 'laptop']
data_path = './data/'
print('Initialize Word Dictionary & Embedding')
word_dict = data_init(tasks)

# GLOVE_EMB = '/home/woodong/glove.840B.300d.txt'
GLOVE_EMB = '/home/chenzhuang/glove.840B.300d.txt'
# GLOVE_EMB = './data/glove.840B.300d.txt'
# RESTAURANT_EMB = '/home/woodong/restaurant_emb.vec'
# LAPTOP_EMB = '/home/woodong/laptop_emb.vec'

init_word_embeddings(data_path, word_dict, GLOVE_EMB, 'glove_word_oov.txt', 300)
get_oov_vector(data_path, 'glove_word_oov.txt', GLOVE_EMB, 'glove_emb_oov.txt', 300)
full_embedding(data_path, word_dict, GLOVE_EMB, 'glove_emb_oov.txt', 'glove_embedding.npy', 300)

# init_word_embeddings(data_path, word_dict, RESTAURANT_EMB, 'restaurant_oov.txt', 100)
# get_oov_vector(data_path, 'restaurant_oov.txt', RESTAURANT_EMB, 'restaurant_emb_oov.txt', 100)
# full_embedding(data_path, word_dict, RESTAURANT_EMB, 'restaurant_emb_oov.txt', 'restaurant_embedding.npy', 100)
#
# init_word_embeddings(data_path, word_dict, LAPTOP_EMB, 'laptop_oov.txt', 100)
# get_oov_vector(data_path, 'laptop_oov.txt', LAPTOP_EMB, 'laptop_emb_oov.txt', 100)
# full_embedding(data_path, word_dict, LAPTOP_EMB, 'laptop_emb_oov.txt', 'laptop_embedding.npy', 100)






