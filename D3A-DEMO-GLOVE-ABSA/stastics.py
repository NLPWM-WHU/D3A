import os

datasets = ['restaurant', 'laptop', 'wnut16', 'wnut17']
processes = ['train', 'dev', 'test']

print('\nDataset\tProcess\tSentence\tToken\tMention\tType')
for dataset in datasets:
    for process in processes:
        dir = 'data/{}/{}'.format(dataset, process)
        if os.path.exists(dir):
            sentence_count = 0
            token_count = 0
            mention_count = 0
            type_list = []

            sentences = open('{}/sentence.txt'.format(dir), 'r', encoding='utf-8').readlines()
            labels = open('{}/label.txt'.format(dir), 'r', encoding='utf-8').readlines()

            for sample_idx in range(len(sentences)):
                sentence_count += 1

                sentence = sentences[sample_idx]
                words = sentence.strip().split()

                label = labels[sample_idx]
                tags = label.strip().split()

                for word_idx in range(len(words)):
                    token_count += 1
                    word = words[word_idx]
                    tag = tags[word_idx]

                    if tag.startswith('B'):
                        mention_count += 1

                    if not tag.startswith('I'):
                        if tag not in type_list:
                            type_list.append(tag)


            print('{}\t{}\t{}\t{}\t{}\t{}'.format(dataset, process, sentence_count, token_count, mention_count, len(type_list)))


