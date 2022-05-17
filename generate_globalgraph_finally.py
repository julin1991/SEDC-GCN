# -*- coding: utf-8 -*-
# ------------------
# @Author: BinLiang
# @Mail: bin.liang@stu.hit.edu.cn
# ------------------

#生成inter_graph图
import numpy as np
import spacy
import pickle
import re
from string import digits
def read_file2(filename):
    with  open(filename, 'r',encoding='utf-8')as f:
        all_words=[]
        text = f.readlines()
        for word in text:
            word=word.rstrip()
            all_words.append(word)
        #返回list类型数据
        #text = text.split('\n')
    return all_words
stop_words=read_file2("../en_stopwords.txt")
stop_words=[]
for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's','1','2','3','4','5','6','7','8','9','0','*','\'\'','&','$','|','-','#','$']:
    stop_words.append(w)
nlp = spacy.load('en_core_web_sm')
def dependency_adj_matrix(text,word_pair_pmi):
    text=text.strip()
    text_list = text.split()
    seq_len = len(text_list)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    flag = 1
    document = nlp(text)
    for i in range(seq_len):
       for j in range(seq_len):
           if i==j:
               matrix[i][j]=1
           else:
               word_i=text_list[i]
               word_j=text_list[j]
               word_str=word_i+","+word_j
               if word_str in word_pair_pmi:
                   #print(word_str)
                   matrix[i][j]=1
                   matrix[j][i] =1
    for token in document:
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            # https://spacy.io/docs/api/token

            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1
    return matrix

def dependency_adj_matrix2(text,word_pair_pmi):
    text=text.strip()
    text_list = text.split()
    seq_len = len(text_list)
    matrix1 = np.zeros((seq_len, seq_len)).astype('float32')
    matrix2 = np.zeros((seq_len, seq_len)).astype('float32')
    flag = 1
    document = nlp(text)
    for i in range(seq_len):
       for j in range(seq_len):
           if i==j:
               matrix1[i][j]=1
           else:
               word_i=text_list[i]
               word_j=text_list[j]
               word_str=word_i+","+word_j
               word_str2 = word_j + "," + word_i
               if word_str in word_pair_pmi:
                   #print(word_str)
                   #matrix1[i][j]=word_pair_pmi[word_str]
                   #matrix1[j][i] =word_pair_pmi[word_str]
                   matrix1[i][j] = 1
                   matrix1[j][i] = 1
               if  word_str2 in word_pair_pmi:
                    #matrix1[i][j] = word_pair_pmi[word_str2]
                    #matrix1[j][i] = word_pair_pmi[word_str2]
                    matrix1[i][j] = 1
                    matrix1[j][i] = 1
    for token in document:
        if token.i < seq_len:
            matrix2[token.i][token.i] = 1
            # https://spacy.io/docs/api/token

            for child in token.children:
                if child.i < seq_len:
                    matrix2[token.i][child.i] = 1
                    matrix2[child.i][token.i] = 1
    return matrix1,matrix2
def get_corpus(filename_train,filename_test,corpusName):
    fin = open(filename_train, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines_train = fin.readlines()
    fin.close()
    fin = open(filename_test, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines_test = fin.readlines()
    fin.close()
    f_out = open(corpusName+'.corpus','w',encoding='utf-8')
    graph_idx = 0
    doc_words_list = []
    punctuation = '!,;:?"\'、，；.'
    for i in range(0, len(lines_train), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines_train[i].partition("$T$")]

        aspect_pos = len(text_left.split())
        # print(aspect_pos)
        aspect = lines_train[i + 1].lower().strip()
        aspect_len = len(aspect.split())
        text=text_left+' '+aspect+' '+text_right
        text = text.lower().strip()
        text = re.sub(r'[{}]+'.format(punctuation), ' ', text)
        text = text.translate(str.maketrans('', '', digits))
        doc_words = []
        words = text.split()
        for word in words:
            if word not in stop_words:
                doc_words.append(word)
        doc_words_str = ' '.join(doc_words)
        #print(doc_words_str)
        #f_out.write(doc_words_str)
        doc_words_list.append(doc_words_str)

    for i in range(0, len(lines_test), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines_test[i].partition("$T$")]

        aspect_pos = len(text_left.split())
        # print(aspect_pos)
        aspect = lines_test[i + 1].lower().strip()
        aspect_len = len(aspect.split())
        text = text_left + ' ' + aspect + ' ' + text_right
        # text=text.replace("n't", "not")
        # text = text.replace("'s", "is")
        # text = text.replace("'m", "am")
        # text = text.replace("'re", "are")
        text = text.lower().strip()
        text = re.sub(r'[{}]+'.format(punctuation), ' ', text)
        doc_words = []
        words = text.split()
        for word in words:
            if word not in stop_words:
               doc_words.append(word)
        doc_words_str = ' '.join(doc_words)
        #print(doc_words_str)
        #f_out.write(doc_words_str)
        doc_words_list.append(doc_words_str)
    doc_words_list_str = '\n'.join(doc_words_list)
    #print(doc_words_list_str)
    f_out.write(doc_words_list_str)
    f_out.close()
def process(filename,corpusfile,window_size):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    fin = open(corpusfile, 'r', encoding='utf-8', newline='\n', errors='ignore')
    doc_words_list=fin.readlines()
    fin.close()
    idx2graph = {}
    graph_idx = 0
    fout = open(filename+'.graph_pmi', 'wb')
    windows = []
    for doc_words in doc_words_list:
        words = doc_words.split()
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            # print(length, length - window_size + 1)
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)
                # print(window)

    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_j = window[j]
                if word_i == word_j:
                    continue
                word_pair_str = str(word_i) + ',' + str(word_j)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_i) + ',' + str(word_j)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    # pmi as weights
    #print(word_window_freq)
    #print(word_pair_count)
    num_window = len(windows)
    word_pair_pmi={}
    max_pmi=0
    max_pim_str=""
    for key in word_pair_count:
        temp = key.split(',')
        word_i = temp[0]
        word_j = temp[1]
        #word_pair_str = str(word_i) + ',' + str(word_j)
        count = word_pair_count[key]
        word_freq_i = word_window_freq[word_i]
        word_freq_j = word_window_freq[word_j]
        pmi = np.log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue

        word_pair_pmi[key]=pmi
    #word_pair_pmi=sorted(word_pair_pmi.keys())
    # for k in word_pair_pmi.keys():
    #     print(str(k))
    #     print(word_pair_pmi[k])
    # print(len(word_pair_pmi))
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]

        aspect_pos = len(text_left.split())
        aspect = lines[i + 1].lower().strip()
        aspect_len = len(aspect.split())
        matrix1, matrix2=dependency_adj_matrix2(text_left + ' ' + aspect + ' ' + text_right, word_pair_pmi)
        #adj_matrix = dependency_adj_matrix(text_left + ' ' + aspect + ' ' + text_right, word_pair_pmi)
        idx2graph[i] = [matrix1, matrix2]
    pickle.dump(idx2graph, fout)
    print('done !!!'+filename)
    fout.close()

if __name__ == '__main__':
    #process('./test_datasets/rest14_train.raw',3)
    window_size=3
    get_corpus('../datasets/acl-14-short-data/train.raw', '../datasets/acl-14-short-data/test.raw', '../datasets/acl-14-short-data')
    get_corpus('../datasets/semeval14/restaurant_train.raw','../datasets/semeval14/restaurant_test.raw','../datasets/rest14')
    get_corpus('../datasets/semeval14/laptop_train.raw', '../datasets/semeval14/laptop_test.raw','../datasets/lap14')
    get_corpus('../datasets/semeval15/restaurant_train.raw','../datasets/semeval15/restaurant_test.raw','../datasets/rest15')
    get_corpus('../datasets/semeval16/restaurant_train.raw','../datasets/semeval16/restaurant_test.raw','../datasets/rest16')
    process('../datasets/acl-14-short-data/train.raw', '../datasets/acl-14-short-data.corpus', window_size)
    process('../datasets/acl-14-short-data/test.raw', '../datasets/acl-14-short-data.corpus',window_size)
    process('../datasets/semeval14/restaurant_train.raw','../datasets/rest14.corpus', window_size)
    process('../datasets/semeval14/restaurant_test.raw','../datasets/rest14.corpus',window_size)
    process('../datasets/semeval14/laptop_train.raw','../datasets/lap14.corpus',window_size)
    process('../datasets/semeval14/laptop_test.raw','../datasets/lap14.corpus',window_size)
    process('../datasets/semeval15/restaurant_train.raw','../datasets/rest15.corpus',window_size)
    process('../datasets/semeval15/restaurant_test.raw','../datasets/rest15.corpus',window_size)
    process('../datasets/semeval16/restaurant_train.raw','../datasets/rest16.corpus',window_size)
    process('../datasets/semeval16/restaurant_test.raw','../datasets/rest16.corpus',window_size)

    # process('./new_datasets/semeval14/restaurant_test_1.raw', './datasets/rest14.corpus', window_size)
    # process('./new_datasets/semeval14/restaurant_test_2.raw', './datasets/rest14.corpus', window_size)
    # process('./new_datasets/semeval14/restaurant_test_3.raw', './datasets/rest14.corpus', window_size)
    # process('./new_datasets/semeval14/restaurant_test_4.raw', './datasets/rest14.corpus', window_size)
    # process('./new_datasets/semeval14/restaurant_test_5.raw', './datasets/rest14.corpus', window_size)
    # process('./new_datasets/semeval14/restaurant_test_6.raw', './datasets/rest14.corpus', window_size)
    # process('./new_datasets/semeval14/restaurant_test_7.raw', './datasets/rest14.corpus', window_size)
    # process('./new_datasets/semeval14/restaurant_test_12.raw', './datasets/rest14.corpus', window_size)
    # process('./new_datasets/semeval14/laptop_test_1.raw', './datasets/lap14.corpus', window_size)
    # process('./new_datasets/semeval14/laptop_test_2.raw', './datasets/lap14.corpus', window_size)
    # process('./new_datasets/semeval14/laptop_test_3.raw', './datasets/lap14.corpus', window_size)
    # process('./new_datasets/semeval14/laptop_test_4.raw', './datasets/lap14.corpus', window_size)
    # process('./new_datasets/semeval14/laptop_test_5.raw', './datasets/lap14.corpus', window_size)
    # process('./new_datasets/semeval14/laptop_test_6.raw', './datasets/lap14.corpus', window_size)
    #process('./new_datasets4/semeval15/same_polarity_aspect_rest15.raw', './datasets/rest15.corpus', window_size)
    #process('./new_datasets4/semeval16/same_polarity_aspect_rest16.raw', './datasets/rest16.corpus', window_size)
    # process('./new_datasets/semeval15/restaurant_test_1.raw', './datasets/rest15.corpus', window_size)
    # process('./new_datasets/semeval15/restaurant_test_2.raw', './datasets/rest15.corpus', window_size)
    # process('./new_datasets/semeval15/restaurant_test_3.raw', './datasets/rest15.corpus', window_size)
    # process('./new_datasets/semeval15/restaurant_test_4.raw', './datasets/rest15.corpus', window_size)
    # process('./new_datasets/semeval16/restaurant_test_1.raw', './datasets/rest16.corpus', window_size)
    # process('./new_datasets/semeval16/restaurant_test_2.raw', './datasets/rest16.corpus', window_size)
    # process('./new_datasets/semeval16/restaurant_test_3.raw', './datasets/rest16.corpus', window_size)
    # process('./new_datasets/semeval16/restaurant_test_4.raw', './datasets/rest16.corpus', window_size)



