

#生成seq_graph图
import numpy as np
#import spacy
import pickle

#nlp = spacy.load('en_core_web_sm')
def dependency_adj_matrix(text,pmi_v):
    text=text.strip()
    text_list = text.split()
    seq_len = len(text_list)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    flag = 1
    for i in range(seq_len):
        for r in range(pmi_v+1):
            if i+r<seq_len:
               matrix[i][i+r]=1
               matrix[i+r][i] = 1
            if i-r>=0:
              matrix[i][i-r] = 1
              matrix[i-r][i]= 1
    return matrix

def process(filename,pmi_v):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph_seq', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect_pos=len(text_left.split())
        aspect = lines[i + 1].lower().strip()
        aspect_len=len(aspect.split())
        sentence=text_left+' '+aspect+' '+text_right
        adj_matrix = dependency_adj_matrix(sentence,pmi_v)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()
    print('done !!!'+filename)
    fout.close() 

if __name__ == '__main__':
    #process('./test_datasets/rest14_train.raw',2)
    process('../datasets/acl-14-short-data/train.raw',2)
    process('../datasets/acl-14-short-data/test.raw',2)
    process('../datasets/semeval14/restaurant_train.raw',2)
    process('../datasets/semeval14/restaurant_test.raw',2)
    process('../datasets/semeval14/laptop_train.raw',2)
    process('../datasets/semeval14/laptop_test.raw',2)
    process('../datasets/semeval15/restaurant_train.raw',2)
    process('../datasets/semeval15/restaurant_test.raw',2)
    process('../datasets/semeval16/restaurant_train.raw',2)
    process('../datasets/semeval16/restaurant_test.raw',2)
    window_size=2
    # process('./new_datasets/semeval14/laptop_test_1.raw', window_size)
    # process('./new_datasets/semeval14/laptop_test_2.raw', window_size)
    # process('./new_datasets/semeval14/laptop_test_3.raw', window_size)
    # process('./new_datasets/semeval14/laptop_test_4.raw', window_size)
    # process('./new_datasets/semeval14/laptop_test_5.raw', window_size)
    # process('./new_datasets/semeval14/laptop_test_6.raw', window_size)
    # process('./new_datasets/semeval14/restaurant_test_1.raw', window_size)
    #     # process('./new_datasets/semeval14/restaurant_test_2.raw', window_size)
    #     # process('./new_datasets/semeval14/restaurant_test_3.raw', window_size)
    #     # process('./new_datasets/semeval14/restaurant_test_4.raw', window_size)
    #     # process('./new_datasets/semeval14/restaurant_test_5.raw', window_size)
    #     # process('./new_datasets/semeval14/restaurant_test_6.raw', window_size)
    #     # process('./new_datasets/semeval14/restaurant_test_7.raw', window_size)
    #     # process('./new_datasets/semeval14/restaurant_test_12.raw', window_size)
    # process('./multi_datasets/semeval14/laptop_test_1.raw', window_size)
    # process('./multi_datasets/semeval14/multi_polarity_aspect_lap14.raw', window_size)
    # process('./multi_datasets/semeval14/same_polarity_aspect_lap14.raw', window_size)
    #
    # process('./multi_datasets/semeval14/restaurant_test_1.raw', window_size)
    # process('./multi_datasets/semeval14/multi_polarity_aspect_rest14.raw', window_size)
    # process('./multi_datasets/semeval14/same_polarity_aspect_rest14.raw', window_size)
    #
    #
    # process('./multi_datasets/semeval15/restaurant_test_1.raw', window_size)
    # process('./multi_datasets/semeval15/multi_polarity_aspect_rest15.raw', window_size)
    # process('./multi_datasets/semeval15/same_polarity_aspect_rest15.raw', window_size)
    #
    # process('./multi_datasets/semeval16/restaurant_test_1.raw', window_size)
    # process('./multi_datasets/semeval16/multi_polarity_aspect_rest16.raw', window_size)
    # process('./multi_datasets/semeval16/same_polarity_aspect_rest16.raw', window_size)




