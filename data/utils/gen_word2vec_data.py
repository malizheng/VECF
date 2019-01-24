import pandas as pd
import gzip
import numpy as np
import nltk



def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def gen_word2vec_data(folder, path):
    stop_words = 'stopwords'
    sw = pd.read_csv(stop_words, header=None)
    english_punctuations = ['-', '``', "''", '"', '--', '...', ',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!',
                            '*', '@', '#', '$', '%']

    stop = []
    #stop = english_punctuations + sw.values[:,0].tolist()
    #stop = english_punctuations

    word_data = []
    for d in parse(path):
        if not pd.isnull(d['reviewText']):
            print d['reviewText']
            raw_input()
            review = '$$$$&&&&'.join(['_START_']+[i.lower() for i in nltk.word_tokenize(d['reviewText']) if i.lower() not in stop]+['_END_'])
            word_data.append(review)
        else:
            print 'meet review=nan'
        
        if not pd.isnull(d['summary']):
            review = '$$$$&&&&'.join(['_START_']+[i.lower() for i in nltk.word_tokenize(d['summary']) if i.lower() not in stop]+['_END_'])
            word_data.append(review)
        else:
            print 'meet sum review=nan'


    t = pd.DataFrame(word_data)
    t.to_csv(folder+'word_data', header=False, index=None, sep='\t')

    f_in = open(folder+'word_data', 'rb')
    f_out = gzip.open(folder+'word_data.gz', 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()



category = 'Men'
filter = '0'
folder = './categoried_data/' + category + '/' + filter + '/'

gen_word2vec_data(folder, 'reviews_Clothing_Shoes_and_Jewelry_5.json.gz')
