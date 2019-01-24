import pandas as pd
import gzip
import pickle
import nltk
import sys

file = 'reviews_Clothing_Shoes_and_Jewelry_5.json.gz'
stop_words = 'stopwords'
english_punctuations = ['-', '``', "''", '"', '--', '...', ',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']



def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def process(path, category, filter):
    sw = pd.read_csv(stop_words, header=None)
    folder = './categoried_data/' + category + '/' + filter + '/'
    stop = []
    data = []
    user_purchased_items = dict()
    item_purchased_users = dict()
    
    #stop += english_punctuations+ list(sw.values[:,0])

    item_meta_dict = pickle.load(open('item_meta_dict', 'rb'))
    item_image_features_dict = pickle.load(open('item_image_features_dict', 'rb'))
    item_image_features_dict = {k.split('.')[0]:v for (k,v) in item_image_features_dict.items()}
    item_global_image_features_dict = pickle.load(open('item_global_image_features_dict', 'rb'))
    
    # first read to get category related records and user item distribution
    print 'first read begin ...'
    line_num = 0
    p = 0
    for d in parse(path):
        if line_num % 1000 == 0:
            print 'read ' + str(line_num) + ' lines'
        line_num += 1
        user = d['reviewerID']
        item = d['asin']
        category_list = item_meta_dict[item]
        if category in category_list and item in item_global_image_features_dict.keys(): # if item in item_global_image_features_dict.keys() then item will in item_image_features_dict.keys() according to get_image_global_features.py 
            print item
            if user not in user_purchased_items.keys():
                user_purchased_items[user] = [item]
            else:
                user_purchased_items[user].append(item)

            if item not in item_purchased_users.keys():
                item_purchased_users[item] = [user]
            else:
                item_purchased_users[item].append(user)
        sys.stdout.flush()
    print 'first read end ...'
    
    
    # second read to conduct filter
    print 'second read begin ...'
    line_num = 0
    for d in parse(path):
        if line_num % 1000 == 0:
            print 'read ' + str(line_num) + ' lines'
        line_num += 1
        user = d['reviewerID']
        item = d['asin']
        rating = d['overall']
        review = d['reviewText']
        review_sum = d['summary']
        if user in user_purchased_items.keys() and len(user_purchased_items[user]) > int(filter):
            if item in item_purchased_users.keys() and len(item_purchased_users[item]) > int(filter):
                data.append([user, item, rating, review, review_sum])
        sys.stdout.flush()

    t = pd.DataFrame(data)
    t.to_csv(folder + 'named_whole_data', index=False, header=None)
    print 'second read end ...'
    

    
    # third read to generate user item dicts
    print 'third read begin ...'
    item_id_dict = dict()
    user_id_dict = dict()
    id_item_dict = dict()
    id_user_dict = dict()
    categoried_item_image_features_dict = dict()
    categoried_item_global_image_features_dict = dict()

    item_number = 0
    user_number = 0
    #sw = pd.read_csv(stop_words, header=None)
    #stop = english_punctuations+sw
    
    whole_data = pd.read_csv(folder + 'named_whole_data', header=None)
    
    # generate dicts
    line_num = 0
    words = []
    words.append('_START_')
    words.append('_END_')
    for line in whole_data.values:
        if line_num % 1000 == 0:
            print 'read ' + str(line_num) + ' lines'
        line_num += 1
        user = line[0]
        item = line[1]
        review = line[3]
        review_sum = line[4]
        
        if pd.isnull(review) == False:
            review_tokens = [i.lower() for i in nltk.word_tokenize(review) if i.lower() not in stop]
        if pd.isnull(review_sum) == False:
            review_sum_tokens = [i.lower() for i in nltk.word_tokenize(review_sum) if i.lower() not in stop]
        
        for w in review_tokens:
            words.append(w)
        for w in review_sum_tokens:  
            words.append(w)  

        if user not in user_id_dict.keys():
            user_id_dict[user] = str(user_number)
            id_user_dict[str(user_number)] = user
            user_number += 1
        if item not in item_id_dict.keys():
            item_id_dict[item] = str(item_number)
            id_item_dict[str(item_number)] = item
            categoried_item_global_image_features_dict[item] = item_global_image_features_dict[item]
            categoried_item_image_features_dict[item] = item_image_features_dict[item]
            item_number += 1
        sys.stdout.flush()
    pickle.dump(user_id_dict, open(folder + 'user_id_dict', 'wb'))
    pickle.dump(id_user_dict, open(folder + 'id_user_dict', 'wb'))
    pickle.dump(item_id_dict, open(folder + 'item_id_dict', 'wb'))
    pickle.dump(categoried_item_image_features_dict, open(folder + 'c_item_image_features_dict', 'wb'))
    pickle.dump(categoried_item_global_image_features_dict, open(folder + 'c_item_global_image_features_dict', 'wb'))
    pickle.dump(id_item_dict, open(folder + 'id_item_dict', 'wb'))
    
    word_id_dict_tmp = pickle.load(open(folder + 'word_id_dict_tmp', 'r'))
    word_emb_dict_tmp = pickle.load(open(folder + 'word_emb_tmp.pkl', 'rb'))
    print word_id_dict_tmp[',']
    word_id_dict = dict()
    id_word_dict = dict()
    final_embeddings = []
    words = set(words)
    word_number = 0
    word_id_dict['_NULL_'] = str(word_number)
    final_embeddings.append([0.0]*64)
    word_number += 1
    for w in words:
        if w in word_id_dict_tmp.keys():
            word_id_dict[w] = str(word_number)
            id_word_dict[str(word_number)] = w
            final_embeddings.append(word_emb_dict_tmp[word_id_dict_tmp[w]])
            word_number += 1
        else:
            print 'not in big dict!'
    pickle.dump(word_id_dict, open(folder + 'word_id_dict', 'wb'))
    pickle.dump(id_word_dict, open(folder + 'id_word_dict', 'wb'))
    pickle.dump(final_embeddings, open(folder+'word_emb.pkl', "wb"))
    print 'check word_id_dict ...'
    print word_id_dict['_END_']
    print word_id_dict['_START_']
    print 'third read end ...'
        

    # 4th read to map real name to id
    print 'fourth read begin ...'    
    whole_data = pd.read_csv(folder + 'named_whole_data', header=None)
    data = []
    line_num = 0
    user_purchased_items = dict()
    item_purchased_users = dict()
    for line in whole_data.values:
        if line_num % 1000 == 0:
            print 'read ' + str(line_num) + ' lines'
        line_num += 1
        user = line[0]
        item = line[1]
        rating = line[2]
        review = line[3]
        review_sum = line[4]
        review_tokens = []
        review_sum_tokens = []
        if user not in user_purchased_items.keys():
            user_purchased_items[user] = [item]
        else:
            user_purchased_items[user].append(item)

        if item not in item_purchased_users.keys():
            item_purchased_users[item] = [user]
        else:
            item_purchased_users[item].append(user)

        if pd.isnull(review) == False:
            review_tokens = [i.lower() for i in nltk.word_tokenize(review) if i.lower() not in stop]
        if pd.isnull(review_sum) == False:
            review_sum_tokens = [i.lower() for i in nltk.word_tokenize(review_sum) if i.lower() not in stop]
        review_ids = word_id_dict['_START_']+'::'
        review_sum_ids = word_id_dict['_START_']+'::'
       
        for w in review_tokens:
            if w in word_id_dict.keys():
                review_ids += word_id_dict[w]+'::'
        review_ids += word_id_dict['_END_']
        
        for w in review_sum_tokens:
            if w in word_id_dict.keys():
                review_sum_ids += word_id_dict[w]+'::'
        review_sum_ids += word_id_dict['_END_']

        data.append([user_id_dict[user], item_id_dict[item], rating, review_ids, review_sum_ids])
        sys.stdout.flush()

    pickle.dump(user_purchased_items, open(folder + 'user_purchased_items', 'wb'))
    pickle.dump(item_purchased_users, open(folder + 'item_purchased_users', 'wb'))
    t = pd.DataFrame(data)
    t.to_csv(folder + 'ided_whole_data', index=False, header=None)

    print 'fourth read end ...'
    print 'user number :' + str(len(user_id_dict.items()))
    print 'item number :' + str(len(item_id_dict.items()))
    print 'word number :' + str(len(word_id_dict.items()))


category = 'Men'
filter = '0'
process(file, category, filter)
