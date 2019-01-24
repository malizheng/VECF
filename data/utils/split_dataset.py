import pandas as pd
import numpy as np
import pickle

c = 'Men'
filter = '0'

folder = './categoried_data/' + c + '/' + filter + '/'
ided_data = pd.read_csv(folder + 'ided_whole_data', header=None)
id_word_dict = pickle.load(open(folder + 'id_word_dict', 'r'))


split_ratio = 0.7
user_purchased_items = dict()
item_purchased_users = dict()
user_purchased_item_review = dict()

print len(ided_data.values)

for line in ided_data.values:
    user_id = line[0]
    item_id = line[1]
    review = line[3]
    review_ab = line[4]
    #print [id_word_dict[i] for i in review[:-2].split('::')]
    if user_id not in user_purchased_items.keys():
        user_purchased_items[user_id] = [item_id]
        user_purchased_item_review[user_id] = [str(item_id)+'||'+review]
    else:
        user_purchased_items[user_id].append(item_id)
        user_purchased_item_review[user_id].append(str(item_id)+'||'+review)

    if item_id not in item_purchased_users.keys():
        item_purchased_users[item_id] = [user_id]
    else:
        item_purchased_users[item_id].append(user_id)


#print user_purchased_items[298]
filter_num = 0
train_user_purchased_items = {k:v[:int(len(v)*split_ratio+1)] for (k, v) in user_purchased_items.items() if len(v) > filter_num}
train_user_purchased_item_review = {k:v[:int(len(v)*split_ratio+1)] for (k, v) in user_purchased_item_review.items() if len(v) > filter_num}




#print train_user_purchased_items[298]

#raw_input()



train_item_purchased_users = dict()
for (k,v) in train_user_purchased_items.items():
    for i in v:
        if i not in train_item_purchased_users.keys():
            train_item_purchased_users[i] = [k]
        else:
            train_item_purchased_users[i].append(k)




test_user_purchased_items = {k:v[int(len(v)*split_ratio+1):] for (k, v) in user_purchased_items.items() if len(v) > filter_num}
test_user_purchased_item_review = {k:v[int(len(v)*split_ratio+1):] for (k, v) in user_purchased_item_review.items() if len(v) > filter_num}


test_item_purchased_users = dict()
items_in_the_trainset = train_item_purchased_users.keys()
test_user_purchased_items = {k: [i for i in v if i in items_in_the_trainset] for (k, v) in test_user_purchased_items.items()}
test_user_purchased_item_review = {k: [i for i in v if int(i.split('||')[0]) in items_in_the_trainset] for (k, v) in test_user_purchased_item_review.items()}

for (k, v) in test_user_purchased_items.items():
    for i in v:
        if i not in items_in_the_trainset:
            print 'wtf'
        else:
            if i not in test_item_purchased_users.keys():
                test_item_purchased_users[i] = [k]
            else:
                test_item_purchased_users[i].append(k)

pickle.dump(train_user_purchased_items, open(folder + str(split_ratio) + '_train_user_purchased_items', 'wb'))
pickle.dump(train_item_purchased_users, open(folder + str(split_ratio) + '_train_item_purchased_users', 'wb'))
pickle.dump(test_user_purchased_items, open(folder + str(split_ratio) + '_test_user_purchased_items', 'wb'))
pickle.dump(test_item_purchased_users, open(folder + str(split_ratio) + '_test_item_purchased_users', 'wb'))

pickle.dump(test_user_purchased_item_review, open(folder + str(split_ratio) + '_test_user_purchased_item_review', 'wb'))
pickle.dump(train_user_purchased_item_review, open(folder + str(split_ratio) + '_train_user_purchased_item_review', 'wb'))





# split
train_data = []
test_data = []

for line in ided_data.values:
    user_id = line[0]
    item_id = line[1]
    if user_id in train_user_purchased_items.keys() and item_id in train_user_purchased_items[user_id]:
        train_data.append(line)
    elif user_id in test_user_purchased_items.keys() and item_id in test_user_purchased_items[user_id]:
        test_data.append(line)
    else:
        print 'removed records: ' + str(line)

t = pd.DataFrame(train_data)
t.to_csv(folder + str(split_ratio) + '_train_ided_whole_data', index=False, header=None)
t = pd.DataFrame(test_data)
t.to_csv(folder + str(split_ratio) + '_test_ided_whole_data', index=False, header=None)



# statistics
print 'whole user number: ' + str(len(user_purchased_items.items()))
print 'whole item number: ' + str(len(item_purchased_users.items()))

print 'train user number: ' + str(len(train_user_purchased_items.items()))
print 'train item number: ' + str(len(train_item_purchased_users.items()))

print 'test user number: ' + str(len(test_user_purchased_items.items()))
print 'test item number: ' + str(len(test_item_purchased_users.items()))
