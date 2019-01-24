import time
import numpy as np
import pickle
import random

class FromMemory():

    def __init__(self, path, ratio):
        random.seed(0)
        np.random.seed(0)
        s = time.time()
        self.item_image_features_dict = pickle.load(open(path+'c_item_image_features_dict', 'rb'))
        self.id_item_dict = pickle.load(open(path+'id_item_dict', 'rb'))
        test_item_image_dict_path = path + '/test_item_image_dict'
        candidate_items_dict = pickle.load(open(test_item_image_dict_path, 'r'))
        self.candidate_items = list(candidate_items_dict.keys())
        self.word_id_dict = pickle.load(open(path+'word_id_dict', 'rb'))
        self.train_user_purchsed_items_dict = pickle.load(open(path+str(ratio) + '_train_user_purchased_items', 'rb'))
        self.train_user_purchsed_item_review_dict = pickle.load(open(path + str(ratio) + '_train_user_purchased_item_review', 'rb'))
        time_consuming = str(time.time() - s)
        print(time_consuming)

    def gen_train_neg(self, user_id):
        candidates = [int(i) for i in self.candidate_items if i not in self.train_user_purchsed_items_dict[int(user_id)]]
        neg_item = np.random.choice(candidates)
        return neg_item

    def train_data_generator(self):
        user_batch = []
        item_batch = []
        pol_batch = []
        input_review_batch = []
        review_length_batch = []
        output_review_batch = []
        image_batch = []

        train_records_num = 0
        max_review_words = 30
        neg_num = 1
        image_d = [196, 512]
        null = int(self.word_id_dict['_NULL_'])
        s = time.time()
        for user, v in self.train_user_purchsed_item_review_dict.items():
            for item_review in v:
                if train_records_num % 5000 == 0:
                    e = time.time()
                    print 'loading number: ', train_records_num,  'cost time: ', (e - s), 'sec'
                    s = e
                user_id = int(user)
                item_id = int(item_review.split('||')[0])
                review = item_review.split('||')[1]
                review_no_pad = [int(i) for i in str(review).split('::')]
                if len(review_no_pad) >= 1:
                    real_length = len(review_no_pad)
                    if real_length < max_review_words:
                        review_input = review_no_pad[:-1] + [null] * (max_review_words - real_length)
                        review_output = review_no_pad[1:] + [null] * (max_review_words - real_length)
                        length = real_length - 1
                    else:
                        review_input = review_no_pad[:max_review_words - 1]
                        review_output = review_no_pad[1:max_review_words]
                        length = max_review_words - 1
                else:
                    review_input = [null] * (max_review_words - 1)
                    review_output = [null] * (max_review_words - 1)
                    length = 0
                image = self.item_image_features_dict[self.id_item_dict[str(item_id)]]["conv5"].transpose(0, 2, 3, 1)[
                    0].reshape([-1, image_d[1]])
                image = 0.01 * image

                user_batch.append(int(user_id))
                item_batch.append(int(item_id))
                pol_batch.append(1.0)
                input_review_batch.append(review_input)
                output_review_batch.append(review_output)
                review_length_batch.append(length)
                image_batch.append(image)
                train_records_num += 1

                for i in range(neg_num):
                    neg_item_id = self.gen_train_neg(user_id)
                    image = \
                    self.item_image_features_dict[self.id_item_dict[str(neg_item_id)]]["conv5"].transpose(0, 2, 3, 1)[
                        0].reshape([-1, image_d[1]])
                    image = 0.01 * image

                    user_batch.append(int(user_id))
                    item_batch.append(int(neg_item_id))
                    pol_batch.append(0.0)
                    input_review_batch.append([null] * (max_review_words - 1))
                    output_review_batch.append([null] * (max_review_words - 1))
                    review_length_batch.append(0)
                    image_batch.append(image)
                    train_records_num += 1


        return user_batch, item_batch, pol_batch, input_review_batch, review_length_batch, output_review_batch, image_batch
