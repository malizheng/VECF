#************************************************
# Author: xxxxxxxxx
# Usage: Code for the paper of "Personalized Fashion Recommendation with Visual Explanations based on Multi-model Attention Network"
# Date: 2019-1-27
#************************************************

import traceback
import VECFModel
import tensorflow as tf
import sys
import pickle
import time
import numpy as np
import FromMemory
import pandas as pd
import random


def shuffle(train_input_user, train_input_item, train_input_pol, train_review_input, train_review_length, train_review_output, train_input_image):
    train_records_num = len(train_input_user)
    index = np.array(range(train_records_num))
    np.random.shuffle(index)

    input_user = list(np.array(train_input_user)[index])
    input_item = list(np.array(train_input_item)[index])
    input_pol = list(np.array(train_input_pol)[index])
    review_input = list(np.array(train_review_input)[index])
    review_length = list(np.array(train_review_length)[index])
    review_output = list(np.array(train_review_output)[index])
    input_image = list(np.array(train_input_image)[index])

    return input_user, input_item, input_pol, review_input, review_length, review_output, input_image

def run(model, ratio, train_path, test_path_dict):
    cfg = tf.ConfigProto(allow_soft_placement=True)
    cfg.gpu_options.allow_growth = True
    sess = tf.Session(config=cfg)

    start_time = time.time()
    # read word embedding
    emb = pickle.load(open(model.params.word_embedding_file, "rb"))
    model.run_init_all(sess, emb)
    del emb
    print 'loading training corpus ... '
    data_loader = FromMemory.FromMemory(train_path, ratio)
    train_input_user, train_input_item, train_input_pol, train_review_input, train_review_length, train_review_output, train_input_image = \
        data_loader.train_data_generator()

    print 'loading testing corpus ... '
    test_users = pickle.load(open(test_path_dict['test_user_dict_path'], 'r'))['users']
    test_item_image_dict = pickle.load(open(test_path_dict['test_item_image_dict_path'], 'r'))
    test_items = list(test_item_image_dict.keys())
    test_images = list(test_item_image_dict.values())
    test_user_purchased_item_dict = pickle.load(open(test_path_dict['test_user_purchased_item_dict_path'], 'r'))

    print 'loading saving corpus ... '
    if model.params.is_saving:
        saving_user_purchased_item_dict = pickle.load(open(test_path_dict['saving_user_purchased_item_dict_path'], 'r'))
        saving_item_image_dict = pickle.load(open(test_path_dict['saving_item_image_dict_path'], 'r'))
        save_input_user, save_input_item, save_input_image = ([] for i in range(3))

        for k, v in saving_user_purchased_item_dict.items():
            for i in v:
                save_input_user.append(k)
                save_input_item.append(i)
                save_input_image.append(saving_item_image_dict[i])
    end_time = time.time()
    print 'files loading time: ', (end_time - start_time), ' sec'


    print '-------------------- learning begin --------------------'
    result = []
    max_value = 0
    best_result_index = 0

    # training -> evaluate performance -> save result (learned visual attention + predicted review inforamtion)
    for epoch in range(model.params.max_epoch):
        print '********************* training begin *********************'
        start_time = time.time()
        step = 0
        p_loss = 0.0
        r_loss = 0.0
        train_record_number = len(train_input_user)
        print 'training sample number: ', train_record_number
        '''
        train_input_user, train_input_item, train_input_pol, train_review_input, \
        train_review_length, train_review_output, train_input_image = \
            shuffle(train_input_user, train_input_item, train_input_pol, train_review_input, train_review_length, train_review_output, train_input_image)
        '''
        max_steps = train_record_number / model.params.batch_size
        s = time.time()
        while step <= max_steps:
            if (step + 1) * model.params.batch_size > train_record_number:
                b = train_record_number - step * model.params.batch_size
            else:
                b = model.params.batch_size
            start = step * model.params.batch_size
            end = start + b
            if end > start:
                p, r = model.run_train_step(sess, train_input_user[start:end], train_input_item[start:end],
                                            train_input_pol[start:end],
                                            train_review_input[start:end], train_review_length[start:end],
                                            train_review_output[start:end], train_input_image[start:end])
                p_loss += p
                r_loss += r
                if step % 100 == 0:
                    e = time.time()
                    print 'samples utill:', end, 'step training time: \t', (e - s), ' sec'
                    s = e
            step += 1

        end_time = time.time()
        print 'epoch:', epoch, 'p_loss: ', p_loss, ' r_loss: ', r_loss, 'training time: ', (end_time - start_time), ' sec'
        print '********************* training end *********************'


        if model.params.is_debug:
            check_value = model.get_check(sess, save_input_user[:10], save_input_image[:10])
            print "w_u"
            print check_value[0][0:5]
            print "w_i"
            print check_value[1][0:5]
            print "self.b"
            print check_value[2][0:5]
            print "self.w_out"
            print check_value[3][0:5]
            print "self.b_out"
            print check_value[4][0:5]
            print "self.embedded_u"
            print check_value[5][0:5]
            print "self.out_att"
            print check_value[6][0][:20]

        else:
            print '&&&&&&&&&&&&&&&&&&&& test begin &&&&&&&&&&&&&&&&&&&&'
            start_time = time.time()
            p_test, r_test, f1_test, hr_test, ndcg_test = performance_eval(sess, model, test_users, test_items,
                                                                           test_images, test_user_purchased_item_dict)
            end_time = time.time()
            result.append([p_test, r_test, f1_test, hr_test, ndcg_test])
            if f1_test > max_value:
                max_value = f1_test
                best_result_index = epoch
            print 'current best performance: ' + str(result[best_result_index])
            print 'epoch:', epoch, 'testing performance: ', p_test, r_test, f1_test, hr_test, ndcg_test, 'testing time: \t', (
                        end_time - start_time), ' sec'
            print '&&&&&&&&&&&&&&&&&&&& test end &&&&&&&&&&&&&&&&&&&&'


        if model.params.is_saving:
            print '&&&&&&&&&&&&&&&&&&&& saving begin &&&&&&&&&&&&&&&&&&&&'
            start_time = time.time()
            result_saving(sess, model, save_input_user, save_input_item, save_input_image)
            end_time = time.time()
            print 'saving attention time: ', (end_time - start_time), ' sec'
            print '&&&&&&&&&&&&&&&&&&&& saving end &&&&&&&&&&&&&&&&&&&&'
        sys.stdout.flush()

    print '-------------------- learning end --------------------'
    print result
    print best_result_index
    print 'final best performance: ' + str(result[best_result_index])
    return result[best_result_index]


def ndcg_per_user(recommend_list, purchased_list):
    temp = 0
    Z_u = 0

    for j in range(min(len(recommend_list), len(purchased_list))):
        Z_u = Z_u + 1 / np.log2(j + 2)

    for j in range(len(recommend_list)):
        if recommend_list[j] in purchased_list:
            temp = temp + 1 / np.log2(j + 2)

    if Z_u == 0:
        temp = 0
    else:
        temp = temp / Z_u
    return temp

def top_k_per_user(recommend_list, purchased_list):
    cross = float(len([i for i in recommend_list if i in purchased_list]))
    p = cross / len(recommend_list)
    r = cross / len(purchased_list)
    if cross > 0:
        f = 2.0 * p * r / (p + r)
    else:
        f = 0.0
    hit = 1.0 if cross > 0 else 0.0
    return p, r, f, hit

def performance_eval(sess, model, test_users, test_items, test_images, test_user_purchased_item_dict):
    user_batch_size = 1
    print 'testing user number: ', len(test_users)
    print 'testing item number: ', len(test_items)
    all_p, all_r, all_f, all_hit, all_ndcg = [],[],[],[],[]
    for u in test_users:
        u_extend = [u]*len(test_items)
        scores = model.get_test_score(sess, u_extend, test_items, test_images)
        index = np.argsort(np.array(scores))[::-1][:model.params.K]
        recommended = np.array(test_items)[index]
        gt = test_user_purchased_item_dict[u]
        p, r, f, hit = top_k_per_user(recommended, gt)
        ndcg = ndcg_per_user(recommended, gt)
        all_p.append(p)
        all_r.append(r)
        all_f.append(f)
        all_hit.append(hit)
        all_ndcg.append(ndcg)
    return np.array(all_p).mean(),np.array(all_r).mean(),np.array(all_f).mean(), np.array(all_hit).mean(),np.array(all_ndcg).mean()

def result_saving(sess, model, input_user, input_item, input_image):
    all_attention = []
    all_generated_review_id = []
    users = []
    items = []
    step = 0
    record_number = len(input_user)
    max_steps = record_number / model.params.batch_size
    while step <= max_steps:
        if (step + 1) * model.params.batch_size > record_number:
            b = record_number - step * model.params.batch_size
        else:
            b = model.params.batch_size
        start = step * model.params.batch_size
        end = start + b
        if end > start:
            attention_w, generated_review = model.get_attention_score(sess, input_user[start:end],
                                                                      input_item[start:end], input_image[start:end])
            for i in range(len(input_user[start:end])):
                all_attention.append('@@'.join([str(k) for k in attention_w[i]]))
                all_generated_review_id.append('@@'.join([str(k) for k in generated_review[i]]))
                users.append(input_user[start:end][i])
                items.append(input_item[start:end][i])
            step += 1


    path = '../../results/'
    t = pd.DataFrame(users)
    t.to_csv(path + 'all_users' + str(model.params.data_type.split('/')[0]) + str(model.params.beta), index=False, header=None)
    t = pd.DataFrame(items)
    t.to_csv(path + 'all_pre_items' + str(model.params.data_type.split('/')[0]) + str(model.params.beta), index=False, header=None)
    t = pd.DataFrame(all_attention)
    t.to_csv(path + 'all_attention' + str(model.params.data_type.split('/')[0]) + str(model.params.beta), index=False, header=None)
    t = pd.DataFrame(all_generated_review_id)
    t.to_csv(path + 'all_generated_review_id' + str(model.params.data_type.split('/')[0]) + str(model.params.beta), index=False, header=None)


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    print 'Run Cmd: ', sys.argv
    # complete parameter list, including some potentially valuable parameters
    data_type = 'Men/0' ######## need change ##########
    batch_size = 256
    global_dimension = 10
    item_dimension = 10 # should be the same as global_dimension when using mul or add merging methods
    hidden_dimension = 100
    context_dimension = 50
    word_embedding_dimension = 64
    reg = 0.0001
    learning_rate = 0.01
    optimization = 'Adam'
    word_embedding_file = '../../data/categoried_data/' + data_type + '/word_emb.pkl'

    train_path = '../../data/categoried_data/' + data_type + '/'
    test_user_dict_path = '../../data/categoried_data/' + data_type + '/test_user_dict'
    test_item_image_dict_path = '../../data/categoried_data/' + data_type + '/test_item_image_dict'
    test_user_purchased_item_dict_path = '../../data/categoried_data/' + data_type + '/filtered_test_user_purchased_items_dict'
    saving_item_image_dict_path = '../../data/categoried_data/' + data_type + '/test_item_image_dict'
    saving_user_purchased_item_dict_path = '../../data/categoried_data/' + data_type + '/filtered_test_user_purchased_items_dict'

    word_dict_path = '../../data/categoried_data/' + data_type + '/word_id_dict'
    user_dict_path = '../../data/categoried_data/' + data_type + '/user_id_dict'
    item_dict_path = '../../data/categoried_data/' + data_type + '/item_id_dict'
    max_review_length = 30 ######## need change ##########
    image_d = [196, 512]
    max_epoch = 30
    K = 10
    beta = 0.0001  # 0.001-100
    att = 10 # 0 means 1 layer;  >0 means 2 layers
    ctx2out = 0
    prev2out = 0
    is_saving = 1
    is_debug = 0
    including_image = 1



    if len(sys.argv) > 1:
        data_type = sys.argv[1]
        batch_size = int(sys.argv[2])
        global_dimension = int(sys.argv[3])
        item_dimension = int(sys.argv[4])
        hidden_dimension = int(sys.argv[5])
        context_dimension = int(sys.argv[6])
        word_embedding_dimension = int(sys.argv[7])
        reg = float(sys.argv[8])
        learning_rate = float(sys.argv[9])
        optimization = sys.argv[10]
        word_embedding_file = sys.argv[11]
        train_ep = sys.argv[12]
        test_user_dict = sys.argv[13]
        test_item_dict = sys.argv[14]
        test_image_dict = sys.argv[15]
        test_user_purchased_item_dict = sys.argv[16]
        word_dict_path = sys.argv[17]
        user_dict_path = sys.argv[18]
        item_dict_path = sys.argv[19]
        max_review_length = int(sys.argv[20])
        image_d[0] = int(sys.argv[21])
        image_d[1] = int(sys.argv[22])
        max_epoch = int(sys.argv[23])
        K = int(sys.argv[24])
        beta = float(sys.argv[25])
        att = int(sys.argv[26])
        ctx2out = int(sys.argv[27])
        prev2out = int(sys.argv[28])
        is_saving = int(sys.argv[29])
        is_debug = int(sys.argv[30])
        including_image = int(sys.argv[31])


    all_r = []
    ratio = 0.7

    test_path_dict = dict()
    test_path_dict['test_user_dict_path'] = test_user_dict_path
    test_path_dict['test_item_image_dict_path'] = test_item_image_dict_path
    test_path_dict['test_user_purchased_item_dict_path'] = test_user_purchased_item_dict_path
    test_path_dict['saving_item_image_dict_path'] = saving_item_image_dict_path
    test_path_dict['saving_user_purchased_item_dict_path'] = saving_user_purchased_item_dict_path

    try:
        for d in [50, 100, 150, 200, 250, 300]:
        #for d in [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]:
        #for d in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
            print "-----------------new experiment----------------------"
            with tf.variable_scope(str(d)):
                parameters = VECFModel.Params(
                    data_type=data_type,
                    batch_size=batch_size,
                    global_dimension=d,
                    item_dimension=d,
                    hidden_dimension=hidden_dimension,
                    context_dimension=context_dimension,
                    word_embedding_dimension=word_embedding_dimension,
                    reg=reg,
                    learning_rate=learning_rate,
                    optimization=optimization,
                    word_embedding_file=word_embedding_file,
                    word_dict_path=word_dict_path,
                    user_dict_path=user_dict_path,
                    item_dict_path=item_dict_path,
                    max_review_length=max_review_length,
                    image_d=image_d,
                    max_epoch=max_epoch,
                    K=K,
                    beta=beta,
                    att=att,
                    ctx2out=ctx2out,
                    prev2out=prev2out,
                    is_saving=is_saving,
                    is_debug=is_debug,
                    including_image=including_image,
                )
                print 'Settings = ', parameters
                model = VECFModel.VECFModel(parameters)
                r = run(model, ratio, train_path, test_path_dict)
                all_r.append(r)
        print all_r
        print 'average of 3 times : ' + str(np.array(all_r).mean(0))
        #print 'std of 10 times : ' + str(np.array(all_r).std(0))
    except Exception:
        traceback.print_exc()

    print 'Done'