#************************************************
# Author: xxxxxxxxx
# Usage: Code for the paper of "Personalized Fashion Recommendation with Visual Explanations based on Multi-model Attention Network"
# Date: 2019-1-27
#************************************************

import tensorflow as tf
from collections import namedtuple
import time
import pickle
import numpy as np
import random

Params = namedtuple('Params',
                    'data_type, batch_size, global_dimension, item_dimension, hidden_dimension, '
                    'context_dimension, word_embedding_dimension, reg, learning_rate, optimization,'
                    'word_embedding_file, '
                    'word_dict_path, user_dict_path, item_dict_path, '
                    'max_review_length, image_d, max_epoch, K, beta, '
                    'att, ctx2out, prev2out, is_saving, is_debug, including_image, '
                    )
class VECFModel(object):

    def __init__(self, params):
        random.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)
        s = time.time()
        self.params = params

        self.user_dict = pickle.load(open(self.params.user_dict_path, 'r'))
        self.item_dict = pickle.load(open(self.params.item_dict_path, 'r'))
        self.word_dict = pickle.load(open(self.params.word_dict_path, 'r'))

        self.user_batch = tf.placeholder(tf.int32, [None], name="user_batch")
        self.item_batch = tf.placeholder(tf.int32, [None], name="item_batch")
        self.review_input_batch = tf.placeholder(tf.int32, shape=[None, self.params.max_review_length - 1], name='review_input_batch')
        self.review_output_batch = tf.placeholder(tf.int32, shape=[None, self.params.max_review_length - 1], name='review_output_batch')
        self.review_length_batch = tf.placeholder(tf.int32, shape=[None], name='review_length_batch')
        self.image_batch = tf.placeholder(tf.float32, shape=[None, self.params.image_d[0], self.params.image_d[1]], name='images_batch')
        self.pol_batch = tf.placeholder(tf.float32, [None], name="pol_batch")

        self.pre_mask = tf.placeholder(tf.float32, [None], name="pre_mask")
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.w_initializer = tf.random_uniform_initializer(minval=-1, maxval=1)
        self.const_initializer = tf.constant_initializer(0.0)

        self._start = self.word_dict['_START_']
        self._end = self.word_dict['_END_']
        self._null = self.word_dict['_NULL_']
        self.lstm = 1
        e = time.time()
        print 'preparing time: ', (e - s), ' sec'


        self.word_vocab_size = len(self.word_dict.items())
        self.emb_initializer = tf.placeholder(dtype=tf.float32, shape=[self.word_vocab_size, self.params.word_embedding_dimension])
        self.word_embedding = tf.Variable(self.emb_initializer, trainable=False, collections=[], name='word_embedding')

        self.user_embedding = tf.get_variable('user_embedding', [len(self.user_dict.keys()), self.params.global_dimension], initializer=self.w_initializer)
        self.item_embedding = tf.get_variable('item_embedding', [len(self.item_dict.keys()), self.params.item_dimension], initializer=self.w_initializer)
        self.user_bais = tf.get_variable('user_bias', [len(self.user_dict.keys())], initializer=self.w_initializer)
        self.item_bais = tf.get_variable('item_bias', [len(self.item_dict.keys())], initializer=self.w_initializer)

        self.layer_w = tf.get_variable('layer_w', [self.params.global_dimension, self.params.global_dimension/2], initializer=self.w_initializer)
        self.layer_b = tf.get_variable('layer_b', [self.params.global_dimension/2], initializer=self.w_initializer)


        print('building model begin ...')
        s = time.time()
        self.embedded_u = tf.nn.embedding_lookup(self.user_embedding, self.user_batch, name='user_embed')
        self.embedded_i = tf.nn.embedding_lookup(self.item_embedding, self.item_batch, name='item_embed')
        self.embedded_u_b = tf.nn.embedding_lookup(self.user_bais, self.user_batch, name='user_embed_b')
        self.embedded_i_b = tf.nn.embedding_lookup(self.item_bais, self.item_batch, name='item_embed_b')
        self.embedded_review_input = tf.nn.embedding_lookup(self.word_embedding, self.review_input_batch)
        mask = tf.to_float(tf.not_equal(self.review_input_batch, int(self._null)))
        with tf.device('/gpu:1'):
            # likeness_loss
            self.attentioned_image, self.attention_weights = self.attention_on_image(self.embedded_u, self.image_batch)
            self.new_item_embedding = self.rating_merge_item_image(self.embedded_i, self.attentioned_image)
            if self.params.including_image:
                self.pre_pol = self.predict(self.embedded_u, self.new_item_embedding)
            else:
                self.pre_pol = self.predict(self.embedded_u, self.embedded_i)
            # Key Point
            self.pol_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.pol_batch, logits=self.pre_pol, name='sigmoid'))

            # review_loss
            self.review_loss = 0.0
            if self.lstm == 1:
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.params.hidden_dimension, state_is_tuple=True)
            else:
                gru_cell = tf.contrib.rnn.GRUCell(num_units=self.params.hidden_dimension)
            self.c, self.h = self.get_initial_lstm(self.embedded_u, self.embedded_i)
            for t in range(self.params.max_review_length - 1):
                if t != 0:
                    self.context = self.word_generation_context(self.embedded_u, self.embedded_i, self.h, self.attentioned_image, reuse=(t!=1))
                else:
                    batch_size = tf.shape(self.embedded_u)[0]
                    self.context = tf.zeros([batch_size, self.params.context_dimension])
                with tf.variable_scope('lstm', reuse=(t!=0)):
                    if self.lstm == 1:
                        _, (self.c, self.h) = lstm_cell(inputs=tf.concat([self.embedded_review_input[:, t, :], self.context], 1), state=[self.c, self.h])
                    else:
                        _, self.h = gru_cell(inputs=tf.concat([self.embedded_review_input[:, t, :], self.context], 1),state=self.h)
                self.logits = self.hidden2logits(self.embedded_review_input[:, t, :], self.context, self.h, reuse=(t!=0))
                # Key Point
                self.review_loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.review_output_batch[:, t]) * mask[:,t])

            # total_loss
            with tf.variable_scope('loss'):
                self.pol_review_loss = tf.add(self.pol_loss, self.params.beta * self.review_loss)
                self.regular_loss = tf.add(self.params.reg * tf.nn.l2_loss(self.user_embedding), self.params.reg * tf.nn.l2_loss(self.item_embedding))
                self.total_loss = tf.add(self.pol_review_loss, self.regular_loss)

            # training operation
            with tf.variable_scope('train_operation'):
                allvars = tf.trainable_variables()
                self.names_all = [str(d.name) for d in allvars]
                print self.names_all
                if self.params.optimization == 'Adam':
                    optimizer = tf.train.AdamOptimizer(self.params.learning_rate)
                else:
                    optimizer = tf.train.GradientDescentOptimizer(self.params.learning_rate)
                params = [v for v in allvars]
                self.names_act = [str(d.name) for d in params]
                self.train_op = optimizer.minimize(self.total_loss, var_list=params, global_step=self.global_step, name='train_step')

            # review predictor
            tf.get_variable_scope().reuse_variables()
            self.generate_review_no_beam(20)
            time_consuming = str(time.time() - s)
            print ('building model end ... time consuming: ' + time_consuming)

    def predict(self, user_embedding, item_embedding, mode='linear', concat=1):
        if mode == 'linear':
            if concat:
                rating_num_layes = 4
                with tf.variable_scope('rating_layer_weight'):
                    layer_w_u_rating = tf.get_variable('layer_w_u_rating', [self.params.global_dimension, self.params.hidden_dimension], initializer=self.w_initializer)
                    layer_w_v_rating = tf.get_variable('layer_w_v_rating', [self.params.global_dimension, self.params.hidden_dimension], initializer=self.w_initializer)
                    layer_b_rating = tf.get_variable('layer_b_rating', [self.params.hidden_dimension], initializer=self.w_initializer)

                    layer_ws_rating = []
                    layer_bs_rating = []
                    for i in range(rating_num_layes):
                        layer_ws_rating.append(tf.get_variable('layer_w_rating_' + str(i), [self.params.hidden_dimension, self.params.hidden_dimension], initializer=self.w_initializer))
                        layer_bs_rating.append(tf.get_variable('layer_b_rating_' + str(i), [self.params.hidden_dimension], initializer=self.w_initializer))

                    layer_w_out_rating = tf.get_variable('layer_w_out_rating', [self.params.hidden_dimension, 1], initializer=self.w_initializer)
                    layer_b_out_rating = tf.get_variable('layer_b_out_rating', [1], initializer=self.w_initializer)


                h_layers = []
                h_r = tf.nn.sigmoid(tf.matmul(user_embedding, layer_w_u_rating) + tf.matmul(item_embedding, layer_w_v_rating) + layer_b_rating)
                l = h_r
                h_layers.append(l)
                for i in range(rating_num_layes):
                    l = tf.nn.sigmoid(tf.matmul(l, layer_ws_rating[i]) + layer_bs_rating[i])
                    h_layers.append(l)
                result = tf.matmul(h_layers[-1], layer_w_out_rating) + layer_b_out_rating
                result = tf.reduce_sum(result, axis=1)
                #user_item_merge = tf.concat([user_embedding, item_embedding], axis=1)
                #result = tf.reduce_sum(user_item_merge, axis=1)
            else:
                user_item_merge = tf.multiply(user_embedding, item_embedding)
                result = tf.reduce_sum(user_item_merge, axis=1)
            return result
        elif mode == 'FM':
            if concat:
                user_item_merge = tf.concat([user_embedding, item_embedding], axis=1)
                FM_W = tf.get_variable('FM_W', [2 * self.params.global_dimension, 1], initializer=self.w_initializer)
                FM_V = tf.get_variable('FM_V', [2 * self.params.global_dimension, 3], initializer=self.w_initializer)
            else:
                user_item_merge = tf.multiply(user_embedding, item_embedding)
                FM_W = tf.get_variable('FM_W', [self.params.global_dimension, 1], initializer=self.w_initializer)
                FM_V = tf.get_variable('FM_V', [self.params.global_dimension, 3], initializer=self.w_initializer)

            fir_inter = tf.squeeze(tf.matmul(user_item_merge, FM_W), [1])
            sec_inter = 0.5 * tf.reduce_sum(
                tf.square(tf.matmul(user_item_merge, FM_V)) - tf.matmul(tf.square(user_item_merge), tf.square(FM_V)), 1)
            result = fir_inter + sec_inter
            return result



    def get_initial_lstm(self, user_embedding, item_embedding, reuse=False):
        with tf.variable_scope('get_initial_lstm', reuse=reuse):
            u = tf.get_variable('u', [self.params.global_dimension, self.params.hidden_dimension], initializer=self.w_initializer)
            i = tf.get_variable('i', [self.params.item_dimension, self.params.hidden_dimension], initializer=self.w_initializer)
            h = tf.nn.tanh(tf.matmul(user_embedding, u) + tf.matmul(item_embedding,i))
            c = tf.nn.tanh(tf.matmul(user_embedding, u) + tf.matmul(item_embedding,i))
            #batch_size = tf.shape(self.embedded_u)[0]
            #h = tf.zeros([batch_size, self.params.hidden_dimension])
            #c = tf.zeros([batch_size, self.params.hidden_dimension])
            return c, h

    def run_init_all(self, sess, pre_trained_emb):
        sess.run(self.word_embedding.initializer, feed_dict={self.emb_initializer: pre_trained_emb})
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init_g)
        sess.run(init_l)

    def attention_on_image(self, user_embeddings, image_features):
        # Key Point
        if self.params.att == 0:
            with tf.variable_scope('attention_on_image'):
                self.w_u = tf.get_variable('w_u', [self.params.global_dimension, 1], initializer=self.w_initializer)
                self.w_i = tf.get_variable('w_i', [self.params.image_d[1], 1], initializer=self.w_initializer)
                self.b = tf.get_variable('b', [1], initializer=self.w_initializer)

                w_image = tf.reshape(image_features, [-1, self.params.image_d[1]])
                w_image = tf.matmul(w_image, self.w_i)
                w_image = tf.reshape(w_image, [-1, self.params.image_d[0], 1])

                self.out_att = tf.nn.relu(tf.reduce_sum(w_image, 2) + tf.matmul(user_embeddings, self.w_u) + self.b)
                attentioned_image_weights = tf.nn.softmax(self.out_att)
                attentioned_image = tf.reduce_sum(image_features * tf.expand_dims(attentioned_image_weights, 2), 1,
                                                  name='attentioned_image')
                return attentioned_image, attentioned_image_weights
        else:
            with tf.variable_scope('attention_on_image'):
                self.w_u = tf.get_variable('w_u', [self.params.global_dimension, self.params.att],
                                      initializer=self.w_initializer)
                self.w_i = tf.get_variable('w_i', [self.params.image_d[1], self.params.att], initializer=self.w_initializer)
                self.b = tf.get_variable('b', [self.params.att], initializer=self.w_initializer)
                self.w_out = tf.get_variable('attention_w_out', [self.params.att, 1], initializer=self.w_initializer)
                self.b_out = tf.get_variable('attention_b_out', [1], initializer=self.w_initializer)

                w_image = tf.reshape(image_features, [-1, self.params.image_d[1]])
                w_image = tf.matmul(w_image, self.w_i)
                w_image = tf.reshape(w_image, [-1, self.params.image_d[0], self.params.att])
                user_att = tf.expand_dims(tf.matmul(user_embeddings, self.w_u), 1)
                out_att = tf.nn.relu(tf.multiply(w_image, user_att) + 0*self.b)


                out_att = tf.reshape(out_att, [-1, self.params.att])
                out_att = tf.matmul(out_att, self.w_out) + 0*self.b_out
                self.out_att = tf.reshape(out_att, [-1, self.params.image_d[0], 1])

                self.attentioned_image_weights = tf.nn.softmax(tf.reduce_sum(self.out_att, 2))
                attentioned_image = tf.reduce_sum(
                    tf.multiply(image_features, tf.expand_dims(self.attentioned_image_weights, 2)), 1,
                    name='attentioned_image')
                return attentioned_image, self.attentioned_image_weights


    def word_generation_context(self, user_embeddings, item_embeddings, h_embeddings, rating_attentioned_image, reuse=False):
        with tf.variable_scope('word_generation_context', reuse=reuse):
            gamma_h_w = tf.get_variable('gamma_h_w', [self.params.hidden_dimension, 1], initializer=self.w_initializer)
            context_w_u = tf.get_variable('context_w_u', [self.params.global_dimension, self.params.context_dimension], initializer=self.w_initializer)
            context_w_i = tf.get_variable('context_w_i', [self.params.item_dimension, self.params.context_dimension], initializer=self.w_initializer)
            context_w_v = tf.get_variable('context_w_v', [self.params.image_d[1], self.params.context_dimension], initializer=self.w_initializer)
            context_b = tf.get_variable('context_b', [self.params.context_dimension], initializer=self.w_initializer)

            gamma = tf.sigmoid(tf.matmul(h_embeddings, gamma_h_w))
            gamma_mul_image = tf.multiply(tf.matmul(rating_attentioned_image, context_w_v), gamma)
            gamma_mul_user_item = tf.multiply(tf.matmul(user_embeddings, context_w_u)+tf.matmul(item_embeddings, context_w_i), 1-gamma)
            if self.params.including_image:
                context = tf.nn.relu(gamma_mul_image + gamma_mul_user_item + context_b)
            else:
                context = tf.nn.relu(tf.matmul(user_embeddings, context_w_u)+tf.matmul(item_embeddings, context_w_i) + context_b)
            return context

    def rating_merge_item_image(self, item_embeddings, image_features, mode='mul'):
        with tf.variable_scope('rating_merge_item_image', reuse=False):
            # Key Point
            k = 0.5
            w_am = tf.get_variable('w_am', [self.params.image_d[1], self.params.global_dimension], initializer=self.w_initializer)
            w_concat = tf.get_variable('w_concat', [self.params.image_d[1], self.params.global_dimension - self.params.item_dimension], initializer=self.w_initializer)
            if mode == 'add':
                result = tf.add(k*item_embeddings, (1-k)*tf.matmul(image_features, w_am))
            elif mode == 'mul':
                result = tf.multiply(item_embeddings, 0.01*tf.matmul(image_features, w_am))
            elif mode == 'concat':
                result = tf.concat([item_embeddings, k*tf.matmul(image_features, w_concat)], 1)
            return result

    def hidden2logits(self, word_embedding, context, h, drop_out=1.0, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_out = tf.get_variable('w_out', [self.params.hidden_dimension, self.word_vocab_size], initializer=self.w_initializer)
            b_out = tf.get_variable('b_out', [self.word_vocab_size], initializer=self.const_initializer)
            h_logits = tf.nn.dropout(h, drop_out)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits

    def run_train_step(self, sess, input_user, input_item, input_pol, review_input, review_length, review_output, input_image):
        act_return = [self.train_op, self.pol_loss, self.review_loss]
        _, p, r = sess.run(act_return, feed_dict={
                                      self.user_batch: input_user,
                                      self.item_batch: input_item,
                                      self.pol_batch: input_pol,
                                      self.review_input_batch: review_input,
                                      self.review_length_batch: review_length,
                                      self.review_output_batch: review_output,
                                      self.image_batch: input_image,
                                  })
        return p, r

    def generate_review_no_beam(self, max_len=10):
        self.xs = []
        self.hs = []
        self.ls = []

        with tf.device('/gpu:1'):
            c, h = self.get_initial_lstm(self.embedded_u, self.embedded_i, reuse=True)
            sampled_word_list = []
            if self.lstm == 1:
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.params.hidden_dimension)
            else:
                gru_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.hidden_dimension)

            for t in range(max_len):
                if t == 0:
                    x = tf.nn.embedding_lookup(self.word_embedding, tf.to_int32(tf.fill([tf.shape(self.image_batch)[0]], int(self._start))))
                    batch_size = tf.shape(self.embedded_u)[0]
                    context = tf.zeros([batch_size, self.params.context_dimension])
                else:
                    x = tf.nn.embedding_lookup(self.word_embedding, sampled_word)
                    context = self.word_generation_context(self.embedded_u, self.embedded_i, h, self.attentioned_image)

                #context = self.word_generation_context(self.embedded_u, self.embedded_i, h, self.attentioned_image)

                with tf.variable_scope('lstm', reuse=True):
                    if self.lstm == 1:
                        _, (c, h) = lstm_cell(inputs=tf.concat([x, context], 1), state=[c, h])
                    else:
                        _, h = gru_cell(inputs=tf.concat([x, context], 1), state=h)

                logits = self.hidden2logits(x, context, h, reuse=True)
                self.xs.append(x)
                self.hs.append(h)
                self.ls.append(logits)
                sampled_word = tf.argmax(logits, 1)
                sampled_word_list.append(sampled_word)
            self.generated_review_id = tf.transpose(tf.stack(sampled_word_list), (1, 0))

    def generate_review_beam(self, max_len=10):
        self.xs = []
        self.hs = []
        self.ls = []

        with tf.device('/gpu:1'):
            c, h = self.get_initial_lstm(self.embedded_u, self.embedded_i)
            sampled_word_list = []
            if self.lstm == 1:
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.params.hidden_dimension)
            else:
                gru_cell = tf.nn.rnn_cell.GRUCell(num_units=self.params.hidden_dimension)

            for t in range(max_len):
                if t == 0:
                    x = tf.nn.embedding_lookup(self.word_embedding, tf.to_int32(tf.fill([tf.shape(self.image_batch)[0]], int(self._start))))
                else:
                    x = tf.nn.embedding_lookup(self.word_embedding, sampled_word)

                context = self.word_generation_context(self.embedded_u, self.embedded_i, h, self.attentioned_image)

                with tf.variable_scope('lstm', reuse=(t != 0)):
                    if self.lstm == 1:
                        _, (c, h) = lstm_cell(inputs=tf.concat([x, context], 1), state=[c, h])
                    else:
                        _, h = gru_cell(inputs=tf.concat([x, context], 1), state=h)

                logits = self.hidden2logits(x, context, h, reuse=(t != 0))
                self.xs.append(x)
                self.hs.append(h)
                self.ls.append(logits)
                sampled_word = tf.argmax(logits, 1)
                sampled_word_list.append(sampled_word)
            self.generated_review_id = tf.transpose(tf.stack(sampled_word_list), (1, 0))


    def get_test_score(self, sess, input_user, input_item, input_image):
        score = sess.run(self.pre_pol,
                        feed_dict={self.user_batch: input_user,
                                   self.item_batch: input_item,
                                   self.image_batch: input_image})
        return score

    def get_check(self, sess, users, images):
        value = sess.run([self.w_u, self.w_i, self.b, self.w_out, self.b_out, self.embedded_u, self.out_att],
                         feed_dict={self.user_batch:users, self.image_batch:images})
        return value

    def get_attention_score(self, sess, input_user, input_item, input_image):
        attention_w, generated_review = sess.run([self.attention_weights, self.generated_review_id],
                                                 feed_dict={self.user_batch: input_user,
                                                            self.item_batch: input_item,
                                                            self.image_batch: input_image})
        return attention_w, generated_review






