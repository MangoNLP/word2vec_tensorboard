import tensorflow as tf
import ast
import argparse
import pickle
import os
import numpy as np
import multiprocessing
import codecs

# import matplotlib.pyplot as plt
from tensorflow.contrib.tensorboard.plugins import projector
from gensim.models import word2vec
from konlpy.corpus import kobill
from konlpy.tag import Mecab, Twitter

try:
    tag = Mecab()
except:
    tag = Twitter()

fname = 'ko_word2vec.model'

import datetime


def log(msg):
    now = datetime.datetime.now()
    nowTime = now.strftime('%H:%M:%S')
    print("[{}] {}".format(nowTime, msg))


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", help='모델 파일')
    return parser.parse_args()


def get_line(filePath):
    with codecs.open(filePath, 'r', 'utf8') as f:
        for line in f.readlines():
            yield line


if __name__ == '__main__':
    '''
    https://www.lucypark.kr/courses/2015-ba/text-mining.html#
    '''

    args = arg_parser()
    pos = lambda d: ['/'.join(p) for p in tag.pos(d)]

    if args.load is None:
        log("Train")

        # docs_ko = [kobill.open(i).read() for i in kobill.fileids()]
        # texts_ko = [pos(doc) for doc in docs_ko]

        if not os.path.exists('docs_ko.pk'):
            log('New pickling docs_ko.pk')
            docs_ko = [ast.literal_eval(i)['body'] for i in get_line('data.txt')]
            with open('docs_ko.pk', 'wb') as f:
                pickle.dump(docs_ko, f)
        else:
            log('load pickling docs_ko.pk')
            with open('docs_ko.pk', 'rb') as f:
                docs_ko = pickle.load(f)

        log("Count: {}".format(len(docs_ko)))
        log('docsko succ')

        if not os.path.exists('texts_ko.pk'):
            log("New Pickling texts_ko.pk")
            texts_ko = [pos(doc) for doc in docs_ko]
            with open('texts_ko.pk', 'wb') as f:
                pickle.dump(texts_ko, f)
        else:
            log("load pickling texts_ko.pk")
            with open('texts_ko.pk', 'rb') as f:
                texts_ko = pickle.load(f)

        log("Pos Count: {}".format(len(texts_ko)))
        log('texts_ko succ')

        log("Start train")
        wv_model_ko = word2vec.Word2Vec(texts_ko, workers=multiprocessing.cpu_count())
        wv_model_ko.init_sims(replace=True)
        wv_model_ko.save(fname)
        log('train end')

        log('start initialize for tensorboard projector')
        vocab = wv_model_ko.wv.vocab
        num_w2v = len(wv_model_ko.wv.index2word)
        w2v = np.zeros((num_w2v, 100))
        with open("metadata.tsv", 'wb+') as file_metadata:
            for i, word in enumerate(wv_model_ko.wv.index2word):
                w2v[i] = wv_model_ko[word]
                file_metadata.write((word + '\n').encode('utf8'))
        # setup a TensorFlow session
        tf.reset_default_graph()
        sess = tf.InteractiveSession()

        X = tf.Variable([0.0], name='embedding')
        place = tf.placeholder(tf.float32, shape=[None, 100])
        set_x = tf.assign(X, place, validate_shape=False)

        sess.run(tf.global_variables_initializer())
        sess.run(set_x, feed_dict={place: w2v})

        # create a TensorFlow summary writer
        summary_writer = tf.summary.FileWriter('./projector', sess.graph)
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = 'embedding:0'
        embedding_conf.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(summary_writer, config)

        # save the model
        saver = tf.train.Saver()
        saver.save(sess, './projector/model.ckpt')
        sess.close()
        log('end initialize tensorboard project')

    else:
        log("Load model")
        wv_model_ko = word2vec.Word2Vec.load(fname)

        print(wv_model_ko.most_similar_cosmul(pos(args.load)))
        print(wv_model_ko.wv.vocab['임산부/NNG'], wv_model_ko.wv.vocab['가시나/NNG'])
    log("end")
