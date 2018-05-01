#!/usr/bin/env python

from argparse import ArgumentParser
import codecs
from collections import Counter
from functools import partial
import logging
from math import log
import os.path
import pickle as p
from random import shuffle

import msgpack
import numpy as np
from scipy import sparse

from util import listify


logger = logging.getLogger("glove")


def parse_args():  # 命令行解析
    parser = ArgumentParser(
        description=('Build a GloVe vector-space model from the '
                     'provided corpus'))

    parser.add_argument('corpus', metavar='corpus_path',
                        type=partial(codecs.open, encoding='utf-8'))

    g_vocab = parser.add_argument_group('Vocabulary options')
    g_vocab.add_argument('--vocab-path',
                         help=('Path to vocabulary file. If this path '
                               'exists, the vocabulary will be loaded '
                               'from the file. If it does not exist, '
                               'the vocabulary will be written to this '
                               'file.'))

    g_cooccur = parser.add_argument_group('Cooccurrence tracking options')
    g_cooccur.add_argument('--cooccur-path',
                           help=('Path to cooccurrence matrix file. If '
                                 'this path exists, the matrix will be '
                                 'loaded from the file. If it does not '
                                 'exist, the matrix will be written to '
                                 'this file.'))
    g_cooccur.add_argument('-w', '--window-size', type=int, default=10,
                           help=('Number of context words to track to '
                                 'left and right of each word'))
    g_cooccur.add_argument('--min-count', type=int, default=10,
                           help=('Discard cooccurrence pairs where at '
                                 'least one of the words occurs fewer '
                                 'than this many times in the training '
                                 'corpus'))

    g_glove = parser.add_argument_group('GloVe options')
    g_glove.add_argument('--vector-path',
                         help=('Path to which to save computed word '
                               'vectors'))
    g_glove.add_argument('-s', '--vector-size', type=int, default=100,
                         help=('Dimensionality of output word vectors'))
    g_glove.add_argument('--iterations', type=int, default=25,
                         help='Number of training iterations')
    g_glove.add_argument('--learning-rate', type=float, default=0.05,
                         help='Initial learning rate')
    g_glove.add_argument('--save-often', action='store_true', default=False,
                         help=('Save vectors after every training '
                               'iteration'))

    return parser.parse_args()


def get_or_build(path, build_fn, *args, **kwargs):
    """
    Load from serialized form or build an object, saving the built
    object.

    Remaining arguments are provided to `build_fn`.
    """

    save = False
    obj = None

    if path is not None and os.path.isfile(path):
        with open(path, 'rb') as obj_f:
            obj = msgpack.load(obj_f, use_list=False, encoding='utf-8')
    else:
        save = True

    if obj is None:
        obj = build_fn(*args, **kwargs)

        if save and path is not None:
            with open(path, 'wb') as obj_f:
                msgpack.dump(obj, obj_f)

    return obj


def build_vocab(corpus):  # 使用语料建立词典
    """
    Build a vocabulary with word frequencies for an entire corpus.

    Returns a dictionary `w -> (i, f)`, mapping word strings to pairs of
    word ID and word corpus frequency.
    """

    logger.info("Building vocab from corpus")

    vocab = Counter()
    for line in corpus:
        tokens = line.strip().split()
        vocab.update(tokens)

    logger.info("Done building vocab from corpus.")

    return {word: (i, freq) for i, (word, freq) in enumerate(vocab.items())}


@listify
def build_cooccur(vocab, corpus, window_size=10, min_count=None):
    """
    Build a word co-occurrence list for the given corpus.

    This function is a tuple generator, where each element (representing
    a cooccurrence pair) is of the form

        (i_main, i_context, cooccurrence)

    where `i_main` is the ID of the main word in the cooccurrence and
    `i_context` is the ID of the context word, and `cooccurrence` is the
    `X_{ij}` cooccurrence value as described in Pennington et al.
    (2014).

    If `min_count` is not `None`, cooccurrence pairs where either word
    occurs in the corpus fewer than `min_count` times are ignored.
    """

    vocab_size = len(vocab)
    id2word = dict((i, word) for word, (i, _) in vocab.items())

    # Collect cooccurrences internally as a sparse matrix for passable
    # indexing speed; we'll convert into a list later
    cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),
                                      dtype=np.float64)
    # lil_matrix则是使用两个列表存储非0元素。data保存每行中的非零元素,rows保存非零元素所在的列
    # 这种格式也很适合逐个添加元素，并且能快速获取行相关的数据。

    # 对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
    for i, line in enumerate(corpus):
        if i % 1000 == 0:
            logger.info("Building cooccurrence matrix: on line %i", i)

        tokens = line.strip().split()  # 以空格为标准分出每个单词
        token_ids = [vocab[word][0] for word in tokens]

        for center_i, center_id in enumerate(token_ids):
            # 将所有单词ID收集在中心词的左侧窗口中
            context_ids = token_ids[max(0, center_i - window_size): center_i]
            contexts_len = len(context_ids)

            for left_i, left_id in enumerate(context_ids):
                distance = contexts_len - left_i  # 与中心词之间的距离

                increment = 1.0 / float(distance)  # 单词之间距离的倒数加权

                # 对称地构建共生矩阵（假设我们也在计算正确的上下文）
                cooccurrences[center_id, left_id] += increment
                cooccurrences[left_id, center_id] += increment

    # 产生元组序列（挖掘到LiL-matrix内部来快速遍历所有非零单元格）
    for i, (row, data) in enumerate(zip(cooccurrences.rows,
                                                   cooccurrences.data)):
        if min_count is not None and vocab[id2word[i]][1] < min_count:
            continue

        for data_idx, j in enumerate(row):
            if min_count is not None and vocab[id2word[j]][1] < min_count:
                continue

            yield i, j, data[data_idx]


def run_iter(vocab, data, learning_rate=0.05, x_max=100, alpha=0.75):
    """
    Run a single iteration of GloVe training using the given
    cooccurrence data and the previously computed weight vectors /
    biases and accompanying gradient histories.

    `data` is a pre-fetched data / weights list where each element is of
    the form

        (v_main, v_context,
         b_main, b_context,
         gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context,
         cooccurrence)

    as produced by the `train_glove` function. Each element in this
    tuple is an `ndarray` view into the data structure which contains
    it.

    See the `train_glove` function for information on the shapes of `W`,
    `biases`, `gradient_squared`, `gradient_squared_biases` and how they
    should be initialized.

    The parameters `x_max`, `alpha` define our weighting function when
    computing the cost for two word pairs; see the GloVe paper for more
    details.

    Returns the cost associated with the given weight assignments and
    updates the weights by online AdaGrad in place.
    """

    global_cost = 0

    # 随机迭代数据，以免无意中偏向单词向量内容
    shuffle(data)

    for (v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context, cooccurrence) in data:

        weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1

        # 计算成本函数的内部成分，用于总成本计算和梯度计算
        #
        #   $$ J' = w_i^Tw_j + b_i + b_j - log(X_{ij}) $$
        cost_inner = (v_main.dot(v_context)
                      + b_main[0] + b_context[0]
                      - log(cooccurrence))

        # 计算损失函数
        #
        #   $$ J = f(X_{ij}) (J')^2 $$
        cost = weight * (cost_inner ** 2)

        # Add weighted cost to the global cost tracker
        # 为全局损失追踪器添加加权成本
        global_cost += 0.5 * cost

        # 计算词向量梯度
        #
        # 注意：`main_word`只是`W`的视图（不是副本），所以我们这里的修改会影响全局权重矩阵
        # 同样适用于context_word, biases等.
        grad_main = weight * cost_inner * v_context
        grad_context = weight * cost_inner * v_main

        # 计算偏差项的梯度
        grad_bias_main = weight * cost_inner
        grad_bias_context = weight * cost_inner

        # 执行自适应更新
        v_main -= (learning_rate * grad_main / np.sqrt(gradsq_W_main))
        v_context -= (learning_rate * grad_context / np.sqrt(gradsq_W_context))

        b_main -= (learning_rate * grad_bias_main / np.sqrt(gradsq_b_main))
        b_context -= (learning_rate * grad_bias_context / np.sqrt(
                gradsq_b_context))

        # Update squared gradient sums
        gradsq_W_main += np.square(grad_main)
        gradsq_W_context += np.square(grad_context)
        gradsq_b_main += grad_bias_main ** 2
        gradsq_b_context += grad_bias_context ** 2

    return global_cost


def train_glove(vocab, cooccurrences, iter_callback=None, vector_size=100,
                iterations=25, **kwargs):
    """
    Train GloVe vectors on the given generator `cooccurrences`, where
    each element is of the form

        (word_i_id, word_j_id, x_ij)

     其中`x_ij`是一个共生值$ X_ {ij} $

     如果`iter_callback`不是`None`，则提供的函数将会是
     在每次迭代之后用迄今为止学习的“W”矩阵调用。

     关键字参数被传递到迭代步骤函数
    `run_iter`。

     返回计算的字矢量矩阵`W`。
    """

    vocab_size = len(vocab)

    # 字向量矩阵。 这个矩阵的大小是（2V）*d，其中N是语料库词汇的大小，d是词向量的维数。
    # 所有元素都在范围（-0.5，0.5）中随机初始化，我们为每个单词构建两个单词向量：一个单词作为主（中心）单词，另一个单词作为上下文单词。

    # 由用户决定如何处理所产生的两个向量。
    # 为每个单词添加或平均这两个词，或丢弃上下文向量。
    W = (np.random.rand(vocab_size * 2, vector_size) - 0.5) / float(vector_size + 1)

    # 偏置项，每项与单个矢量相关联。 一个大小为$ 2V $的数组，在范围（-0.5,0.5）内随机初始化。
    biases = (np.random.rand(vocab_size * 2) - 0.5) / float(vector_size + 1)

    # 训练通过自适应梯度下降（AdaGrad）完成。 为了做到这一点，我们需要存储所有先前渐变的平方和。
    #
    # Like `W`, this matrix is (2V) * d.
    #
    # 将所有平方梯度和初始化为1，这样我们的初始自适应学习率就是全局学习率。
    gradient_squared = np.ones((vocab_size * 2, vector_size),
                               dtype=np.float64)

    # 偏差项的平方梯度之和。
    gradient_squared_biases = np.ones(vocab_size * 2, dtype=np.float64)

    # 从给定的cooccurrence generator生成可重复使用的列表，预取所有必要的数据。
    #
    # 注意：这些都是实际数据矩阵的视图，因此它们的更新将传递给真实的数据结构
    #
    # （我们甚至将单元素偏置提取为切片，以便我们将它们用作视图）
    data = [(W[i_main], W[i_context + vocab_size],
             biases[i_main: i_main + 1],
             biases[i_context + vocab_size: i_context + vocab_size + 1],
             gradient_squared[i_main], gradient_squared[i_context + vocab_size],
             gradient_squared_biases[i_main: i_main + 1],
             gradient_squared_biases[i_context + vocab_size
                                     : i_context + vocab_size + 1],
             cooccurrence)
            for i_main, i_context, cooccurrence in cooccurrences]

    for i in range(iterations):
        logger.info("\tBeginning iteration %i..", i)

        cost = run_iter(vocab, data, **kwargs)

        logger.info("\t\tDone (cost %f)", cost)

        if iter_callback is not None:
            iter_callback(W)

    return W


def save_model(W, path):
    with open(path, 'wb') as vector_f:
        p.dump(W, vector_f, protocol=2)

    logger.info("Saved vectors to %s", path)


def main(arguments):
    corpus = arguments.corpus

    logger.info("Fetching vocab..")
    vocab = get_or_build(arguments.vocab_path, build_vocab, corpus)
    logger.info("Vocab has %i elements.\n", len(vocab))

    logger.info("Fetching cooccurrence list..")
    corpus.seek(0)
    cooccurrences = get_or_build(arguments.cooccur_path,
                                 build_cooccur, vocab, corpus,
                                 window_size=arguments.window_size,
                                 min_count=arguments.min_count)
    logger.info("Cooccurrence list fetch complete (%i pairs).\n",
                len(cooccurrences))

    if arguments.save_often:
        iter_callback = partial(save_model, path=arguments.vector_path)
    else:
        iter_callback = None

    logger.info("Beginning GloVe training..")
    W = train_glove(vocab, cooccurrences,
                    iter_callback=iter_callback,
                    vector_size=arguments.vector_size,
                    iterations=arguments.iterations,
                    learning_rate=arguments.learning_rate)

    # TODO shave off bias values, do something with context vectors
    save_model(W, arguments.vector_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s\t%(message)s")
    main(parse_args())
