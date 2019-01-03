"""scoring.py: Script that demonstrates the multi-label classification used."""

import numpy
import sys
import os

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict
from gensim.models import Word2Vec, KeyedVectors
from six import iteritems
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer

class TopKRanker(OneVsRestClassifier):
        def predict(self, X, top_k_list):
                assert X.shape[0] == len(top_k_list)
                probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
                all_labels = []
                for i, k in enumerate(top_k_list):
                        probs_ = probs[i, :]
                        labels = self.classes_[probs_.argsort()[-k:]].tolist()
                        all_labels.append(labels)
                return all_labels

def sparse2graph(x):
        G = defaultdict(lambda: set())
        cx = x.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
                G[i].add(j)
        return {str(k): [str(x) for x in v] for k,v in iteritems(G)}

def main(predefined_args=None):
    parser = ArgumentParser("scoring",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--emb", required=True, help='Embeddings file, could be in word2vec or Matlab format.')
    parser.add_argument("--network", required=True,
                        help='A .mat file containing the adjacency matrix and node labels of the input network.')
    parser.add_argument("--adj-matrix-name", default='network',
                        help='Variable name of the adjacency matrix inside the .mat file.')
    parser.add_argument("--label-matrix-name", default='group',
                        help='Variable name of the labels matrix inside the .mat file.')
    parser.add_argument("--num-shuffles", default=2, type=int, help='Number of shuffles.')
    parser.add_argument("--all", default=False, action='store_true',
                        help='''The embeddings are evaluated on all training percents
                        from 10 to 90 when this flag is set to true.
                        By default, only training percents of 10, 20 and 50 are used.''')
    if predefined_args:
        args = parser.parse_args(predefined_args)
    else:
        args = parser.parse_args()
    # 0. Files
    embeddings_file = args.emb
    matfile = args.network

    # 1. Load labels
    mat = loadmat(matfile)
    labels_matrix = mat[args.label_matrix_name]
    labels_count = labels_matrix.shape[1]
    mlb = MultiLabelBinarizer(range(labels_count))

    # 2. Load Embeddings
    _, file_extension = os.path.splitext(embeddings_file)
    # assume that the only key in the .mat file is the embeddings matrix
    if file_extension == '.mat':
        emb_mat = loadmat(embeddings_file)
        feature_key = [y for y in emb_mat.keys() if y not in ('__header__', '__version__', '__globals__')]
        assert len(feature_key) == 1
        features_matrix = emb_mat[feature_key[0]]
    else:
        model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
        # Map nodes to their features (note: assumes nodes are labeled as integers 0:N-1)
        N = len(model.vocab)
        features_matrix = numpy.asarray([model[str(node)] for node in range(N)])

    # 2. Shuffle, to create train/test groups
    shuffles = []
    for x in range(args.num_shuffles):
        shuffles.append(skshuffle(features_matrix, labels_matrix))

    # 3. to score each train/test group
    all_results = defaultdict(list)

    if args.all:
        training_percents = numpy.asarray(range(1, 10)) * .1
    else:
        training_percents = [0.1, 0.2, 0.5]
    for train_percent in training_percents:
        for shuf in shuffles:

            X, y = shuf

            training_size = int(train_percent * X.shape[0])

            X_train = X[:training_size, :]
            y_train_ = y[:training_size]

            y_train = [[] for x in range(y_train_.shape[0])]

            cy = y_train_.tocoo()
            for i, j in zip(cy.row, cy.col):
                    y_train[i].append(j)

            assert sum(len(l) for l in y_train) == y_train_.nnz

            X_test = X[training_size:, :]
            y_test_ = y[training_size:]

            y_test = [[] for _ in range(y_test_.shape[0])]

            cy = y_test_.tocoo()
            for i, j in zip(cy.row, cy.col):
                    y_test[i].append(j)

            clf = TopKRanker(LogisticRegression())
            clf.fit(X_train, y_train_)

            # find out how many labels should be predicted
            top_k_list = [len(l) for l in y_test]
            preds = clf.predict(X_test, top_k_list)

            results = {}
            averages = ["micro", "macro"]
            for average in averages:
                    results[average] = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average=average)

            all_results[train_percent].append(results)

    print ('Results, using embeddings of dimensionality', X.shape[1])
    print ('-------------------')
    for train_percent in sorted(all_results.keys()):
        print ('Train percent:', train_percent)
        for index, result in enumerate(all_results[train_percent]):
            print ('Shuffle #%d:     ' % (index + 1), result)
        avg_score = defaultdict(float)
        for score_dict in all_results[train_percent]:
            for metric, score in iteritems(score_dict):
                avg_score[metric] += score
        for metric in avg_score:
            avg_score[metric] /= len(all_results[train_percent])
        print ('Average score:', dict(avg_score))
        print ('-------------------')

if __name__ == "__main__":
    sys.exit(main())
