import logging
import os
import sys

import numpy as np

from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from scipy.io import mmread, mmwrite, loadmat, savemat
from scipy.sparse import csc_matrix, spdiags
from sklearn import random_projection

def randne_projection(A, q=3, dim=128):
    transformer = random_projection.GaussianRandomProjection(n_components=dim, random_state=42)
    # Random projection for A
    cur_U = transformer.fit_transform(A)
    U_list = [cur_U]

    for i in range(2, q + 1):
        cur_U = A @ cur_U
        U_list.append(cur_U)
    return U_list

def randne_merge(U_list, weights):
    U = np.zeros_like(U_list[0])
    for cur_U, weight in zip(U_list, weights):
        U += cur_U * weight
    return U

def main():
    parser = ArgumentParser('randne',
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input', nargs='?', required=True,
                        help='Input graph file')
    parser.add_argument('--matfile-variable-name', default='network',
                        help='Variable name of adjacency matrix inside a .mat file')
    parser.add_argument('--output', required=True,
                        help='Output representation file')
    parser.add_argument('--use-trans-matrix', default=False, action='store_true',
                        help='''The input matrix for RandNE. Adjacency matrix is used by default;
                        set this flag to use the transition matrix instead.''')
    parser.add_argument('-q', '--order', default=3, type=int,
                        help='Maximum order of adjacency matrix.')
    parser.add_argument('-d', '--representation-size', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--weights', nargs='+', required=True,
                        help='Weights for each power of the adjacency matrix (or transition matrix).')
    args = parser.parse_args()

    # Process args
    mat_obj = loadmat(args.input)
    A = mat_obj[args.matfile_variable_name]
    if args.use_trans_matrix:
        N = A.shape[0]
        normalizer = spdiags(np.squeeze(1.0 / csc_matrix.sum(A, axis=1) ), 0, N, N)
        input_mat = normalizer @ A
    else:
        input_mat = A
    weights = list(map(float, args.weights))

    # Start RandNE
    U_list = randne_projection(input_mat, q=args.order, dim=args.representation_size)
    U = randne_merge(U_list, weights)

    savemat(args.output, {'emb': U})

if __name__ == '__main__':
    sys.exit(main())
