import argparse
from input_fn import *
import numpy as np
import sys

NUM_PRINT_TERMS = 10
np.set_printoptions(threshold=sys.maxsize)


def parse_args():
    parser = argparse.ArgumentParser(description='Description the Command Line of DSTP Model')
    parser.add_argument('--data_source', help='Select data source: neuroscience(neuro), bioinformatics(bio), '
                                              '(default = %(default)s)', default='bio', choices=['bio', 'neuro'])
    parser.add_argument('--type', default='table', choices=['table', 'trend', 'trend-analysis'],
                        help="Provide type of visualization, such as table")
    parser.add_argument('--topk', help='Number of top K terms for visualization (default = %(default)s)', type=int,
                        default=10, choices=range(1, 101), metavar='(1, ..., 101)')
    parser.add_argument('--model_folder', help='Specify the model folder name.', type=str)


    args = parser.parse_args()
    if args.model_folder is None:
        parser.error('Please provide model folder')

    return args


def dstp_parameter_estimation(kw, kt, ks, ztot, alpha, beta):
    num_topics = kw.shape[0]
    num_vocabs = kw.shape[1]
    num_tools = kt.shape[1]
    num_datasets = ks.shape[1]

    est_kw = np.zeros((num_topics, num_vocabs))
    est_kt = np.zeros((num_topics, num_tools))
    est_ks = np.zeros((num_topics, num_datasets))

    for k in xrange(num_topics):
        est_kw[k, :] = (1.0 * (kw[k, :] + beta)) / (ztot[k] + num_vocabs * beta)
        est_kt[k, :] = (1.0 * (kt[k, :] + alpha)) / (np.sum(kt[k, :]) + num_tools * alpha)
        est_ks[k, :] = (1.0 * (ks[k, :] + alpha)) / (np.sum(ks[k, :]) + num_datasets * alpha)

    return est_kw, est_kt, est_ks


def dstp_topics_printer(args):
    # get inputs data
    inputs = input_fn('demo', args.data_source)
    vocabs = inputs['vocab']
    tools = inputs['tools']
    datasets = inputs['datasets']
    topk = args.topk

    # load model file
    folder_name = MODELS_FOLDER + args.model_folder + '/'
    kw = np.loadtxt(folder_name + 'kw.dat')
    kt = np.loadtxt(folder_name + 'kt.dat')
    ks = np.loadtxt(folder_name + 'ks.dat')
    ztot = np.loadtxt(folder_name + 'ztot.dat')

    with open(MODELS_FOLDER + args.model_folder + '/settings.json') as fp:
        settings = json.load(fp)  # docs with bag of words
        alpha = settings['alpha']
        beta = settings['beta']

    est_kw, est_kt, est_ks = dstp_parameter_estimation(kw, kt, ks, ztot, alpha, beta)

    num_topics = kw.shape[0]

    for k in xrange(num_topics):
        kw_idx = np.argsort(est_kw[k, :])[::-1][:topk]

        kt_idx = np.argsort(est_kt[k, :])[::-1]
        kt_idx = kt_idx[kt_idx < len(tools)][:topk]

        ks_idx = np.argsort(est_ks[k, :])[::-1]
        ks_idx = ks_idx[ks_idx < len(datasets)][:topk]

        print 'topic %d:' % k
        for x in xrange(topk):
            print '%20s  \t---\t  %.4f' % (vocabs[kw_idx[x]], est_kw[k, kw_idx[x]]) + '%20s  \t---\t  %.4f' % (
            tools[kt_idx[x]], est_kt[k, kt_idx[x]]) + '%20s  \t---\t  %.4f' % (
                  datasets[ks_idx[x]], est_ks[k, ks_idx[x]])
        print
        print


if __name__ == "__main__":
    args = parse_args()

    if args.type == 'table':
        dstp_topics_printer(args)

