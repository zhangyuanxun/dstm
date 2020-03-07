import argparse
from trend_utilis import *


np.set_printoptions(threshold=sys.maxsize)


def parse_args():
    parser = argparse.ArgumentParser(description='Description the Command Line of DSTP Model')
    parser.add_argument('--data_source', help='Select data source: neuroscience(neuro), bioinformatics(bio), '
                                              '(default = %(default)s)', default='bio', choices=['bio', 'neuro'])
    parser.add_argument('--type', default='table', choices=['table', 'trend', 'trend_analysis'],
                        help="Provide type of visualization, such as table")
    parser.add_argument('--topk', help='Number of top K terms for visualization (default = %(default)s)', type=int,
                        default=10, choices=range(1, 101), metavar='(1, ..., 101)')
    parser.add_argument('--topic_id', help='Provide the topic id of when you use the trend type.', type=int)
    parser.add_argument('--model_folder', help='Specify the model folder name.', type=str)
    parser.add_argument('--trend_type', help='Select trend type: tool or dataset, ', choices=['tool', 'dataset'])

    args = parser.parse_args()
    if args.model_folder is None:
        parser.error('Please provide model folder')

    if args.type == 'trend' or args.type == 'trend_analysis':
        if args.trend_type != 'tool' and args.trend_type != 'dataset':
            parser.error('Please provide trend type')

        if args.type == 'trend' and args.topic_id is None:
            parser.error('Please provide topic id')

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


def dstm_topics_tables(args):
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


def dstm_topics_trend_analysis(args):
    data_source = args.data_source
    model_folder = args.model_folder
    trend_type = args.trend_type

    if trend_type == 'tool':
        tool_trend_analysis(data_source, model_folder)
    elif trend_type == 'dataset':
        dataset_trend_analysis(data_source, model_folder)


def dstm_topics_trend(args):
    data_source = args.data_source
    model_folder = args.model_folder
    trend_type = args.trend_type
    topic_id = args.topic_id

    if trend_type == 'tool':
        tool_trend_demonstration(data_source, model_folder, topic_id)
    elif trend_type == 'dataset':
        dataset_trend_demonstration(data_source, model_folder, topic_id)


if __name__ == "__main__":
    args = parse_args()

    if args.type == 'table':
        dstm_topics_tables(args)

    if args.type == 'trend_analysis':
        dstm_topics_trend_analysis(args)

    if args.type == 'trend':
        dstm_topics_trend(args)

