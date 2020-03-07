import argparse
from input_fn import *
from model.dstm import *
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

AX_SIZE = 14


def tool_trend_analysis(data_source, model_folder):
    # get all journals information
    if data_source == 'bio':
        journals_info_path = [BIO_GENO_BIO_INFO_PATH, BIO_BMCINFOR_INFO_PATH, BIO_PLOS_COMPBIO_INFO_PATH,
                              BIO_GENO_BIO_INFO_PATH, BIO_NUCLEIC_INFO_PATH]
    else:
        journals_info_path = [NEURO_JOCN_INFO_PATH, NEURO_FCN_INFO_PATH, NEURO_JON_INFO_PATH,
                              NEURO_NEURON_INFO_PATH]
    docs_info = {}
    for j in journals_info_path:
        with open(j, 'r') as fp:
            docs_info.update(json.load(fp))

    # get inputs data
    inputs = input_fn('demo', data_source)

    docs = inputs['docs']
    doc_tool_map = inputs['doc_tool_map']
    doc_dataset_map = inputs['doc_dataset_map']
    tools = inputs['tools']
    datasets = inputs['datasets']
    docs_idx = inputs['docs_idx']
    num_vocabs = len(inputs['vocab'])

    # load model file
    folder_name = MODELS_FOLDER + model_folder + '/'
    kw = np.loadtxt(folder_name + 'kw.dat')
    kt = np.loadtxt(folder_name + 'kt.dat')
    ks = np.loadtxt(folder_name + 'ks.dat')
    xtot = np.loadtxt(folder_name + '/xtot.dat')
    ytot = np.loadtxt(folder_name + '/ytot.dat')
    ztot = np.loadtxt(folder_name + '/ztot.dat')
    lbtot = np.loadtxt(folder_name + '/lbtot.dat')

    with open(MODELS_FOLDER + model_folder + '/settings.json') as fp:
        settings = json.load(fp)  # docs with bag of words
        alpha = settings['alpha']
        beta = settings['beta']
        eta0 = settings['eta0']
        eta1 = settings['eta1']

    num_topics = kw.shape[0]
    num_samples = 100

    # initialize year tracking metrics
    tool_trend = {}
    for y in range(2009, 2020):
        tool_trend[y] = np.zeros((num_topics, len(tools)))

    with tqdm(total=len(docs), unit="docs", desc="tools trend analysis") as p_bar:
        for d in xrange(len(docs)):
            # get year of this paper
            year = docs_info[docs_idx[d]]['year']

            # get tool, datasets idx
            tools_idx = doc_tool_map[d]
            datasets_idx = doc_dataset_map[d]

            # ignore those paper without tools
            if len(tools_idx) == 0:
                continue

            topic_assignment = np.zeros(num_topics)
            tool_assignment = np.zeros(len(tools))

            for it in xrange(num_samples):
                for w in docs[d].keys():
                    c = docs[d][w]  # Number of occurrences of word w in document d

                    for i in xrange(c):
                        words_prob = (1.0 * (kw[:, w] + beta)) / (ztot + num_vocabs * beta)

                        tool_topic_probs = []
                        for ii in xrange(len(tools_idx)):
                            tools_probs = ((kt[:, tools_idx[ii]] + alpha) /
                                           (xtot[tools_idx[ii]] + num_topics * alpha))
                            p = (1.0 * (lbtot[d][0] + eta0 - 1) / (lbtot[d][0] + lbtot[d][1] + eta0 + eta1 - 1)) \
                                * tools_probs * words_prob
                            tool_topic_probs += p.tolist()

                        dataset_topic_probs = []
                        for ii in xrange(len(datasets_idx)):
                            datasets_probs = ((ks[:, datasets_idx[ii]] + alpha) /
                                              (ytot[datasets_idx[ii]] + num_topics * alpha))
                            p = (1.0 * (lbtot[d][1] + eta1 - 1) / (lbtot[d][0] + lbtot[d][1] + eta0 + eta1 - 1))\
                                * datasets_probs * words_prob
                            dataset_topic_probs += p.tolist()

                        sample_tool = np.random.binomial(1, sum(tool_topic_probs) /
                                                             (sum(tool_topic_probs) + sum(dataset_topic_probs)),
                                                             1).item(0) == 1
                        if sample_tool:
                            new_topic, new_tool = sample_discrete(tool_topic_probs, len(tools_idx), num_topics)
                            topic_assignment[new_topic] += 1
                            tool_assignment[tools_idx[new_tool]] += 1

            # assignment top two topics for this documents
            top_topics = np.argsort(topic_assignment)[::-1][:2]

            if len(tools_idx) > 2:
                top_tools = np.argsort(tool_assignment)[::-1][:2]
            else:
                top_tools = np.argsort(tool_assignment)[::-1][:1]
            tool_trend[year][top_topics, top_tools] += 1
            p_bar.update(1)

    # save trend model
    folder_name = MODELS_FOLDER + model_folder + '/' + "tool trend/"

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    for y in range(2009, 2020):
        np.savetxt(folder_name + str(y) + '.dat', tool_trend[y])


def dataset_trend_analysis(data_source, model_folder):
    # get all journals information
    if data_source == 'bio':
        journals_info_path = [BIO_GENO_BIO_INFO_PATH, BIO_BMCINFOR_INFO_PATH, BIO_PLOS_COMPBIO_INFO_PATH,
                              BIO_GENO_BIO_INFO_PATH, BIO_NUCLEIC_INFO_PATH]
    else:
        journals_info_path = [NEURO_JOCN_INFO_PATH, NEURO_FCN_INFO_PATH, NEURO_JON_INFO_PATH,
                              NEURO_NEURON_INFO_PATH]
    docs_info = {}
    for j in journals_info_path:
        with open(j, 'r') as fp:
            docs_info.update(json.load(fp))

    # get inputs data
    inputs = input_fn('demo', data_source)

    docs = inputs['docs']
    doc_tool_map = inputs['doc_tool_map']
    doc_dataset_map = inputs['doc_dataset_map']
    datasets = inputs['datasets']
    docs_idx = inputs['docs_idx']
    num_vocabs = len(inputs['vocab'])

    # load model file
    folder_name = MODELS_FOLDER + model_folder + '/'
    kw = np.loadtxt(folder_name + 'kw.dat')
    kt = np.loadtxt(folder_name + 'kt.dat')
    ks = np.loadtxt(folder_name + 'ks.dat')
    xtot = np.loadtxt(folder_name + '/xtot.dat')
    ytot = np.loadtxt(folder_name + '/ytot.dat')
    ztot = np.loadtxt(folder_name + '/ztot.dat')
    lbtot = np.loadtxt(folder_name + '/lbtot.dat')

    with open(MODELS_FOLDER + model_folder + '/settings.json') as fp:
        settings = json.load(fp)  # docs with bag of words
        alpha = settings['alpha']
        beta = settings['beta']
        eta0 = settings['eta0']
        eta1 = settings['eta1']

    num_topics = kw.shape[0]
    num_samples = 100

    # initialize year tracking metrics
    dataset_trend = {}
    for y in range(2009, 2020):
        dataset_trend[y] = np.zeros((num_topics, len(datasets)))

    with tqdm(total=len(docs), unit="docs", desc="dataset trend analysis") as p_bar:
        for d in xrange(len(docs)):
            # get year of this paper
            year = docs_info[docs_idx[d]]['year']

            # get tool, datasets idx
            tools_idx = doc_tool_map[d]
            datasets_idx = doc_dataset_map[d]

            # ignore those paper without tools
            if len(datasets_idx) == 0:
                continue

            topic_assignment = np.zeros(num_topics)
            dataset_assignment = np.zeros(len(datasets))

            for it in xrange(num_samples):
                for w in docs[d].keys():
                    c = docs[d][w]  # Number of occurrences of word w in document d

                    for i in xrange(c):
                        words_prob = (1.0 * (kw[:, w] + beta)) / (ztot + num_vocabs * beta)

                        tool_topic_probs = []
                        for ii in xrange(len(tools_idx)):
                            tools_probs = ((kt[:, tools_idx[ii]] + alpha) /
                                           (xtot[tools_idx[ii]] + num_topics * alpha))
                            p = (1.0 * (lbtot[d][0] + eta0 - 1) / (lbtot[d][0] + lbtot[d][1] + eta0 + eta1 - 1)) \
                                * tools_probs * words_prob
                            tool_topic_probs += p.tolist()

                        dataset_topic_probs = []
                        for ii in xrange(len(datasets_idx)):
                            datasets_probs = ((ks[:, datasets_idx[ii]] + alpha) /
                                              (ytot[datasets_idx[ii]] + num_topics * alpha))
                            p = (1.0 * (lbtot[d][1] + eta1 - 1) / (lbtot[d][0] + lbtot[d][1] + eta0 + eta1 - 1))\
                                * datasets_probs * words_prob
                            dataset_topic_probs += p.tolist()

                        sample_tool = np.random.binomial(1, sum(tool_topic_probs) /
                                                            (sum(tool_topic_probs) + sum(dataset_topic_probs)),
                                                            1).item(0) == 1
                        if not sample_tool:
                            new_topic, new_dataset = sample_discrete(dataset_topic_probs, len(datasets_idx), num_topics)
                            topic_assignment[new_topic] += 1
                            dataset_assignment[datasets_idx[new_dataset]] += 1

            # assignment top two topics for this documents
            top_topics = np.argsort(topic_assignment)[::-1][:2]

            if len(tools_idx) > 2:
                top_datasets = np.argsort(dataset_assignment)[::-1][:2]
            else:
                top_datasets = np.argsort(dataset_assignment)[::-1][:1]
            dataset_trend[year][top_topics, top_datasets] += 1
            p_bar.update(1)

    # save trend model
    folder_name = MODELS_FOLDER + model_folder + '/' + "dataset trend/"

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    for y in range(2009, 2020):
        np.savetxt(folder_name + str(y) + '.dat', dataset_trend[y])


def plot_trend(topic_id, trend_ratio, names, year_start, model_folder):
    markers = itertools.cycle(['+', '*', 'x', 'o', '1', 'p', 'h'])
    colors = itertools.cycle('bgrmykb')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.tick_params(axis='x', labelsize=AX_SIZE)
    ax.tick_params(axis='y', labelsize=AX_SIZE)

    for i in xrange(len(names)):
        if np.max(trend_ratio[:, i]) - np.min(trend_ratio[:, i]) > 0.05:
            y_values = trend_ratio[:, i].tolist()
            x_values = range(year_start, year_start + len(y_values))
            plt.plot(x_values, y_values, marker=markers.next(),
                     color=colors.next(), markersize=10, label=names[i])
            print names[i], '=', y_values

    print "year", x_values
    plt.xticks(range(year_start, year_start + len(y_values), 2), fontsize=AX_SIZE)
    plt.xlabel('Year', size=AX_SIZE)
    plt.ylabel('Ratio', size=AX_SIZE)

    plt.legend(loc='best', fontsize=AX_SIZE, fancybox=False)
    plt.show()

    fig_name = "topic-" + str(topic_id) + ".pdf"
    with PdfPages(model_folder + fig_name) as pdf:
        pdf.savefig(fig)


def tool_trend_demonstration(data_source, model_folder, target_topic=38):
    # get inputs data
    inputs = input_fn('demo', data_source)
    tools = inputs['tools']

    folder_name = MODELS_FOLDER + model_folder + '/' + 'tool trend' + '/'

    tool_trend = {}
    year_end = 2018
    year_start = 2009
    for y in range(year_start, year_end + 1):
        tool_trend[y] = np.loadtxt(folder_name + str(y) + '.dat')

    trend_tool = np.zeros((len(tools), year_end - year_start + 1))
    for y in range(year_start, year_end + 1):
        tools_list = tool_trend[y][target_topic,:]
        trend_tool[:, y - year_start] = tools_list

    weight = 0.4
    for i in range(len(tools)):
        # auto-regress
        for ii in range(1, year_end - year_start + 1):
            # trend_tool[i, ii] = trend_tool[i, ii - 1] * 0.05 + trend_tool[i, ii] * 0.95

            if ii == 1:
                trend_tool[i, ii] = (trend_tool[i, ii - 1]) * weight + trend_tool[i, ii] * (1 - weight)
            elif ii == 2:
                trend_tool[i, ii] = (trend_tool[i, ii - 1] + trend_tool[i, ii - 2]) * weight \
                                    + trend_tool[i, ii] * (1 - weight)
            else:
                trend_tool[i, ii] = (trend_tool[i, ii - 1] + trend_tool[i, ii - 2] + trend_tool[i, ii - 3]) * weight \
                                    + trend_tool[i, ii] * (1 - weight)

    trend_tool = trend_tool / (np.sum(trend_tool, 0) + 0.001)
    trend_tool = trend_tool.T

    plot_trend(target_topic, trend_tool, tools, year_start, folder_name)


def dataset_trend_demonstration(data_source, model_folder, target_topic=18):
    # get inputs data
    inputs = input_fn('demo', data_source)
    datasets = inputs['datasets']

    folder_name = MODELS_FOLDER + model_folder + '/' + 'dataset trend' + '/'

    dataset_trend = {}
    year_end = 2018
    year_start = 2009
    for y in range(year_start, year_end + 1):
        dataset_trend[y] = np.loadtxt(folder_name + str(y) + '.dat')

    trend_dataset = np.zeros((len(datasets), year_end - year_start + 1))
    for y in range(year_start, year_end + 1):
        dataset_list = dataset_trend[y][target_topic, :]
        trend_dataset[:, y - year_start] = dataset_list

    weight = 0.4
    for i in range(len(datasets)):
        # auto-regress
        for ii in range(1, year_end - year_start + 1):
            # trend_tool[i, ii] = trend_tool[i, ii - 1] * 0.05 + trend_tool[i, ii] * 0.95

            if ii == 1:
                trend_dataset[i, ii] = (trend_dataset[i, ii - 1]) * weight + trend_dataset[i, ii] * (1 - weight)
            elif ii == 2:
                trend_dataset[i, ii] = (trend_dataset[i, ii - 1] + trend_dataset[i, ii - 2]) * weight \
                                    + trend_dataset[i, ii] * (1 - weight)
            else:
                trend_dataset[i, ii] = (trend_dataset[i, ii - 1] + trend_dataset[i, ii - 2] + trend_dataset[i, ii - 3]) * weight \
                                    + trend_dataset[i, ii] * (1 - weight)

    trend_dataset = trend_dataset / (np.sum(trend_dataset, 0) + 0.001)
    trend_dataset = trend_dataset.T

    plot_trend(target_topic, trend_dataset, datasets, year_start, folder_name)
