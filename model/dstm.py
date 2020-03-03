import numpy as np
import random
from datetime import datetime
import sys
from time import time
from contants import *
import json
from scipy.special import gammaln

np.set_printoptions(threshold=sys.maxsize)
NUM_CHOICES = 2


def print_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": "


def parameter_estimation(kw, kt, ks, ztot, alpha, beta):
    num_topics = kw.shape[0]
    num_vocabs = kw.shape[1]
    num_tools = kt.shape[1]
    num_datasets = ks.shape[1]

    est_kw = np.zeros((num_topics, num_vocabs))
    est_kt = np.zeros((num_topics, num_tools))
    est_ks = np.zeros((num_topics, num_datasets))

    for t in xrange(num_tools):
        est_kt[:, t] = (1.0 * (kt[:, t] + alpha)) / (np.sum(kt[:, t]) + num_topics * alpha)

    for s in xrange(num_datasets):
        est_ks[:, s] = (1.0 * (ks[:, s] + alpha)) / (np.sum(ks[:, s]) + num_topics * alpha)

    for k in xrange(num_topics):
        est_kw[k, :] = (1.0 * (kw[k, :] + beta)) / (ztot[k] + num_vocabs * beta)
        est_ks[k, :] = est_ks[k, :] / np.sum(est_ks[k, :])
        est_kt[k, :] = est_kt[k, :] / np.sum(est_kt[k, :])

    return est_kw, est_kt, est_ks


def sample_discrete(probs, n, num_topics):
    r = np.sum(probs) * np.random.uniform(0, 1, 1)
    _max = probs[0]

    i = ii = 0
    topic = 0
    if n == 1:
        while r > _max:
            topic += 1
            _max += probs[topic]
    else:
        while r > _max:
            ii += 1
            topic += 1

            if topic == num_topics:
                i += 1
                topic = 0
            _max += probs[ii]

    return topic, i


def safe_minus_one(v):
    """
    Safe minus one operations to avoid negative value

    Parameters
    ----------
    v : number
        input value

    Returns
    -------
    v: number >= 0
    """
    v -= 1
    if v < 0:
        v = 0
    return v


class DSTP:
    def __init__(self, inputs):
        self.docs = inputs['docs']
        self.doc_tool_map = inputs['doc_tool_map']
        self.doc_dataset_map = inputs['doc_dataset_map']
        self.common_tool = inputs['common_tool']
        self.num_tools = inputs['num_tools']
        self.num_datasets = inputs['num_datasets']
        self.num_docs = inputs['num_docs']
        self.num_vocabs = inputs['num_vocabs']
        self.mode = inputs['mode']
        self.run_mode = inputs['run_mode']
        self.alpha = None
        self.beta = None
        self.eta0 = None
        self.eta1 = None
        self.seed = None
        self.verbose = None
        self.model_folder = None
        self.data_source = None
        self.num_iterations = None
        self.num_topics = None
        self.paras = {}

    def model_init(self, inputs):
        print print_timestamp() + "DSTP Model initialization..."
        print print_timestamp() + "Model mode is " + self.mode

        if self.run_mode == 'start' and self.mode != 'inf':
            self.alpha = inputs['alpha']
            self.beta = inputs['beta']
            self.eta0 = inputs['eta0']
            self.eta1 = inputs['eta1']
            self.num_topics = inputs['num_topics']
            self.seed = inputs['seed']
            self.verbose = inputs['verbose']
            self.model_folder = inputs['model_folder']
            self.data_source = inputs['data_source']
            self.num_iterations = inputs['num_iterations']

            kw = np.zeros((self.num_topics, self.num_vocabs))  # K x V (num_topics x num_vocabs) matrix
            kt = np.zeros((self.num_topics, self.num_tools))  # K x T (num_topics x num_tools) matrix
            ks = np.zeros((self.num_topics, self.num_datasets))  # K x S (num_topics x num_datasets) matrix
            z = [None] * self.num_docs  # keep topic assignment for all tokens in docs
            x = [None] * self.num_docs  # keep tool assignment for all tokens in docs
            y = [None] * self.num_docs  # keep dataset assignment for all tokens in docs
            lb = [None] * self.num_docs  # keep label assignments for all tokens in docs
            ztot = np.zeros(self.num_topics)  # total number of assignment for each topic k
            xtot = np.zeros(self.num_tools)  # total number of assignment for each tool t
            ytot = np.zeros(self.num_datasets)  # total number of assignment for each dataset s
            lbtot = np.zeros((self.num_docs, NUM_CHOICES))  # total number of labels (0: t, 1: s) in each document

            log_probs = []  # save the record of perplexity
            print print_timestamp() + "Start to initialize randomly..."

            # initialize seed
            np.random.seed(self.seed)
            for d in xrange(self.num_docs):
                zd = {}  # topic assignment for each word in document d
                xd = {}  # tool assignment for each word in document d
                yd = {}  # dataset assignment for each word in ducment d
                lbd = {}  # label assignment

                for w in self.docs[d].keys():
                    c = self.docs[d][w]  # Number of occurrences of word w in document d
                    lbd[w] = []

                    # a) Randomly assign a topic for each token
                    rand_topics = np.random.randint(self.num_topics,
                                                    size=c)  # Randomly assignment a topic for each occurrence
                    zd[w] = rand_topics.tolist()

                    # update matrix based on new topic assignments
                    ztot[rand_topics] += 1
                    kw[rand_topics, w] += 1

                    # b) Randomly assign a tool and dataset for each token
                    xd[w] = [None] * c
                    yd[w] = [None] * c
                    for i in xrange(c):

                        # Randomly assign a tool and dataset for each token
                        rand_label = np.random.randint(NUM_CHOICES, size=1)[0]
                        if rand_label == 0:  # Sample from tool
                            rand_tool = np.random.randint(self.num_tools, size=1)[0]

                            # Update matrix
                            xd[w][i] = rand_tool
                            xtot[rand_tool] += 1
                            kt[rand_topics[i], rand_tool] += 1
                            lbtot[d, 0] += 1
                            lbd[w].append(0)
                        else:
                            rand_dataset = np.random.randint(self.num_datasets, size=1)[0]

                            # Update matrix
                            yd[w][i] = rand_dataset
                            ytot[rand_dataset] += 1
                            ks[rand_topics[i], rand_dataset] += 1
                            lbtot[d, 1] += 1
                            lbd[w].append(1)
                z[d] = zd
                x[d] = xd
                y[d] = yd
                lb[d] = lbd

            self.paras = {'kw': kw, 'kt': kt, 'ks': ks, 'z': z,
                          'x': x, 'y': y, 'lb': lb, 'ztot': ztot,
                          'xtot': xtot, 'ytot': ytot, 'lbtot': lbtot}
            if self.verbose == 1:
                lp = self.log_likelihood()
                log_probs.append(lp)
                print print_timestamp() + "The log likelihood of words %f" % lp
            self.paras['log_likelihood'] = log_probs
            print print_timestamp() + "Finish the initialization"
        else:
            self.num_iterations = inputs['num_iterations']
            self.model_folder = inputs['model_folder']

            print print_timestamp() + "Start to load model file: " + str(self.model_folder)
            kw = np.loadtxt(MODELS_FOLDER + self.model_folder + '/kw.dat')
            kt = np.loadtxt(MODELS_FOLDER + self.model_folder + '/kt.dat')
            ks = np.loadtxt(MODELS_FOLDER + self.model_folder + '/ks.dat')
            xtot = np.loadtxt(MODELS_FOLDER + self.model_folder + '/xtot.dat')
            ytot = np.loadtxt(MODELS_FOLDER + self.model_folder + '/ytot.dat')
            ztot = np.loadtxt(MODELS_FOLDER + self.model_folder + '/ztot.dat')
            lbtot = np.loadtxt(MODELS_FOLDER + self.model_folder + '/lbtot.dat')
            log_probs = np.loadtxt(MODELS_FOLDER + self.model_folder + '/log_likelihood.dat').tolist()
            if type(log_probs) == float:
                log_probs = [log_probs]

            with open(MODELS_FOLDER + self.model_folder + '/x.json') as fp:
                x = json.load(fp)  # docs with bag of words

            with open(MODELS_FOLDER + self.model_folder + '/y.json') as fp:
                y = json.load(fp)  # docs with bag of words

            with open(MODELS_FOLDER + self.model_folder + '/z.json') as fp:
                z = json.load(fp)  # docs with bag of words

            with open(MODELS_FOLDER + self.model_folder + '/lb.json') as fp:
                lb = json.load(fp)  # docs with bag of words

            # change key from str to integer
            for d in xrange(len(x)):
                for w in x[d].keys():
                    x[d][int(w)] = x[d].pop(w)

            for d in xrange(len(y)):
                for w in y[d].keys():
                    y[d][int(w)] = y[d].pop(w)

            for d in xrange(len(z)):
                for w in z[d].keys():
                    z[d][int(w)] = z[d].pop(w)

            for d in xrange(len(lb)):
                for w in lb[d].keys():
                    lb[d][int(w)] = lb[d].pop(w)

            self.paras = {'kw': kw, 'kt': kt, 'ks': ks, 'z': z,
                          'x': x, 'y': y, 'lb': lb, 'ztot': ztot,
                          'xtot': xtot, 'ytot': ytot, 'lbtot': lbtot, 'log_likelihood': log_probs}

            # retrieve model setting files:
            with open(MODELS_FOLDER + self.model_folder + '/settings.json') as fp:
                settings = json.load(fp)  # docs with bag of words
                self.alpha = settings['alpha']
                self.beta = settings['beta']
                self.eta0 = settings['eta0']
                self.eta1 = settings['eta1']
                self.num_topics = settings['num_topics']
                self.seed = settings['seed']
                self.verbose = settings['verbose']
                self.data_source = settings['data_source']

            # initialize seed
            np.random.seed(self.seed)
            print print_timestamp() + "Finish model loading"
        sys.stdout.flush()

    def get_tool_ids(self, ori_tools_idx):
        if set(self.common_tool) & set(ori_tools_idx) == set(ori_tools_idx):
            return ori_tools_idx
        else:
            return list(set(ori_tools_idx) - set(self.common_tool) & set(ori_tools_idx))

    def log_likelihood(self):
        # lp = 0  # log probability
        #
        # kw = self.paras['kw']
        # for d in xrange(len(self.docs)):
        #     for w in self.docs[d].keys():
        #         c = self.docs[d][w]
        #         lp += 1.0 * c * np.log(np.sum(kw[:, w]))
        # return lp

        print print_timestamp() + "Start to compute log likelihood..."

        # retrieve parameters
        kw = self.paras['kw']
        kt = self.paras['kt']
        ks = self.paras['ks']
        ztot = self.paras['ztot']
        lbtot = self.paras['lbtot']

        est_kw, est_kt, est_ks = parameter_estimation(kw, kt, ks, ztot, self.alpha, self.beta)
        log_probs = 0

        for d in xrange(self.num_docs):
            # get tool, datasets idx
            tools_idx = self.doc_tool_map[d]
            datasets_idx = self.doc_dataset_map[d]

            prob_tool_select = prob_dataset_select = 0

            if len(tools_idx) != 0:
                prob_tool_select = 1.0 / len(tools_idx)

            if len(datasets_idx) != 0:
                prob_dataset_select = 1.0 / len(datasets_idx)

            # compute the prob. of choose tool from posterior distribution
            prob_label = (1.0 * (lbtot[d][0] + self.eta0) / (lbtot[d][0] + lbtot[d][1] + self.eta0 + self.eta1))

            for w in self.docs[d].keys():
                c = self.docs[d][w]  # Number of occurrences of word w in document d

                sum_p = 0
                for k in xrange(self.num_topics):
                    p = prob_tool_select * prob_label * np.sum(np.multiply(est_kw[k, w], est_kt[k, tools_idx])) + \
                        prob_dataset_select * (1 - prob_label) * np.sum(
                        np.multiply(est_kw[k, w], est_ks[k, datasets_idx]))

                    sum_p += p

                log_probs += 1.0 * c * np.log(sum_p)

        print print_timestamp() + "The likelihood is " + str(log_probs)

        return log_probs

    def gibbs(self):
        print print_timestamp() + "Start to run Gibbs sampling algorithm iteratively..."

        # retrieve parameters
        kw = self.paras['kw']
        kt = self.paras['kt']
        ks = self.paras['ks']
        z = self.paras['z']
        x = self.paras['x']
        y = self.paras['y']
        lb = self.paras['lb']
        ztot = self.paras['ztot']
        xtot = self.paras['xtot']
        ytot = self.paras['ytot']
        lbtot = self.paras['lbtot']
        log_probs = self.paras['log_likelihood']

        for it in xrange(self.num_iterations):
            sys.stdout.flush()
            ts_start_time = time()

            # shuffle sequence of documents for random updating
            docs_idx = list(range(self.num_docs))
            random.shuffle(docs_idx)

            for d in docs_idx:

                # get tool, datasets idx
                tools_idx = self.doc_tool_map[d]
                # # if all tool is common tool, randomly drop it
                # if set(tools_idx) & set(self.common_tool) == set(tools_idx):
                #     if np.random.uniform(0, 1, 1) > 0.7:
                #         continue
                datasets_idx = self.doc_dataset_map[d]

                for w in self.docs[d].keys():
                    c = self.docs[d][w]  # Number of occurrences of word w in document d

                    new_zs = []
                    new_xs = []
                    new_ys = []
                    new_lbs = []

                    for i in xrange(c):
                        cur_k = z[d][w][i]
                        cur_t = x[d][w][i]
                        cur_s = y[d][w][i]
                        cur_lb = lb[d][w][i]

                        # a) Remove current assignment
                        ztot[cur_k] = safe_minus_one(ztot[cur_k])
                        kw[cur_k, w] = safe_minus_one(kw[cur_k, w])

                        if cur_lb == 0:
                            xtot[cur_t] = safe_minus_one(xtot[cur_t])
                            kt[cur_k, cur_t] = safe_minus_one(kt[cur_k, cur_t])
                        else:
                            ytot[cur_s] = safe_minus_one(ytot[cur_s])
                            ks[cur_k, cur_s] = safe_minus_one(ks[cur_k, cur_s])

                        # b) Sample label, tool assignment/dataset assignment, topic assignment jointly
                        # use vector operation to improve performance
                        words_prob = (1.0 * (kw[:, w] + self.beta)) / (ztot + self.num_vocabs * self.beta)

                        tool_topic_probs = []
                        for ii in xrange(len(tools_idx)):
                            tools_probs = ((kt[:, tools_idx[ii]] + self.alpha) /
                                           (xtot[tools_idx[ii]] + self.num_topics * self.alpha))
                            p = (1.0 * (lbtot[d][0] + self.eta0 - 1) / (lbtot[d][0] + lbtot[d][1] + self.eta0 +
                                                                        self.eta1 - 1)) * tools_probs * words_prob
                            tool_topic_probs += p.tolist()

                        dataset_topic_probs = []
                        for ii in xrange(len(datasets_idx)):
                            datasets_probs = ((ks[:, datasets_idx[ii]] + self.alpha) /
                                              (ytot[datasets_idx[ii]] + self.num_topics * self.alpha))
                            p = (1.0 * (lbtot[d][1] + self.eta1 - 1) / (lbtot[d][0] + lbtot[d][1] + self.eta0 +
                                                                        self.eta1 - 1)) * datasets_probs * words_prob
                            dataset_topic_probs += p.tolist()

                        # c) decide current word generated by tool or dataset
                        sample_tool = np.random.binomial(1, sum(tool_topic_probs) /
                                                         (sum(tool_topic_probs) + sum(dataset_topic_probs)),
                                                         1).item(0) == 1
                        if sample_tool:
                            new_topic, new_tool = sample_discrete(tool_topic_probs, len(tools_idx), self.num_topics)

                            # c) update matrix
                            new_zs.append(new_topic)
                            new_xs.append(tools_idx[new_tool])
                            new_ys.append(None)
                            new_lbs.append(0)

                            ztot[new_topic] += 1
                            kw[new_topic, w] += 1
                            kt[new_topic, tools_idx[new_tool]] += 1
                            xtot[tools_idx[new_tool]] += 1

                            if lbtot[d, 1] >= 1:
                                lbtot[d, 0] += 1
                                lbtot[d, 1] -= 1
                        else:
                            new_topic, new_dataset = sample_discrete(dataset_topic_probs, len(datasets_idx),
                                                                     self.num_topics)

                            # c) update matrix
                            new_zs.append(new_topic)
                            new_xs.append(None)
                            new_ys.append(datasets_idx[new_dataset])
                            new_lbs.append(1)

                            ztot[new_topic] += 1
                            kw[new_topic, w] += 1
                            ks[new_topic, datasets_idx[new_dataset]] += 1
                            ytot[datasets_idx[new_dataset]] += 1

                            if lbtot[d, 0] >= 1:
                                lbtot[d, 0] -= 1
                                lbtot[d, 1] += 1

                    z[d][w] = new_zs
                    x[d][w] = new_xs
                    y[d][w] = new_ys
                    lb[d][w] = new_lbs

            ts_end_time = time()

            if (self.verbose == 1 and (it + 1) % 5 == 0) or (it == self.num_iterations - 1):
                lp = self.harmonic_mean()
                log_probs.append(lp)
                print print_timestamp() + "After %d-th iteration, The log likelihood of words %f" % (it + 1, lp)
            print print_timestamp() + "Finish %d-th iteration. The iteration time is %f. " % (it + 1, ts_end_time -
                                                                                              ts_start_time)

        # update parameters
        self.paras = {'kw': kw, 'kt': kt, 'ks': ks, 'z': z,
                      'x': x, 'y': y, 'lb': lb, 'ztot': ztot,
                      'xtot': xtot, 'ytot': ytot, 'lbtot': lbtot, 'log_likelihood': log_probs}
        sys.stdout.flush()

    def harmonic_mean(self):
        # retrieve parameters
        kw = self.paras['kw']

        log_likelihood = 0
        num_topics = kw.shape[0]
        num_vocabs = kw.shape[1]
        ztot = np.sum(kw, axis=1)

        constant = gammaln(num_vocabs * self.beta) - num_vocabs * gammaln(self.beta)

        for k in xrange(num_topics):
            log_likelihood += constant
            log_likelihood += np.sum(gammaln(kw[k, :] + self.beta))
            log_likelihood -= gammaln(ztot[k] + num_vocabs * self.beta)

        print print_timestamp() + "the harmonic mean of words is %f" % log_likelihood
        return log_likelihood

    def inference(self, inputs):
        print print_timestamp() + "Start model inference for testing docs... "
        print print_timestamp() + "The number of test documents: " + str(self.num_docs)

        # retrieve parameters
        kw = self.paras['kw']
        kt = self.paras['kt']
        ks = self.paras['ks']
        z = self.paras['z']
        x = self.paras['x']
        y = self.paras['y']
        lb = self.paras['lb']
        ztot = self.paras['ztot']
        xtot = self.paras['xtot']
        ytot = self.paras['ytot']
        lbtot = self.paras['lbtot']

        trained_tool = inputs['trained_tool']
        trained_dataset = inputs['trained_dataset']

        # compute the perplexity for total words in the test docs
        total_perplexity = 0  # total perplexity
        num_sample = 10

        est_kw, est_kt, est_ks = parameter_estimation(kw, kt, ks, ztot, self.alpha, self.beta)

        num_docs = 0
        for d in xrange(self.num_docs):
            # randomly assignment label for document d
            lb_d = [0, 0]

            # get tool, datasets idx
            tools_idx = self.doc_tool_map[d]
            datasets_idx = self.doc_dataset_map[d]

            if len(set(trained_tool) & set(tools_idx)) == 0 and len(set(trained_dataset) & set(datasets_idx)) == 0:
                continue

            num_docs += 1
            perplexity = 0
            num_words = sum(self.docs[d].values())  # number of words in doc d

            # run some iterations of Gibbs sampling for test document d
            for it in xrange(num_sample):
                lp = 0  # log likelihood

                for w in self.docs[d].keys():
                    c = self.docs[d][w]  # Number of occurrences of word w in document d
                    words_prob = (1.0 * (kw[:, w] + self.beta)) / (ztot + self.num_vocabs * self.beta)

                    for i in xrange(c):

                        # a) Sample labels to decide choosing tool or dataset
                        # if sample from tool
                        tool_topic_probs = []
                        for ii in xrange(len(tools_idx)):
                            tools_probs = ((kt[:, tools_idx[ii]] + self.alpha) /
                                           (xtot[tools_idx[ii]] + self.num_topics * self.alpha))
                            p = (1.0 * (lbtot[d][0] + self.eta0 - 1) / (lbtot[d][0] + lbtot[d][1] + self.eta0 +
                                                                        self.eta1 - 1)) * tools_probs * words_prob
                            tool_topic_probs += p.tolist()

                        dataset_topic_probs = []
                        for ii in xrange(len(datasets_idx)):
                            datasets_probs = ((ks[:, datasets_idx[ii]] + self.alpha) /
                                              (ytot[datasets_idx[ii]] + self.num_topics * self.alpha))
                            p = (1.0 * (lbtot[d][1] + self.eta1 - 1) / (lbtot[d][0] + lbtot[d][1] + self.eta0 +
                                                                        self.eta1 - 1)) * datasets_probs * words_prob
                            dataset_topic_probs += p.tolist()

                        # b) decide current word generated by tool or dataset
                        sample_tool = np.random.binomial(1, sum(tool_topic_probs) /
                                                         (sum(tool_topic_probs) + sum(dataset_topic_probs)),
                                                         1).item(0) == 1

                        # c) update labels
                        if sample_tool:
                            if lb_d[0] >= 1:
                                lb_d[0] -= 1
                                lb_d[1] += 1
                        else:
                            if lb_d[1] >= 1:
                                lb_d[0] += 1
                                lb_d[1] -= 1

                        # d) compute the prob. of choose tool from posterior distribution
                        pl = (1.0 * (lb_d[0] + self.eta0) / (lb_d[0] + lb_d[1] + self.eta0 + self.eta1))

                        sum_p = 0
                        for k in xrange(self.num_topics):
                            p = pl * np.sum(np.multiply(est_kw[k, w], est_kt[k, tools_idx])) + \
                                (1 - pl) * np.sum(np.multiply(est_kw[k, w], est_ks[k, datasets_idx]))
                            sum_p += p

                        lp += 1.0 * np.log(sum_p)

                perplexity += np.exp(-lp / num_words)

            perplexity = perplexity / num_sample         # compute average perplexity for document d
            total_perplexity += perplexity

            print print_timestamp() + "the doc %d, the perplexity is %f" % (d, perplexity)
            sys.stdout.flush()

        average_perplexity = total_perplexity / num_docs
        print print_timestamp() + "Total number of test docs is % d" % num_docs
        print print_timestamp() + "Finish inference processing, the final perplexity is %f" % average_perplexity

    def save(self):
        print print_timestamp() + "Save model files..."
        if self.run_mode == 'start':
            # create a new folder
            ts = datetime.now().strftime('%m%d%H%M')
            self.model_folder = "DSTP" + ts
            folder_path = MODELS_FOLDER + self.model_folder + '/'
            os.mkdir(folder_path)

            # save setting file
            settings = {'data_source': self.data_source, 'num_topics': self.num_topics,
                        'num_iterations': self.num_iterations, 'seed': self.seed, 'alpha': self.alpha,
                        'beta': self.beta, 'eta0': self.eta0, 'eta1': self.eta1, 'verbose': self.verbose}

        else:
            folder_path = MODELS_FOLDER + self.model_folder + '/'

            # load previous setting file
            with open(MODELS_FOLDER + self.model_folder + '/settings.json') as fp:
                settings = json.load(fp)

            # update number of iterations
            settings['num_iterations'] += self.num_iterations

        # retrieve parameters
        kw = self.paras['kw']
        kt = self.paras['kt']
        ks = self.paras['ks']
        z = self.paras['z']
        x = self.paras['x']
        y = self.paras['y']
        lb = self.paras['lb']
        ztot = self.paras['ztot']
        xtot = self.paras['xtot']
        ytot = self.paras['ytot']
        lbtot = self.paras['lbtot']
        log_probs = self.paras['log_likelihood']
        if len(log_probs) == 0:
            log_probs.append(0)

        # save model file
        np.savetxt(folder_path + 'kw.dat', kw)
        np.savetxt(folder_path + 'kt.dat', kt)
        np.savetxt(folder_path + 'ks.dat', ks)
        np.savetxt(folder_path + 'xtot.dat', xtot)
        np.savetxt(folder_path + 'ytot.dat', ytot)
        np.savetxt(folder_path + 'ztot.dat', ztot)
        np.savetxt(folder_path + 'lbtot.dat', lbtot)
        np.savetxt(folder_path + 'log_likelihood.dat', log_probs)

        with open(folder_path + 'x.json', 'w') as fp:
            json.dump(x, fp)

        with open(folder_path + 'y.json', 'w') as fp:
            json.dump(y, fp)

        with open(folder_path + 'z.json', 'w') as fp:
            json.dump(z, fp)

        with open(folder_path + 'lb.json', 'w') as fp:
            json.dump(lb, fp)

        # save settings file
        with open(folder_path + 'settings.json', 'w') as fp:
            json.dump(settings, fp)

        print print_timestamp() + "Finishing model saving (%s)." % self.model_folder
