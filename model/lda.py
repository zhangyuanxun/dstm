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


def parameter_estimation(kw, kd, alpha, beta):
    num_topics = kw.shape[0]
    num_vocabs = kw.shape[1]
    num_docs = kd.shape[1]

    est_kw = np.zeros((num_topics, num_vocabs))
    est_kd = np.zeros((num_topics, num_docs))
    ztot = np.sum(kw, axis=1)

    for k in xrange(num_topics):
        est_kw[k, :] = (1.0 * (kw[k, :] + beta)) / (ztot[k] + num_vocabs * beta)

    for d in xrange(num_docs):
        for k in xrange(num_topics):
            est_kd[k, d] = (1.0 * (kd[k, d] + alpha)) / (np.sum(kd[:, d]) + num_topics * alpha)

    return est_kw, est_kd


class LDA:
    def __init__(self, inputs):
        self.docs = inputs['docs']
        self.num_docs = inputs['num_docs']
        self.num_vocabs = inputs['num_vocabs']
        self.mode = inputs['mode']
        self.run_mode = inputs['run_mode']
        self.alpha = None
        self.beta = None
        self.seed = None
        self.verbose = None
        self.model_folder = None
        self.data_source = None
        self.num_iterations = None
        self.num_topics = None
        self.paras = {}

    def log_likelihood(self):
        lp = 0  # log probability
        n = 0  # total number of words

        kw = self.paras['kw']
        kd = self.paras['kd']

        est_kw, est_kd = parameter_estimation(kw, kd, self.alpha, self.beta)
        lp = np.sum(np.log(est_kw))
        return lp

    def model_init(self, inputs):
        print print_timestamp() + "LDA Model initialization..."
        print print_timestamp() + "Model mode is " + self.mode

        if self.run_mode == 'start' and self.mode != 'inf':
            self.alpha = inputs['alpha']
            self.beta = inputs['beta']
            self.seed = inputs['seed']
            self.num_topics = inputs['num_topics']
            self.verbose = inputs['verbose']
            self.model_folder = inputs['model_folder']
            self.data_source = inputs['data_source']
            self.num_iterations = inputs['num_iterations']

            kw = np.zeros((self.num_topics, self.num_vocabs))  # K x V (num_topics x num_vocabs) matrix
            kd = np.zeros((self.num_topics, self.num_docs))  # K x D (num_topics x num_docs) matrix
            z = [None] * self.num_docs  # keep topic assignment for all tokens in docs
            ztot = np.zeros(self.num_topics)  # total number of assignment for each topic k

            log_probs = []  # save the record of perplexity
            print print_timestamp() + "Start to initialize randomly..."

            for d in xrange(self.num_docs):
                zd = {}                                                  # Topic assignment for each word in document d
                for w in self.docs[d].keys():
                    c = self.docs[d][w]                                  # Number of occurrences of word w in document d
                    topics = np.random.randint(self.num_topics, size=c)  # Randomly assignment topic for each occurrence
                    zd[w] = topics.tolist()

                    # update matrix based on new topic assignments
                    for k in topics:
                        ztot[k] += 1
                        kw[k, w] += 1
                        kd[k, d] += 1

                z[d] = zd

            # update parameters
            self.paras = {'kw': kw, 'kd': kd, 'z': z, 'ztot': ztot}

            if self.verbose == 1:
                lp = self.harmonic_mean()
                log_probs.append(lp)
                print print_timestamp() + "The log likelihood of words %f" % lp
            self.paras['log_likelihood'] = log_probs
            print print_timestamp() + "Finish the initialization"
        else:
            self.num_iterations = inputs['num_iterations']
            self.model_folder = inputs['model_folder']

            print print_timestamp() + "Start to load model file: " + str(self.model_folder)
            kw = np.loadtxt(MODELS_FOLDER + self.model_folder + '/kw.dat')
            kd = np.loadtxt(MODELS_FOLDER + self.model_folder + '/kd.dat')
            ztot = np.loadtxt(MODELS_FOLDER + self.model_folder + '/ztot.dat')
            log_probs = np.loadtxt(MODELS_FOLDER + self.model_folder + '/log_likelihood.dat').tolist()

            with open(MODELS_FOLDER + self.model_folder + '/z.json') as fp:
                z = json.load(fp)  # docs with bag of words
            fp.close()

            # change key from str to integer
            for d in xrange(len(z)):
                for w in z[d].keys():
                    z[d][int(w)] = z[d].pop(w)

            self.paras = {'kw': kw, 'kd': kd, 'z': z, 'ztot': ztot, 'log_likelihood': log_probs}

            # retrieve model setting files:
            with open(MODELS_FOLDER + self.model_folder + '/settings.json') as fp:
                settings = json.load(fp)  # docs with bag of words
                self.alpha = settings['alpha']
                self.beta = settings['beta']
                self.num_topics = settings['num_topics']
                self.seed = settings['seed']
                self.verbose = settings['verbose']
                self.data_source = settings['data_source']

            # initialize seed
            np.random.seed(self.seed)
            print print_timestamp() + "Finish model loading"
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

        print print_timestamp() + "the log likelihood is %f" % log_likelihood
        return log_likelihood

    def gibbs(self):
        print print_timestamp() + "Start to run Gibbs sampling algorithm iteratively..."

        # retrieve parameters
        kw = self.paras['kw']
        kd = self.paras['kd']
        z = self.paras['z']
        ztot = self.paras['ztot']
        log_probs = self.paras['log_likelihood']

        for it in xrange(self.num_iterations):
            sys.stdout.flush()
            ts_start_time = time()

            # shuffle sequence of documents for random updating
            docs_idx = list(range(self.num_docs))
            random.shuffle(docs_idx)

            for d in docs_idx:

                for w in self.docs[d].keys():
                    cur_topics = z[d][w]        # get old topic assignments for word w in document d
                    new_topics = []

                    for k in cur_topics:
                        # remove from current assignment
                        ztot[k] -= 1
                        kw[k, w] -= 1
                        kd[k, d] -= 1

                        # compute new probability based on previous state
                        probs = (1.0 * (kw[:, w] + self.beta) / (ztot + self.num_vocabs * self.beta)) * \
                                (kd[:, d] + self.alpha)

                        # sample a topic from the distribution
                        samples = np.random.multinomial(1, probs / np.sum(probs))
                        new_topic = np.asscalar(np.where(samples == 1)[0])

                        # update matrix based on new topic assignments
                        new_topics.append(new_topic)
                        ztot[new_topic] += 1
                        kw[new_topic, w] += 1
                        kd[new_topic, d] += 1

                    z[d][w] = new_topics

            ts_end_time = time()
            if self.verbose == 1 or it == (self.num_iterations - 1):
                lp = self.harmonic_mean()
                log_probs.append(lp)
                print print_timestamp() + "After %d-th iteration, The log likelihood of words %f" % (it + 1, lp)
            print print_timestamp() + "Finish %d-th iteration. The iteration time is %f. " % (it + 1, ts_end_time -
                                                                                              ts_start_time)
        # update parameters
        self.paras = {'kw': kw, 'kd': kd, 'z': z, 'ztot': ztot, 'log_likelihood': log_probs}
        sys.stdout.flush()

    def save(self):
        print print_timestamp() + "Save model files..."
        if self.run_mode == 'start':
            # create a new folder
            ts = datetime.now().strftime('%m%d%H%M')
            self.model_folder = "LDA" + ts
            folder_path = MODELS_FOLDER + self.model_folder + '/'
            os.mkdir(folder_path)

            # save setting file
            settings = {'data_source': self.data_source, 'num_topics': self.num_topics,
                        'num_iterations': self.num_iterations, 'seed': self.seed, 'alpha': self.alpha,
                        'beta': self.beta, 'verbose': self.verbose}

        else:
            folder_path = MODELS_FOLDER + self.model_folder + '/'

            # load previous setting file
            with open(MODELS_FOLDER + self.model_folder + '/settings.json') as fp:
                settings = json.load(fp)

            # update number of iterations
            settings['num_iterations'] += self.num_iterations

        # retrieve parameters
        kw = self.paras['kw']
        kd = self.paras['kd']
        z = self.paras['z']
        ztot = self.paras['ztot']
        log_probs = self.paras['log_likelihood']
        if len(log_probs) == 0:
            log_probs.append(0)

        # save model file
        np.savetxt(folder_path + 'kw.dat', kw)
        np.savetxt(folder_path + 'kd.dat', kd)
        np.savetxt(folder_path + 'ztot.dat', ztot)
        np.savetxt(folder_path + 'log_likelihood.dat', log_probs)

        with open(folder_path + 'z.json', 'w') as fp:
            json.dump(z, fp)

        # save settings file
        with open(folder_path + 'settings.json', 'w') as fp:
            json.dump(settings, fp)

        print print_timestamp() + "Finishing model saving (%s)." % self.model_folder

    def inference(self):
        print print_timestamp() + "Start model inference for testing docs... "
        print print_timestamp() + "The number of test documents: " + str(self.num_docs)

        # retrieve parameters
        kw = self.paras['kw']
        ztot = self.paras['ztot']

        test_kd = np.zeros((self.num_topics, self.num_docs))  # K x D (num_topics x num_docs) matrix
        test_z = [None] * self.num_docs  # keep topic assignment for all tokens in docs

        print print_timestamp() + "Start to initialize randomly topic for each word..."
        for d in xrange(self.num_docs):
            zd = {}
            for w in self.docs[d].keys():
                c = self.docs[d][w]  # Number of occurrences of word w in document d
                topics = np.random.randint(self.num_topics, size=c)  # Randomly assignment a topic for each occurrence
                zd[w] = topics.tolist()

                # update matrix based on new topic assignments
                for k in topics:
                    test_kd[k, d] += 1
            test_z[d] = zd

        print print_timestamp() + "Run some iterations of Gibbs sampling for test docs..."

        # compute the perplexity for all words in the test docs
        total_perplexity = 0  # all perplexity
        num_sample = 10
        for d in xrange(self.num_docs):
            sys.stdout.flush()

            # run some iterations of Gibbs sampling for test document d
            perplexity = 0
            for it in xrange(num_sample):
                lp = 0
                n = 0
                for w in self.docs[d].keys():
                    cur_topics = test_z[d][w]  # get old topic assignments for word w in document d
                    new_topics = []
                    for k in cur_topics:
                        test_kd[k, d] -= 1

                        # compute new probability based on previous state
                        probs = (1.0 * (kw[:, w] + self.beta) /
                                 (ztot + self.num_vocabs * self.beta)) * (test_kd[:, d] + self.alpha)

                        # sample a topic from the distribution
                        samples = np.random.multinomial(1, probs / np.sum(probs))
                        new = np.asscalar(np.where(samples == 1)[0])

                        # update matrix based on new topic assignments
                        new_topics.append(new)
                        test_kd[new, d] += 1

                    test_z[d][w] = new_topics

                # Parameter estimation
                est_kd = (1.0 * (test_kd[:, d] + self.alpha)) / (np.sum(test_kd[:, d]) + self.num_topics * self.alpha)

                est_kw = np.zeros((self.num_topics, self.num_vocabs))
                for k in xrange(self.num_topics):
                    est_kw[k, :] = (1.0 * (kw[k, :] + self.beta)) / (ztot[k] + self.num_vocabs * self.beta)

                for w in self.docs[d].keys():
                    c = self.docs[d][w]
                    lp += 1.0 * c * np.log(np.inner(est_kw[:, w], est_kd))
                    n += c

                perplexity += np.exp(-lp / n)

            total_perplexity += perplexity / num_sample

        average_perplexity = total_perplexity / self.num_docs
        print print_timestamp() + "Finish inference processing, the final perplexity is %f" % average_perplexity
