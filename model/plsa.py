import numpy as np
from datetime import datetime
import sys
from contants import *
import json

EM_THRESHOLD = 10


def print_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": "


class PLSA(object):
    def __init__(self, inputs):
        self.docs = inputs['docs']
        self.num_docs = inputs['num_docs']
        self.num_vocabs = inputs['num_vocabs']
        self.num_topics = inputs['num_topics']
        self.num_iterations = inputs['num_iterations']
        self.model_folder = None

    def model_init(self, inputs):
        print print_timestamp() + "Start model initialization"
        self.data_source = inputs['data_source']

        self.p_z_d = np.random.uniform(0, 1, size=[self.num_docs, self.num_topics])  # P(z_k | d_i)
        self.p_w_z = np.random.uniform(0, 1, size=[self.num_topics, self.num_vocabs])  # P(w_j | z_k)
        self.p = np.zeros([self.num_docs, self.num_vocabs, self.num_topics])  # P(z_k | d_i, w_j)

        # normalize parameters
        for i in range(self.num_docs):
            for j in range(self.num_topics):
                self.p_z_d[i, j] /= np.sum(self.p_z_d[i, :])

        for i in range(self.num_topics):
            for j in range(self.num_vocabs):
                self.p_w_z[i, j] /= np.sum(self.p_w_z[i, :])

        self.X = np.zeros([self.num_docs, self.num_vocabs])
        for i in range(self.num_docs):
            keys = self.docs[i].keys()
            values = [self.docs[i][k] for k in keys]

            self.X[i][keys] = values

    def e_step(self):
        """
        Original code without considering vectorization:

        for i in range(self.num_docs):
            for j in range(self.num_vocabs):
                for k in range(self.num_topics):
                    self.p[i, j, k] = self.p_w_z[k, j] * self.p_z_d[i, k]
                    denominator += self.p[i, j, k]

                for k in range(self.num_topics):
                    self.p[i, j, k] /= (denominator + 1e-5)
        :return:
        :rtype:
        """
        for i in range(self.num_docs):
            # using vectorization
            p_z_d = np.tile(self.p_z_d[i, :], (self.p_w_z.shape[1], 1))
            self.p[i, :] = (self.p_w_z * p_z_d.T).T
            denominator = np.sum(self.p[i, :], axis=1)
            self.p[i, :] = (self.p[i, :].T / (denominator + 1e-10)).T

    def m_step(self):
        """
         Original code without considering vectorization:
         # step 1: update p(w_j | z_k)

         for k in range(self.num_topics):
             denominator = 0
             for j in range(self.num_vocabs):
                self.p_w_z[k, j] = 0

                for i in range(self.num_docs):
                    if j in self.docs[i]:
                        c = self.docs[i][j]
                        self.p_w_z[k, j] += c * self.p[i, j, k]
                denominator += self.p_w_z[k, j]

             for j in range(self.num_vocabs):
                self.p_w_z[k, j] /= (denominator + 1e-5)

         # step 2: update p(z_k | d_i)
         for i in range(self.num_docs):
            for k in range(self.num_topics):
                denominator = 0

                # for j in range(self.num_vocabs):
                #     if j in self.docs[i]:
                #         c = self.docs[i][j]
                #         self.p_z_d[i, k] += c * self.p[i, j, k]
                #         denominator += c
                self.p_z_d[i, k] = np.sum(self.X[i, :] * self.p[i, :, k])
                denominator = np.sum(self.X[i, :])

                self.p_z_d[i, k] /= (denominator + 1e-5)
        :return:
        :rtype:
        """

        # step 1: update p(w_j | z_k)
        for k in range(self.num_topics):
            p_w_z = self.X * self.p[:, :, k]
            denominator = np.sum(p_w_z)
            self.p_w_z[k, :] = np.sum(p_w_z, axis=0) / (denominator + 1e-10)

        # step 2: update p(z_k | d_i)
        for i in range(self.num_docs):
            X = np.tile(self.X[i, :], (self.p[i, :, :].shape[1], 1))
            self.p_z_d[i, :] = np.sum(X * self.p[i, :, :].T, axis=1)

            denominator = np.sum(self.X[i, :])
            self.p_z_d[i, :] /= (denominator + 1e-10)

    def compute_log_likelihood(self):
        log_likelihood = 0

        for i in range(self.num_docs):
            for j in range(self.num_vocabs):
                p = 0

                if j in self.docs[i]:
                    c = self.docs[i][j]

                    for k in range(self.num_topics):
                        p += self.p_z_d[i, k] * self.p_w_z[k, j]

                    log_likelihood += c * np.log(p)

        return log_likelihood

    def run(self):
        print print_timestamp() + "Run EM algorithm."
        old_log_likelihood = -float('inf')

        for i in range(self.num_iterations):
            self.e_step()
            self.m_step()
            new_log_likelihood = self.compute_log_likelihood()
            diff = new_log_likelihood - old_log_likelihood
            print print_timestamp() + "At %d-th iteration log likelihood is %f, and diff is %f" % (
            i, new_log_likelihood, diff)
            sys.stdout.flush()

            if abs(diff) < EM_THRESHOLD:
                break

            old_log_likelihood = new_log_likelihood

    def save(self):
        print print_timestamp() + "Save model files..."
        # create a new folder
        ts = datetime.now().strftime('%m%d%H%M')
        self.model_folder = "PLSA" + ts
        folder_path = MODELS_FOLDER + self.model_folder + '/'
        os.mkdir(folder_path)

        # save setting file
        settings = {'data_source': self.data_source, 'num_topics': self.num_topics,
                    'num_iterations': self.num_iterations}
        p = self.p.reshape((3, -1))
        np.savetxt(folder_path + 'p_z_d.dat', self.p_z_d)
        np.savetxt(folder_path + 'p_w_z.dat', self.p_w_z)
        np.savez_compressed(folder_path + 'p.npz', p)

        # save settings file
        with open(folder_path + 'settings.json', 'w') as fp:
            json.dump(settings, fp)

        print print_timestamp() + "Finishing model saving (%s)." % self.model_folder

    def inference(self, test_docs, num_iterations):
        num_test_docs = len(test_docs)
        print print_timestamp() + "Start model inference for testing docs... "
        print print_timestamp() + "The number of test documents: " + str(num_test_docs)

        test_X = np.zeros([num_test_docs, self.num_vocabs])
        for i in xrange(num_test_docs):
            keys = test_docs[i].keys()
            values = [test_docs[i][k] for k in keys]
            test_X[i][keys] = values

        # initialize parameters for test datasets
        test_p_z_d = np.random.uniform(0, 1, size=[num_test_docs, self.num_topics])  # P(z_k | d_i)
        test_p = np.zeros([num_test_docs, self.num_vocabs, self.num_topics])  # P(z_k | d_i, w_j)

        # normalize parameters
        for i in range(num_test_docs):
            for j in range(self.num_topics):
                test_p_z_d[i, j] /= np.sum(test_p_z_d[i, :])

        # run EM algorithm for test docs inference
        for d in xrange(num_iterations):
            sys.stdout.flush()

            # do e-step
            for i in xrange(num_test_docs):
                p_z_d = np.tile(test_p_z_d[i, :], (self.p_w_z.shape[1], 1))
                test_p[i, :] = (self.p_w_z * p_z_d.T).T

                denominator = np.sum(test_p[i, :], axis=1)
                test_p[i, :] = (test_p[i, :].T / (denominator + 1e-10)).T

            # do m-step just update p(z_k | d_i)
            # step 2: update p(z_k | d_i)
            for i in xrange(num_test_docs):
                X = np.tile(test_X[i, :], (test_p[i, :, :].shape[1], 1))
                test_p_z_d[i, :] = np.sum(X * test_p[i, :, :].T, axis=1)

                denominator = np.sum(test_X[i, :])
                test_p_z_d[i, :] /= (denominator + 1e-10)

        # normalize parameters
        for i in range(num_test_docs):
            for j in range(self.num_topics):
                test_p_z_d[i, j] /= np.sum(test_p_z_d[i, :])

        # compute perplexity
        perplexity = 0
        for i in range(num_test_docs):
            log_likelihood = 0
            n = 0
            for j in range(self.num_vocabs):
                p = 1e-20

                if j in test_docs[i]:
                    c = test_docs[i][j]

                    for k in range(self.num_topics):
                        p += test_p_z_d[i, k] * self.p_w_z[k, j] * test_p[i, j, k]

                    log_likelihood += c * np.log(p)
                    n += c

            perplexity += np.exp(-log_likelihood / n)

        average_perplexity = perplexity / num_test_docs
        print print_timestamp() + "Total number of test docs is % d" % num_test_docs
        print print_timestamp() + "Finish inference processing, the final perplexity is %f" % average_perplexity
