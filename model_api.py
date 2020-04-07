import numpy as np
from input_fn import *
from dataset.bio.bio_tools import *
from dataset.bio.bio_dataset import *
from dataset.neuro.neuro_tools import *
from dataset.neuro.neuro_datasets import *

NUM_ITEMS = 3
NUM_WORD_TERMS = 6
NUM_TOPICS = 3


def dxtp_parameter_estimation(kw, kt, ks, ztot, alpha, beta):
    num_topics = kw.shape[0]
    num_vocabs = kw.shape[1]
    num_tools = kt.shape[1]
    num_datasets = ks.shape[1]

    est_kw = np.zeros((num_topics, num_vocabs))
    est_kt = np.zeros((num_topics, num_tools))
    est_ks = np.zeros((num_topics, num_datasets))

    for k in range(num_topics):
        est_kw[k, :] = (1.0 * (kw[k, :] + beta)) / (ztot[k] + num_vocabs * beta)
        est_kt[k, :] = (1.0 * (kt[k, :] + alpha)) / (np.sum(kt[k, :]) + num_tools * alpha)
        est_ks[k, :] = (1.0 * (ks[k, :] + alpha)) / (np.sum(ks[k, :]) + num_datasets * alpha)

    return est_kw, est_kt, est_ks


class DSTM_Model(object):
    def get_data_source(self, model_path):
        with open(model_path + '/settings.json') as fp:
            settings = json.load(fp)  # docs with bag of words
        data_source = settings['data_source']
        return data_source

    def __init__(self, model_path):
        self.model_path = model_path
        self.data_source = self.get_data_source(self.model_path)

        # load model
        inputs = input_fn('demo', self.data_source)
        self.vocabs = inputs['vocab']
        self.tools = inputs['tools']
        self.datasets = inputs['datasets']
        self.tools_info = inputs['tools_info']
        self.datasets_info = inputs['datasets_info']

        kw = np.loadtxt(model_path + 'kw.dat')
        kt = np.loadtxt(model_path + 'kt.dat')
        ks = np.loadtxt(model_path + 'ks.dat')
        ztot = np.loadtxt(model_path + 'ztot.dat')

        with open(model_path + '/settings.json') as fp:
            settings = json.load(fp)  # docs with bag of words
        alpha = settings['alpha']
        beta = settings['beta']

        self.est_kw, self.est_kt, self.est_ks = dxtp_parameter_estimation(kw, kt, ks, ztot, alpha, beta)

    def query(self, s):
        words = s.split()
        ids = []
        output = []
        for w in words:
            if w in self.vocabs:
                ids.append(self.vocabs.index(w))

        if len(ids) == 0:
            output.append({'id': 0, 'summary': '', 'tools': '', 'datasets': ''})
            print("no match topics")
            return output

        num_topics = self.est_kw.shape[0]
        probs = []
        for k in range(num_topics):
            p = np.sum(np.log(self.est_kw[k, ids]))
            probs.append(p)

        topics = np.argsort(probs)[::-1][:NUM_TOPICS]
        print("Highly matched topics is:")
        for i in range(len(topics)):
            topic_summary = ''
            tool_summary = ''
            dataset_summary = ''

            kw_idx = np.argsort(self.est_kw[topics[i], :])[::-1][:NUM_WORD_TERMS]
            kt_idx = np.argsort(self.est_kt[topics[i], :])[::-1]
            kt_idx = kt_idx[kt_idx < len(self.tools)][:NUM_ITEMS]

            ks_idx = np.argsort(self.est_ks[topics[i], :])[::-1]
            ks_idx = ks_idx[ks_idx < len(self.datasets)][:NUM_ITEMS]

            for j in range(NUM_WORD_TERMS):
                topic_summary += self.vocabs[kw_idx[j]]
                if j != NUM_WORD_TERMS - 1:
                    topic_summary += ' '

            tools = list()
            datasets = list()
            for j in range(NUM_ITEMS):
                tool_summary += self.tools[kt_idx[j]]
                if j != NUM_ITEMS - 1:
                    tool_summary += ', '
                tools.append({'name': self.tools[kt_idx[j]], 'desc': self.tools_info[self.tools[kt_idx[j]]][0],
                              'link': self.tools_info[self.tools[kt_idx[j]]][1]})

            for j in range(NUM_ITEMS):
                dataset_summary += self.datasets[ks_idx[j]]
                if j != NUM_ITEMS - 1:
                    dataset_summary += ', '
                datasets.append({'name': self.datasets[ks_idx[j]], 'desc': self.datasets_info[self.datasets[ks_idx[j]]][0],
                                  'link': self.datasets_info[self.datasets[ks_idx[j]]][1]})

            # load the query result into dictionary struct
            output.append({"id": int(topics[i]), "summary": str(topic_summary), "tools": tools,
                           "datasets": datasets})
            print('\t topic %s : %s' % (topics[i], topic_summary))
            print('\t\t Suggested tools:  %s' % tool_summary)
            print('\t\t Suggested datasets: %s' % dataset_summary)
            print

        return output


if __name__ == "__main__":
    model_folder = 'neuro_base_model/'
    model_path = os.path.join(dirname(os.path.realpath(__file__)), 'output/') + model_folder

    model = DSTM_Model(model_path)
    s = 'neuron simulation in neuroscience'
    model.query(s)

